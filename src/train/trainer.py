"""Training loop with progress reporting via the EventBus.

Strategy:
  - Binary cross-entropy on positive vs negative classifier-windows.
  - Class-weighted sampling (oversample positives) since wake-word data is
    extremely imbalanced - a 1.28-second window from a several-hour negative
    pool dominates by orders of magnitude.
  - Validation tracks: loss, accuracy, recall@95% precision, FP/hour.
  - Early stop when val recall stops improving for `early_stop_patience` evals.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.config_schema import TrainingConfig
from src.data.dataset import FeatureMemmapDataset, _dataset_features_at
from src.data.features import CLASSIFIER_WINDOW_EMBEDDINGS, EMBEDDING_DIM
from src.train.export import ThresholdCalibratedModel, export_onnx
from src.train.model import build_model
from src.train.progress import bus

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    onnx_path: Path
    best_val_loss: float
    best_val_recall: float
    best_val_recall_at_target_fp: float
    best_val_fp_per_hour: float
    best_threshold: float
    best_step: int
    history: list[dict]


def _make_loader(
    dataset: FeatureMemmapDataset,
    batch_size: int,
    workers: int,
    weighted: bool,
    positive_fraction: float = 0.5,
) -> DataLoader:
    sampler = None
    shuffle = False
    if weighted:
        labels = np.asarray(dataset.labels[:], dtype=np.int64)
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        if n_pos == 0 or n_neg == 0:
            shuffle = True
        else:
            positive_fraction = float(np.clip(positive_fraction, 0.01, 0.99))
            w_pos = positive_fraction / n_pos
            w_neg = (1.0 - positive_fraction) / n_neg
            weights = np.where(labels == 1, w_pos, w_neg).astype(np.float64)
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(weights),
                num_samples=len(weights),
                replacement=True,
            )
    else:
        shuffle = False

    # drop_last only on training: validation needs every sample, and small
    # smoke-test val sets would otherwise be entirely dropped.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle and sampler is None,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=weighted,
        persistent_workers=workers > 0,
    )


def _weighted_bce(
    probs: torch.Tensor,
    labels: torch.Tensor,
    cfg: TrainingConfig,
) -> torch.Tensor:
    """BCE with extra pressure on false positives.

    Wake-word models are deployed into hours of non-wake-word audio, so a
    symmetric BCE objective is too forgiving. Keep positives at weight 1 and
    make negatives expensive, with an extra bump for negatives currently
    scoring above a low threshold.
    """
    per_item = nn.functional.binary_cross_entropy(probs, labels, reduction="none")
    weights = torch.ones_like(labels)
    neg_mask = labels < 0.5
    weights = torch.where(
        neg_mask,
        torch.full_like(weights, float(cfg.negative_loss_weight)),
        weights,
    )
    hard_neg_mask = neg_mask & (probs.detach() >= float(cfg.hard_negative_threshold))
    weights = torch.where(
        hard_neg_mask,
        weights * float(cfg.hard_negative_loss_weight),
        weights,
    )
    return (per_item * weights).mean()


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainingConfig | None = None,
) -> dict[str, float]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_loss = 0.0
    n_batches = 0
    bce = nn.BCELoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.float().to(device, non_blocking=True)
            p = model(x).squeeze(-1)
            loss = bce(p, y)
            total_loss += loss.item()
            n_batches += 1
            all_probs.append(p.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())
    if not all_probs:
        return {
            "loss": math.nan,
            "accuracy": 0.0,
            "recall_at_p95": 0.0,
            "recall_at_target_fp": 0.0,
            "window_recall_at_target_fp": 0.0,
            "fp_per_hour": 0.0,
            "threshold": 0.5,
            "threshold_p95": 0.5,
            "fp_per_hour_p95": 0.0,
        }
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs > 0.5).astype(np.int64)
    accuracy = float((preds == labels).mean())

    # recall at precision >= 0.95
    pos_mask = labels == 1
    neg_mask = labels == 0
    source_ids = getattr(loader.dataset, "source_ids", None)
    pos_clip_scores: np.ndarray | None = None
    if source_ids is not None:
        source_ids_arr = np.asarray(source_ids[: len(probs)], dtype=np.int64)
        valid_pos_sources = source_ids_arr[pos_mask] >= 0
        if valid_pos_sources.any():
            pos_sources = source_ids_arr[pos_mask][valid_pos_sources]
            pos_scores = probs[pos_mask][valid_pos_sources]
            order = np.argsort(pos_sources)
            sorted_sources = pos_sources[order]
            sorted_scores = pos_scores[order]
            starts = np.r_[0, np.flatnonzero(np.diff(sorted_sources)) + 1]
            pos_clip_scores = np.maximum.reduceat(sorted_scores, starts)

    def positive_recall(threshold: float) -> float:
        if pos_clip_scores is not None and pos_clip_scores.size > 0:
            return float((pos_clip_scores >= threshold).sum() / pos_clip_scores.size)
        return float(((probs >= threshold) & pos_mask).sum() / max(1, int(pos_mask.sum())))

    recall_at_p95 = 0.0
    p95_threshold = 0.5
    if probs.size:
        order = np.argsort(probs)[::-1]
        sorted_scores = probs[order]
        sorted_is_pos = pos_mask[order]
        tp_cum = np.cumsum(sorted_is_pos, dtype=np.int64)
        fp_cum = np.cumsum(~sorted_is_pos, dtype=np.int64)
        denom = tp_cum + fp_cum
        precision = np.divide(tp_cum, denom, out=np.zeros_like(tp_cum, dtype=np.float64), where=denom > 0)
        valid = np.flatnonzero((precision >= 0.95) & (tp_cum > 0))
        if valid.size:
            # Pick the threshold that maximizes positive recall at >=95%
            # precision. Positive recall is clip-level when source IDs exist.
            best_idx = max(valid, key=lambda idx: positive_recall(float(sorted_scores[idx])))
            p95_threshold = float(sorted_scores[best_idx])
            recall_at_p95 = positive_recall(p95_threshold)

    neg_hours = max(1, int(neg_mask.sum())) * 0.08 / 3600.0
    fp_count_p95 = int(((probs >= p95_threshold) & neg_mask).sum())
    fp_per_hour_p95 = fp_count_p95 / neg_hours if neg_hours > 0 else 0.0

    # Select the operating threshold by the configured false-positive target.
    # This is the threshold that will be baked into the exported ONNX so a
    # downstream score threshold of 0.5 maps to the validated operating point.
    if cfg is not None:
        allowed_fp = int(math.floor(float(cfg.target_false_positives_per_hour) * neg_hours))
        neg_scores = np.sort(probs[neg_mask])[::-1]
        if neg_scores.size == 0:
            chosen_threshold = p95_threshold
        elif allowed_fp < neg_scores.size:
            chosen_threshold = float(
                np.nextafter(np.float32(neg_scores[allowed_fp]), np.float32(np.inf))
            )
        else:
            chosen_threshold = 0.0
    else:
        chosen_threshold = p95_threshold

    fp_count = int(((probs >= chosen_threshold) & neg_mask).sum())
    neg_seconds = max(1, int(neg_mask.sum())) * 0.08
    fp_per_hour = fp_count / (neg_seconds / 3600.0) if neg_seconds > 0 else 0.0
    window_recall_at_target_fp = float(
        ((probs >= chosen_threshold) & pos_mask).sum() / max(1, int(pos_mask.sum()))
    )
    recall_at_target_fp = positive_recall(chosen_threshold)

    return {
        "loss": total_loss / max(1, n_batches),
        "accuracy": accuracy,
        "recall_at_p95": recall_at_p95,
        "recall_at_target_fp": recall_at_target_fp,
        "window_recall_at_target_fp": window_recall_at_target_fp,
        "fp_per_hour": fp_per_hour,
        "threshold": chosen_threshold,
        "threshold_p95": p95_threshold,
        "fp_per_hour_p95": fp_per_hour_p95,
    }


def _is_exportable(metrics: dict[str, float], cfg: TrainingConfig) -> bool:
    return (
        metrics.get("recall_at_target_fp", 0.0) >= cfg.min_recall_at_target_fp_for_export
        and metrics["fp_per_hour"] <= cfg.target_false_positives_per_hour
        and not math.isnan(metrics["loss"])
    )


def _checkpoint_score(metrics: dict[str, float], cfg: TrainingConfig) -> tuple:
    """Rank validation checkpoints.

    Exportability is the hard boundary. Within exportable checkpoints, prefer
    lower false positives first, then stronger recall, then lower loss. For
    diagnostics while training, non-exportable checkpoints are still ranked so
    logs can explain what the closest miss was.
    """
    loss_score = -float(metrics["loss"]) if not math.isnan(metrics["loss"]) else -math.inf
    if _is_exportable(metrics, cfg):
        return (
            1,
            float(metrics.get("recall_at_target_fp", 0.0)),
            -float(metrics["fp_per_hour"]),
            float(metrics.get("recall_at_p95", 0.0)),
            loss_score,
        )
    return (
        0,
        float(metrics.get("recall_at_target_fp", 0.0)),
        float(metrics.get("recall_at_p95", 0.0)),
        -float(metrics["fp_per_hour"]),
        loss_score,
    )


def _verify_onnx_parity(model: nn.Module, onnx_path: Path, tolerance: float = 1e-4) -> float:
    """Verify exported ONNX matches the PyTorch classifier head."""
    import onnxruntime as ort

    model = model.eval().cpu()
    rng = np.random.default_rng(0)
    probes = [
        np.zeros((1, CLASSIFIER_WINDOW_EMBEDDINGS, EMBEDDING_DIM), dtype=np.float32),
        rng.standard_normal((1, CLASSIFIER_WINDOW_EMBEDDINGS, EMBEDDING_DIM)).astype(np.float32),
        (
            0.01
            * rng.standard_normal((1, CLASSIFIER_WINDOW_EMBEDDINGS, EMBEDDING_DIM))
        ).astype(np.float32),
    ]

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    max_diff = 0.0
    with torch.no_grad():
        for x in probes:
            torch_out = model(torch.from_numpy(x)).detach().cpu().numpy().reshape(-1)
            onnx_out = session.run(None, {input_name: x})[0].reshape(-1)
            max_diff = max(max_diff, float(np.max(np.abs(torch_out - onnx_out))))

    if max_diff > tolerance:
        raise RuntimeError(
            f"ONNX export parity check failed: max_abs_diff={max_diff:.6g} "
            f"> tolerance={tolerance:.6g}"
        )
    return max_diff


def _mine_hard_negatives(
    model: nn.Module,
    dataset: FeatureMemmapDataset,
    device: torch.device,
    top_k: int,
    batch_size: int,
) -> np.ndarray:
    """Return dataset indices for the highest-scoring negative windows."""
    labels = np.asarray(dataset.labels[:], dtype=np.int64)
    neg_indices = np.flatnonzero(labels == 0)
    if neg_indices.size == 0 or top_k <= 0:
        return np.zeros((0,), dtype=np.int64)

    top_k = min(int(top_k), int(neg_indices.size))
    best_scores = np.full(top_k, -np.inf, dtype=np.float32)
    best_indices = np.full(top_k, -1, dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for start in range(0, neg_indices.size, batch_size):
            ndcs = neg_indices[start : start + batch_size]
            x = torch.from_numpy(_dataset_features_at(dataset, ndcs)).to(device)
            scores = model(x).squeeze(-1).detach().cpu().numpy().astype(np.float32)
            merged_scores = np.concatenate([best_scores, scores])
            merged_indices = np.concatenate([best_indices, ndcs.astype(np.int64)])
            if merged_scores.size > top_k:
                keep = np.argpartition(merged_scores, -top_k)[-top_k:]
                best_scores = merged_scores[keep]
                best_indices = merged_indices[keep]
            else:
                best_scores = merged_scores
                best_indices = merged_indices

    order = np.argsort(best_scores)[::-1]
    return best_indices[order]


def _fine_tune_hard_negatives(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainingConfig,
    train_ds: FeatureMemmapDataset,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    history: list[dict],
    best_score: tuple | None,
    best_metrics: dict[str, float] | None,
    best_step: int,
    best_exportable: bool,
    cancel_flag=None,
) -> tuple[tuple | None, dict[str, float] | None, int, bool]:
    labels = np.asarray(train_ds.labels[:], dtype=np.int64)
    pos_indices = np.flatnonzero(labels == 1)
    if pos_indices.size == 0:
        return best_score, best_metrics, best_step, best_exportable

    bus.log(
        f"Mining top {cfg.hard_negative_mining_top_k:,} hard negatives "
        "from training shard..."
    )
    hard_neg_indices = _mine_hard_negatives(
        model,
        train_ds,
        device,
        top_k=cfg.hard_negative_mining_top_k,
        batch_size=max(4096, cfg.batch_size * 2),
    )
    if hard_neg_indices.size == 0:
        bus.log("No hard negatives found for fine-tuning.", level="warning")
        return best_score, best_metrics, best_step, best_exportable

    bus.log(
        f"Fine-tuning on {hard_neg_indices.size:,} mined hard negatives "
        f"for up to {cfg.hard_negative_finetune_steps:,} steps"
    )
    rng = np.random.default_rng(cfg.seed + 17)
    pos_per_batch = max(
        1,
        min(
            cfg.batch_size - 1,
            int(cfg.batch_size * cfg.hard_negative_finetune_positive_fraction),
        ),
    )
    neg_per_batch = cfg.batch_size - pos_per_batch
    base_step = int(best_step)

    for ft_step in range(1, cfg.hard_negative_finetune_steps + 1):
        if cancel_flag is not None and cancel_flag.is_set():
            break

        pidx = rng.choice(pos_indices, size=pos_per_batch, replace=True)
        nidx = rng.choice(hard_neg_indices, size=neg_per_batch, replace=True)
        ndcs = np.concatenate([pidx, nidx])
        y_np = np.concatenate(
            [
                np.ones(pos_per_batch, dtype=np.float32),
                np.zeros(neg_per_batch, dtype=np.float32),
            ]
        )
        order = rng.permutation(ndcs.size)
        ndcs = ndcs[order]
        y_np = y_np[order]

        x = torch.from_numpy(_dataset_features_at(train_ds, ndcs)).to(device)
        y = torch.from_numpy(y_np).to(device)
        model.train()
        optimizer.zero_grad()
        probs = model(x).squeeze(-1)
        loss = _weighted_bce(probs, y, cfg)
        loss.backward()
        optimizer.step()

        if ft_step % cfg.val_every_n_steps == 0 or ft_step == cfg.hard_negative_finetune_steps:
            metrics = _evaluate(model, val_loader, device, cfg)
            global_step = base_step + ft_step
            metrics["step"] = global_step
            metrics["phase"] = "hard_negative_finetune"
            history.append(metrics)
            bus.metric(
                step=global_step,
                val_loss=metrics["loss"],
                val_accuracy=metrics["accuracy"],
                val_recall_at_p95=metrics["recall_at_p95"],
                val_recall_at_target_fp=metrics["recall_at_target_fp"],
                val_fp_per_hour=metrics["fp_per_hour"],
                threshold=metrics["threshold"],
            )
            bus.log(
                f"hard-neg step={ft_step} val_loss={metrics['loss']:.4f} "
                f"recall@p95={metrics['recall_at_p95']:.3f} "
                f"recall@targetFP={metrics['recall_at_target_fp']:.3f} "
                f"FP/hr={metrics['fp_per_hour']:.2f} "
                f"threshold={metrics['threshold']:.6f}"
            )
            score = _checkpoint_score(metrics, cfg)
            if best_score is None or score > best_score:
                best_score = score
                best_metrics = metrics.copy()
                best_step = global_step
                best_exportable = _is_exportable(metrics, cfg)
                torch.save(model.state_dict(), out_dir / "best_candidate.pt")
                if best_exportable:
                    torch.save(model.state_dict(), out_dir / "best.pt")
                    bus.log(
                        "New exportable hard-negative checkpoint: "
                        f"step={global_step} "
                        f"recall@targetFP={metrics['recall_at_target_fp']:.3f} "
                        f"FP/hr={metrics['fp_per_hour']:.2f}"
                    )
                    break

    return best_score, best_metrics, best_step, best_exportable


def train(
    cfg: TrainingConfig,
    train_ds: FeatureMemmapDataset,
    val_ds: FeatureMemmapDataset,
    out_dir: Path,
    workers: int = 4,
    cancel_flag=None,
) -> TrainResult:
    """Run the training loop. `cancel_flag.is_set()` aborts gracefully."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bus.log(f"Device: {device}")

    model = build_model(cfg.model_type, cfg.layer_dim, cfg.n_blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    train_loader = _make_loader(
        train_ds,
        cfg.batch_size,
        workers,
        weighted=True,
        positive_fraction=cfg.positive_sample_fraction,
    )
    val_loader = _make_loader(val_ds, cfg.batch_size, workers, weighted=False)

    bus.log(
        f"Train windows: {len(train_ds):,} | Val windows: {len(val_ds):,} | "
        f"batch={cfg.batch_size} | max_steps={cfg.max_steps}"
    )

    history: list[dict] = []
    best_metrics: dict[str, float] | None = None
    best_score: tuple | None = None
    best_step = 0
    best_exportable = False
    no_improve = 0

    step = 0
    t0 = time.monotonic()
    train_iter = iter(train_loader)
    while step < cfg.max_steps:
        if cancel_flag is not None and cancel_flag.is_set():
            bus.log("Training cancelled.", level="warning")
            break

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device, non_blocking=True)
        y = y.float().to(device, non_blocking=True)

        model.train()
        optimizer.zero_grad()
        p = model(x).squeeze(-1)
        loss = _weighted_bce(p, y, cfg)
        loss.backward()
        optimizer.step()

        step += 1
        if step % 25 == 0:
            elapsed = time.monotonic() - t0
            steps_per_sec = step / max(1e-6, elapsed)
            eta = (cfg.max_steps - step) / max(1e-6, steps_per_sec)
            bus.metric(
                step=step,
                max_steps=cfg.max_steps,
                train_loss=float(loss.item()),
                steps_per_sec=steps_per_sec,
                eta_seconds=eta,
            )
            bus.progress("train", step / cfg.max_steps, detail=f"step {step}/{cfg.max_steps}")

        if step % cfg.val_every_n_steps == 0 or step == cfg.max_steps:
            metrics = _evaluate(model, val_loader, device, cfg)
            metrics["step"] = step
            history.append(metrics)
            bus.metric(
                step=step,
                val_loss=metrics["loss"],
                val_accuracy=metrics["accuracy"],
                val_recall_at_p95=metrics["recall_at_p95"],
                val_recall_at_target_fp=metrics["recall_at_target_fp"],
                val_fp_per_hour=metrics["fp_per_hour"],
                threshold=metrics["threshold"],
            )
            bus.log(
                f"step={step} val_loss={metrics['loss']:.4f} "
                f"recall@p95={metrics['recall_at_p95']:.3f} "
                f"recall@targetFP={metrics['recall_at_target_fp']:.3f} "
                f"FP/hr={metrics['fp_per_hour']:.2f} "
                f"threshold={metrics['threshold']:.5f}"
            )
            score = _checkpoint_score(metrics, cfg)
            improved = best_score is None or score > best_score
            if improved:
                best_score = score
                best_metrics = metrics.copy()
                best_step = step
                best_exportable = _is_exportable(metrics, cfg)
                no_improve = 0
                torch.save(model.state_dict(), out_dir / "best_candidate.pt")
                if best_exportable:
                    torch.save(model.state_dict(), out_dir / "best.pt")
                    bus.log(
                        "New exportable checkpoint: "
                        f"step={step} recall@targetFP={metrics['recall_at_target_fp']:.3f} "
                        f"FP/hr={metrics['fp_per_hour']:.2f}"
                    )
            else:
                no_improve += 1
                if no_improve >= cfg.early_stop_patience and step >= cfg.early_stop_min_steps:
                    bus.log(
                        f"Early stop after {no_improve} evals without improvement "
                        f"(min_steps={cfg.early_stop_min_steps})",
                        level="warning",
                    )
                    break

            if _is_exportable(metrics, cfg):
                bus.log("Hit target FP/hour with strong recall. Stopping.", level="info")
                break

    # Restore best weights and export.
    best_path = out_dir / "best.pt"
    if not best_exportable and (out_dir / "best_candidate.pt").exists():
        model.load_state_dict(torch.load(out_dir / "best_candidate.pt", map_location=device))
        (
            best_score,
            best_metrics,
            best_step,
            best_exportable,
        ) = _fine_tune_hard_negatives(
            model,
            optimizer,
            cfg,
            train_ds,
            val_loader,
            device,
            out_dir,
            history,
            best_score,
            best_metrics,
            best_step,
            best_exportable,
            cancel_flag=cancel_flag,
        )

    if not best_exportable or not best_path.exists() or best_metrics is None:
        candidate = best_metrics or {
            "loss": math.nan,
            "recall_at_p95": 0.0,
            "fp_per_hour": math.inf,
            "threshold": 0.5,
        }
        msg = (
            "No exportable checkpoint found. "
            f"Best candidate: step={best_step}, "
            f"recall@targetFP={candidate.get('recall_at_target_fp', 0.0):.3f} "
            f"(required >= {cfg.min_recall_at_target_fp_for_export:.3f}), "
            f"FP/hr={candidate['fp_per_hour']:.2f} "
            f"(required <= {cfg.target_false_positives_per_hour:.2f}). "
            "Refusing to export/publish an unusable wake-word model."
        )
        bus.log(msg, level="error")
        raise RuntimeError(msg)

    model.load_state_dict(torch.load(best_path, map_location=device))

    onnx_path = export_onnx(
        model,
        out_dir / "wakeword.onnx",
        score_threshold=best_metrics["threshold"],
    )
    parity_model = ThresholdCalibratedModel(model, best_metrics["threshold"]).eval().cpu()
    parity_diff = _verify_onnx_parity(parity_model, onnx_path)
    bus.log(
        f"Exported ONNX -> {onnx_path} "
        f"(threshold={best_metrics['threshold']:.6f}, "
        f"recall@p95={best_metrics['recall_at_p95']:.3f}, "
        f"recall@targetFP={best_metrics['recall_at_target_fp']:.3f}, "
        f"FP/hr={best_metrics['fp_per_hour']:.2f}, "
        f"onnx_max_abs_diff={parity_diff:.2e})"
    )

    return TrainResult(
        onnx_path=onnx_path,
        best_val_loss=best_metrics["loss"],
        best_val_recall=best_metrics["recall_at_p95"],
        best_val_recall_at_target_fp=best_metrics["recall_at_target_fp"],
        best_val_fp_per_hour=best_metrics["fp_per_hour"],
        best_threshold=best_metrics["threshold"],
        best_step=best_step,
        history=history,
    )
