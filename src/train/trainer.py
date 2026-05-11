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
from src.data.dataset import FeatureMemmapDataset
from src.train.export import export_onnx
from src.train.model import build_model
from src.train.progress import bus

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    onnx_path: Path
    best_val_loss: float
    best_val_recall: float
    best_step: int
    history: list[dict]


def _make_loader(
    dataset: FeatureMemmapDataset,
    batch_size: int,
    workers: int,
    weighted: bool,
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
            w_pos = 0.5 / n_pos
            w_neg = 0.5 / n_neg
            weights = np.where(labels == 1, w_pos, w_neg).astype(np.float64)
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(weights),
                num_samples=len(weights),
                replacement=True,
            )
    else:
        shuffle = True

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


def _evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
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
            "fp_per_hour": 0.0,
            "threshold": 0.5,
        }
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs > 0.5).astype(np.int64)
    accuracy = float((preds == labels).mean())

    # recall at precision >= 0.95
    pos_mask = labels == 1
    neg_mask = labels == 0
    recall_at_p95 = 0.0
    chosen_threshold = 0.5
    for t in np.linspace(0.05, 0.99, 95):
        p_pos = (probs >= t) & pos_mask
        p_neg = (probs >= t) & neg_mask
        tp = int(p_pos.sum())
        fp = int(p_neg.sum())
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        if precision >= 0.95:
            r = tp / max(1, int(pos_mask.sum()))
            if r > recall_at_p95:
                recall_at_p95 = r
                chosen_threshold = float(t)

    # crude FP/hour estimate: 1 window ~ 80 ms hop. fp_count / (n_windows_neg * 0.08 / 3600)
    fp_count = int(((probs >= chosen_threshold) & neg_mask).sum())
    neg_seconds = max(1, int(neg_mask.sum())) * 0.08
    fp_per_hour = fp_count / (neg_seconds / 3600.0) if neg_seconds > 0 else 0.0

    return {
        "loss": total_loss / max(1, n_batches),
        "accuracy": accuracy,
        "recall_at_p95": recall_at_p95,
        "fp_per_hour": fp_per_hour,
        "threshold": chosen_threshold,
    }


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
    bce = nn.BCELoss()

    train_loader = _make_loader(train_ds, cfg.batch_size, workers, weighted=True)
    val_loader = _make_loader(val_ds, cfg.batch_size, workers, weighted=False)

    bus.log(
        f"Train windows: {len(train_ds):,} | Val windows: {len(val_ds):,} | "
        f"batch={cfg.batch_size} | max_steps={cfg.max_steps}"
    )

    history: list[dict] = []
    best_recall = -1.0
    best_loss = math.inf
    best_step = 0
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
        loss = bce(p, y)
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
            metrics = _evaluate(model, val_loader, device)
            metrics["step"] = step
            history.append(metrics)
            bus.metric(
                step=step,
                val_loss=metrics["loss"],
                val_accuracy=metrics["accuracy"],
                val_recall_at_p95=metrics["recall_at_p95"],
                val_fp_per_hour=metrics["fp_per_hour"],
                threshold=metrics["threshold"],
            )
            bus.log(
                f"step={step} val_loss={metrics['loss']:.4f} "
                f"recall@p95={metrics['recall_at_p95']:.3f} "
                f"FP/hr={metrics['fp_per_hour']:.2f}"
            )
            improved = metrics["recall_at_p95"] > best_recall + 1e-4
            if improved:
                best_recall = metrics["recall_at_p95"]
                best_loss = metrics["loss"]
                best_step = step
                no_improve = 0
                torch.save(model.state_dict(), out_dir / "best.pt")
            else:
                no_improve += 1
                if no_improve >= cfg.early_stop_patience:
                    bus.log(f"Early stop after {no_improve} evals without improvement", level="warning")
                    break

            if metrics["fp_per_hour"] <= cfg.target_false_positives_per_hour and metrics["recall_at_p95"] > 0.8:
                bus.log("Hit target FP/hour with strong recall. Stopping.", level="info")
                break

    # Restore best weights and export.
    best_path = out_dir / "best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    onnx_path = export_onnx(model, out_dir / "wakeword.onnx")
    bus.log(f"Exported ONNX -> {onnx_path}")

    return TrainResult(
        onnx_path=onnx_path,
        best_val_loss=best_loss,
        best_val_recall=best_recall,
        best_step=best_step,
        history=history,
    )
