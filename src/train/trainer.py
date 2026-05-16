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
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.config_schema import TrainingConfig
from src.data.dataset import FeatureMemmapDataset, _dataset_features_at
from src.data.features import CLASSIFIER_WINDOW_EMBEDDINGS, EMBEDDING_DIM
from src.train.export import ThresholdCalibratedModel, export_onnx
from src.train.model import build_model
from src.train.progress import bus

logger = logging.getLogger(__name__)

# The classifier is evaluated every 80 ms, but a wake-word false positive is an
# event, not every overlapping high-scoring window inside that event. Count
# nearby high windows from the same source as one trigger.
FP_EVENT_REFRACTORY_WINDOWS = 12


@dataclass
class TrainResult:
    onnx_path: Path
    best_val_loss: float
    best_val_recall: float
    best_val_recall_at_target_fp: float
    best_val_fp_per_hour: float
    best_val_recall_at_0_5: float
    best_val_fp_per_hour_at_0_5: float
    best_positive_median_score: float
    best_positive_p10_score: float
    best_curve_recall: float
    best_curve_median_peak: float
    best_curve_p10_peak: float
    best_curve_median_frames: float
    best_curve_median_span_ms: float
    best_curve_confirmation_rate: float
    best_tablet_curve_recall: float
    best_tablet_curve_median_peak: float
    best_tablet_curve_p10_peak: float
    best_tablet_curve_median_frames: float
    best_tablet_curve_median_span_ms: float
    best_tablet_curve_confirmation_rate: float
    best_threshold: float
    best_step: int
    history: list[dict]


@dataclass
class CurveValidationSet:
    """Sliding full-audio positive validation windows grouped by source clip."""

    features_path: Path
    source_ids_path: Path
    n_windows: int
    n_clips: int
    tablet_features_path: Path | None = None
    tablet_source_ids_path: Path | None = None
    tablet_n_windows: int = 0
    tablet_n_clips: int = 0


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


def _cosine_warmup_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_fraction: float,
    hold_fraction: float,
) -> float:
    warmup_steps = int(total_steps * max(0.0, warmup_fraction))
    hold_steps = int(total_steps * max(0.0, hold_fraction))
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    if step < warmup_steps + hold_steps:
        return base_lr
    decay_steps = max(1, total_steps - warmup_steps - hold_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps - hold_steps) / decay_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _scheduled_negative_weight(cfg: TrainingConfig, step: int) -> float:
    progress = min(1.0, max(0.0, step / max(1, int(cfg.max_steps))))
    return 1.0 + (float(cfg.max_negative_loss_weight) - 1.0) * progress


def _weighted_loss(
    probs: torch.Tensor,
    labels: torch.Tensor,
    cfg: TrainingConfig,
    negative_weight: float | None = None,
) -> torch.Tensor:
    """Reference-style focal/BCE loss with scheduled negative pressure.

    Wake-word models are deployed into hours of non-wake-word audio, so a
    symmetric BCE objective is too forgiving. Focal loss keeps attention on
    decision-boundary examples, label smoothing prevents extreme calibration,
    and the negative weight ramps up over training instead of slamming the
    model into "everything is negative" from step 1.
    """
    raw_labels = labels
    if cfg.label_smoothing > 0:
        labels = labels * (1.0 - float(cfg.label_smoothing)) + 0.5 * float(cfg.label_smoothing)

    per_item = F.binary_cross_entropy(probs, labels, reduction="none")
    if cfg.use_focal_loss:
        p_t = probs * labels + (1.0 - probs) * (1.0 - labels)
        per_item = ((1.0 - p_t) ** float(cfg.focal_gamma)) * per_item

    if negative_weight is None:
        negative_weight = float(cfg.negative_loss_weight)
    weights = torch.ones_like(labels)
    neg_mask = labels < 0.5
    weights = torch.where(
        neg_mask,
        torch.full_like(weights, float(negative_weight)),
        weights,
    )
    hard_neg_mask = neg_mask & (probs.detach() >= float(cfg.hard_negative_threshold))
    weights = torch.where(
        hard_neg_mask,
        weights * float(cfg.hard_negative_loss_weight),
        weights,
    )
    loss = (per_item * weights).sum() / torch.clamp(weights.sum(), min=1.0)

    # Confidence/separation terms are deliberately separate from BCE. They let
    # a run ask for "true wake words should peak high" without solving that by
    # globally weakening negative pressure and reintroducing false positives.
    pos_weight = torch.clamp(raw_labels, min=0.0, max=1.0)
    neg_weight_soft = torch.clamp(1.0 - raw_labels, min=0.0, max=1.0)

    if cfg.positive_confidence_weight > 0 and pos_weight.sum() > 0:
        pos_gap = F.relu(float(cfg.positive_confidence_target) - probs)
        pos_loss = (pos_gap.square() * pos_weight).sum() / torch.clamp(pos_weight.sum(), min=1.0)
        loss = loss + float(cfg.positive_confidence_weight) * pos_loss

    if cfg.negative_confidence_weight > 0 and neg_weight_soft.sum() > 0:
        neg_gap = F.relu(probs - float(cfg.negative_confidence_target))
        neg_loss = (neg_gap.square() * neg_weight_soft).sum() / torch.clamp(neg_weight_soft.sum(), min=1.0)
        loss = loss + float(cfg.negative_confidence_weight) * neg_loss

    if cfg.separation_loss_weight > 0:
        pos_probs = probs[raw_labels >= 0.75]
        neg_probs = probs[raw_labels <= 0.25]
        if pos_probs.numel() > 0 and neg_probs.numel() > 0:
            k_pos = min(int(cfg.separation_top_k), pos_probs.numel())
            k_neg = min(int(cfg.separation_top_k), neg_probs.numel())
            weakest_pos = torch.topk(pos_probs, k=k_pos, largest=False).values
            hardest_neg = torch.topk(neg_probs, k=k_neg, largest=True).values
            pairwise_gap = weakest_pos[:, None] - hardest_neg[None, :]
            sep_loss = F.relu(float(cfg.separation_margin) - pairwise_gap).mean()
            loss = loss + float(cfg.separation_loss_weight) * sep_loss

    return loss


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
            "fp_per_hour_at_0_5": 0.0,
            "recall_at_0_5": 0.0,
            "threshold": 0.5,
            "threshold_p95": 0.5,
            "fp_per_hour_p95": 0.0,
            "positive_median_score": 0.0,
            "positive_p10_score": 0.0,
            "positive_p90_score": 0.0,
        }
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs > 0.5).astype(np.int64)
    accuracy = float((preds == labels).mean())

    # recall at precision >= 0.95
    pos_mask = labels == 1
    neg_mask = labels == 0
    source_ids = getattr(loader.dataset, "source_ids", None)
    source_ids_arr = (
        np.asarray(source_ids[: len(probs)], dtype=np.int64)
        if source_ids is not None
        else np.full(len(probs), -1, dtype=np.int64)
    )
    pos_clip_scores: np.ndarray | None = None
    if source_ids_arr.size:
        valid_pos_sources = source_ids_arr[pos_mask] >= 0
        if valid_pos_sources.any():
            pos_sources = source_ids_arr[pos_mask][valid_pos_sources]
            pos_scores = probs[pos_mask][valid_pos_sources]
            order = np.argsort(pos_sources)
            sorted_sources = pos_sources[order]
            sorted_scores = pos_scores[order]
            starts = np.r_[0, np.flatnonzero(np.diff(sorted_sources)) + 1]
            pos_clip_scores = np.maximum.reduceat(sorted_scores, starts)

    neg_scores_all = probs[neg_mask]
    neg_sources_all = source_ids_arr[neg_mask]

    def false_positive_events(threshold: float) -> int:
        above = np.flatnonzero(neg_scores_all >= threshold)
        if above.size == 0:
            return 0
        src = neg_sources_all[above]
        starts = np.ones(above.size, dtype=bool)
        if above.size > 1:
            starts[1:] = (
                (src[1:] != src[:-1])
                | ((above[1:] - above[:-1]) > FP_EVENT_REFRACTORY_WINDOWS)
            )
        return int(starts.sum())

    def threshold_for_event_fp_budget(allowed_fp: int) -> float:
        if neg_scores_all.size == 0:
            return p95_threshold
        if false_positive_events(0.0) <= allowed_fp:
            return 0.0
        # Find the lowest threshold whose event count stays within budget.
        lo = 0.0
        hi = float(np.nextafter(np.float32(np.max(neg_scores_all)), np.float32(np.inf)))
        for _ in range(24):
            mid = (lo + hi) / 2.0
            if false_positive_events(mid) > allowed_fp:
                lo = mid
            else:
                hi = mid
        return hi

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

    neg_seconds = max(1, int(neg_mask.sum())) * 0.08
    neg_hours = neg_seconds / 3600.0
    fp_count_p95 = false_positive_events(p95_threshold)
    fp_per_hour_p95 = fp_count_p95 / neg_hours if neg_hours > 0 else 0.0
    fixed_threshold = 0.5
    fp_count_at_0_5 = false_positive_events(fixed_threshold)
    fp_per_hour_at_0_5 = fp_count_at_0_5 / neg_hours if neg_hours > 0 else 0.0
    recall_at_0_5 = positive_recall(fixed_threshold)
    if pos_clip_scores is not None and pos_clip_scores.size > 0:
        positive_median_score = float(np.median(pos_clip_scores))
        positive_p10_score = float(np.quantile(pos_clip_scores, 0.10))
        positive_p90_score = float(np.quantile(pos_clip_scores, 0.90))
    else:
        pos_scores = probs[pos_mask]
        positive_median_score = float(np.median(pos_scores)) if pos_scores.size else 0.0
        positive_p10_score = float(np.quantile(pos_scores, 0.10)) if pos_scores.size else 0.0
        positive_p90_score = float(np.quantile(pos_scores, 0.90)) if pos_scores.size else 0.0

    # Select the operating threshold by the configured false-positive target.
    # This is the threshold that will be baked into the exported ONNX so a
    # downstream score threshold of 0.5 maps to the validated operating point.
    if cfg is not None:
        allowed_fp = int(math.floor(float(cfg.target_false_positives_per_hour) * neg_hours))
        chosen_threshold = threshold_for_event_fp_budget(allowed_fp)
    else:
        chosen_threshold = p95_threshold

    fp_count = false_positive_events(chosen_threshold)
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
        "fp_per_hour_at_0_5": fp_per_hour_at_0_5,
        "recall_at_0_5": recall_at_0_5,
        "threshold": chosen_threshold,
        "threshold_p95": p95_threshold,
        "fp_per_hour_p95": fp_per_hour_p95,
        "positive_median_score": positive_median_score,
        "positive_p10_score": positive_p10_score,
        "positive_p90_score": positive_p90_score,
    }


def _calibrate_scores(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Mirror the exported ONNX score calibration for checkpoint validation."""
    scores = np.clip(scores.astype(np.float32, copy=False), 1e-6, 1.0 - 1e-6)
    threshold = float(np.clip(threshold, 1e-6, 1.0 - 1e-6))
    logits = np.log(scores / (1.0 - scores))
    threshold_logit = math.log(threshold / (1.0 - threshold))
    return (1.0 / (1.0 + np.exp(-(logits - threshold_logit)))).astype(np.float32)


def _empty_curve_metrics(enabled: bool) -> dict[str, float]:
    if not enabled:
        return {
            "curve_recall": 1.0,
            "curve_median_peak": 1.0,
            "curve_p10_peak": 1.0,
            "curve_median_frames": 1_000_000.0,
            "curve_median_span_ms": 1_000_000.0,
            "curve_confirmation_rate": 1.0,
        }
    return {
        "curve_recall": 0.0,
        "curve_median_peak": 0.0,
        "curve_p10_peak": 0.0,
        "curve_median_frames": 0.0,
        "curve_median_span_ms": 0.0,
        "curve_confirmation_rate": 0.0,
    }


def _score_curve_features(
    *,
    model: nn.Module,
    features_path: Path,
    source_ids_path: Path,
    n_windows: int,
    n_clips: int,
    device: torch.device,
    cfg: TrainingConfig,
    threshold: float,
) -> dict[str, float]:
    if n_windows <= 0 or n_clips <= 0:
        return _empty_curve_metrics(enabled=True)

    features = np.memmap(
        features_path,
        dtype=np.float32,
        mode="r",
        shape=(n_windows, CLASSIFIER_WINDOW_EMBEDDINGS, EMBEDDING_DIM),
    )
    source_ids = np.load(source_ids_path, mmap_mode="r")
    if int(source_ids.shape[0]) != int(n_windows):
        return _empty_curve_metrics(enabled=True)

    scores: list[np.ndarray] = []
    batch_size = max(4096, int(cfg.batch_size) * 2)
    model.eval()
    with torch.no_grad():
        for start in range(0, n_windows, batch_size):
            end = min(n_windows, start + batch_size)
            x = torch.from_numpy(np.array(features[start:end], dtype=np.float32, copy=True)).to(device)
            p = model(x).squeeze(-1).detach().cpu().numpy().astype(np.float32)
            scores.append(p)
    raw_scores = np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)
    if raw_scores.size == 0:
        return _empty_curve_metrics(enabled=True)

    runtime_scores = _calibrate_scores(raw_scores, threshold)
    order = np.argsort(source_ids)
    sorted_sources = np.asarray(source_ids[order], dtype=np.int64)
    sorted_scores = runtime_scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_sources)) + 1]

    peaks: list[float] = []
    frames: list[int] = []
    spans_ms: list[float] = []
    confirmations: list[float] = []
    min_gap_frames = max(1, int(math.ceil(float(cfg.curve_confirmation_min_gap_ms) / 80.0)))
    for start_idx, end_idx in zip(starts, np.r_[starts[1:], sorted_scores.size]):
        clip_scores = sorted_scores[start_idx:end_idx]
        if clip_scores.size == 0:
            continue
        peaks.append(float(np.max(clip_scores)))
        above = np.flatnonzero(clip_scores >= 0.5)
        frames.append(int(above.size))
        if above.size:
            spans_ms.append(float((above[-1] - above[0] + 1) * 80.0))
        else:
            spans_ms.append(0.0)
        confirmed = False
        if above.size >= 2:
            for i in range(above.size - 1):
                if int(above[-1] - above[i]) >= min_gap_frames:
                    confirmed = True
                    break
        confirmations.append(1.0 if confirmed else 0.0)

    if not peaks:
        return _empty_curve_metrics(enabled=True)

    peaks_arr = np.asarray(peaks, dtype=np.float32)
    frames_arr = np.asarray(frames, dtype=np.float32)
    spans_arr = np.asarray(spans_ms, dtype=np.float32)
    confirmations_arr = np.asarray(confirmations, dtype=np.float32)
    return {
        "curve_recall": float((peaks_arr >= 0.5).sum() / peaks_arr.size),
        "curve_median_peak": float(np.median(peaks_arr)),
        "curve_p10_peak": float(np.quantile(peaks_arr, 0.10)),
        "curve_median_frames": float(np.median(frames_arr)),
        "curve_median_span_ms": float(np.median(spans_arr)),
        "curve_confirmation_rate": float(confirmations_arr.mean()),
    }


def _evaluate_positive_curves(
    model: nn.Module,
    curve_val: CurveValidationSet | None,
    device: torch.device,
    cfg: TrainingConfig,
    threshold: float,
) -> dict[str, float]:
    """Score held-out positive WAVs as full sliding curves.

    Window-level validation can pass a model that produces a single narrow
    spike. This evaluates the runtime-shaped curve after applying the same
    threshold calibration that will be baked into the exported ONNX.
    """
    if not cfg.use_positive_curve_validation:
        metrics = _empty_curve_metrics(enabled=False)
        metrics.update({f"tablet_{k}": v for k, v in _empty_curve_metrics(enabled=False).items()})
        return metrics
    if curve_val is None or curve_val.n_windows <= 0 or curve_val.n_clips <= 0:
        metrics = _empty_curve_metrics(enabled=True)
        metrics.update({f"tablet_{k}": v for k, v in _empty_curve_metrics(enabled=True).items()})
        return metrics

    metrics = _score_curve_features(
        model=model,
        features_path=curve_val.features_path,
        source_ids_path=curve_val.source_ids_path,
        n_windows=curve_val.n_windows,
        n_clips=curve_val.n_clips,
        device=device,
        cfg=cfg,
        threshold=threshold,
    )
    if (
        cfg.use_tablet_curve_validation
        and curve_val.tablet_features_path is not None
        and curve_val.tablet_source_ids_path is not None
        and curve_val.tablet_n_windows > 0
        and curve_val.tablet_n_clips > 0
    ):
        tablet_metrics = _score_curve_features(
            model=model,
            features_path=curve_val.tablet_features_path,
            source_ids_path=curve_val.tablet_source_ids_path,
            n_windows=curve_val.tablet_n_windows,
            n_clips=curve_val.tablet_n_clips,
            device=device,
            cfg=cfg,
            threshold=threshold,
        )
    else:
        tablet_metrics = _empty_curve_metrics(enabled=cfg.use_tablet_curve_validation)
    metrics.update({f"tablet_{k}": v for k, v in tablet_metrics.items()})
    return metrics


def _is_exportable(metrics: dict[str, float], cfg: TrainingConfig) -> bool:
    base_ok = (
        metrics.get("recall_at_target_fp", 0.0) >= cfg.min_recall_at_target_fp_for_export
        and metrics["fp_per_hour"] <= cfg.target_false_positives_per_hour
        and metrics.get("threshold", 1.0) <= cfg.max_calibration_threshold_for_export
        and metrics.get("recall_at_0_5", 0.0) >= cfg.min_recall_at_0_5_for_export
        and metrics.get("fp_per_hour_at_0_5", math.inf) <= cfg.max_fp_per_hour_at_0_5_for_export
        and metrics.get("positive_median_score", 0.0) >= cfg.min_positive_median_score_for_export
        and metrics.get("positive_p10_score", 0.0) >= cfg.min_positive_p10_score_for_export
        and not math.isnan(metrics["loss"])
    )
    if not base_ok:
        return False
    if not cfg.use_positive_curve_validation:
        return True
    return (
        metrics.get("curve_recall", 0.0) >= cfg.min_curve_recall_for_export
        and metrics.get("curve_median_peak", 0.0) >= cfg.min_curve_median_peak_for_export
        and metrics.get("curve_p10_peak", 0.0) >= cfg.min_curve_p10_peak_for_export
        and metrics.get("curve_median_frames", 0.0) >= cfg.min_curve_median_frames_for_export
        and metrics.get("curve_median_span_ms", 0.0) >= cfg.min_curve_median_span_ms_for_export
        and metrics.get("curve_confirmation_rate", 0.0) >= cfg.min_curve_confirmation_rate_for_export
        and (
            not cfg.use_tablet_curve_validation
            or (
                metrics.get("tablet_curve_recall", 0.0) >= cfg.min_tablet_curve_recall_for_export
                and metrics.get("tablet_curve_median_frames", 0.0) >= cfg.min_tablet_curve_median_frames_for_export
                and metrics.get("tablet_curve_median_span_ms", 0.0) >= cfg.min_tablet_curve_median_span_ms_for_export
                and metrics.get("tablet_curve_confirmation_rate", 0.0) >= cfg.min_tablet_curve_confirmation_rate_for_export
            )
        )
    )


def _export_gate_failures(metrics: dict[str, float], cfg: TrainingConfig) -> list[str]:
    failures: list[str] = []
    if metrics.get("recall_at_target_fp", 0.0) < cfg.min_recall_at_target_fp_for_export:
        failures.append(
            "recall@targetFP="
            f"{metrics.get('recall_at_target_fp', 0.0):.3f} "
            f"< {cfg.min_recall_at_target_fp_for_export:.3f}"
        )
    if metrics.get("fp_per_hour", math.inf) > cfg.target_false_positives_per_hour:
        failures.append(
            f"FP/hr={metrics.get('fp_per_hour', math.inf):.2f} "
            f"> {cfg.target_false_positives_per_hour:.2f}"
        )
    if metrics.get("threshold", 1.0) > cfg.max_calibration_threshold_for_export:
        failures.append(
            f"calibration_threshold={metrics.get('threshold', 1.0):.3f} "
            f"> {cfg.max_calibration_threshold_for_export:.3f}"
        )
    if metrics.get("recall_at_0_5", 0.0) < cfg.min_recall_at_0_5_for_export:
        failures.append(
            f"raw_recall@0.5={metrics.get('recall_at_0_5', 0.0):.3f} "
            f"< {cfg.min_recall_at_0_5_for_export:.3f}"
        )
    if metrics.get("fp_per_hour_at_0_5", math.inf) > cfg.max_fp_per_hour_at_0_5_for_export:
        failures.append(
            f"rawFP/hr@0.5={metrics.get('fp_per_hour_at_0_5', math.inf):.2f} "
            f"> {cfg.max_fp_per_hour_at_0_5_for_export:.2f}"
        )
    if metrics.get("positive_median_score", 0.0) < cfg.min_positive_median_score_for_export:
        failures.append(
            f"pos_median={metrics.get('positive_median_score', 0.0):.3f} "
            f"< {cfg.min_positive_median_score_for_export:.3f}"
        )
    if metrics.get("positive_p10_score", 0.0) < cfg.min_positive_p10_score_for_export:
        failures.append(
            f"pos_p10={metrics.get('positive_p10_score', 0.0):.3f} "
            f"< {cfg.min_positive_p10_score_for_export:.3f}"
        )
    if cfg.use_positive_curve_validation:
        if metrics.get("curve_recall", 0.0) < cfg.min_curve_recall_for_export:
            failures.append(
                f"curve_recall={metrics.get('curve_recall', 0.0):.3f} "
                f"< {cfg.min_curve_recall_for_export:.3f}"
            )
        if metrics.get("curve_median_peak", 0.0) < cfg.min_curve_median_peak_for_export:
            failures.append(
                f"curve_med_peak={metrics.get('curve_median_peak', 0.0):.3f} "
                f"< {cfg.min_curve_median_peak_for_export:.3f}"
            )
        if metrics.get("curve_p10_peak", 0.0) < cfg.min_curve_p10_peak_for_export:
            failures.append(
                f"curve_p10_peak={metrics.get('curve_p10_peak', 0.0):.3f} "
                f"< {cfg.min_curve_p10_peak_for_export:.3f}"
            )
        if metrics.get("curve_median_frames", 0.0) < cfg.min_curve_median_frames_for_export:
            failures.append(
                f"curve_med_frames={metrics.get('curve_median_frames', 0.0):.1f} "
                f"< {cfg.min_curve_median_frames_for_export:.1f}"
            )
        if metrics.get("curve_median_span_ms", 0.0) < cfg.min_curve_median_span_ms_for_export:
            failures.append(
                f"curve_med_span={metrics.get('curve_median_span_ms', 0.0):.0f}ms "
                f"< {cfg.min_curve_median_span_ms_for_export:.0f}ms"
            )
        if metrics.get("curve_confirmation_rate", 0.0) < cfg.min_curve_confirmation_rate_for_export:
            failures.append(
                f"curve_confirm={metrics.get('curve_confirmation_rate', 0.0):.3f} "
                f"< {cfg.min_curve_confirmation_rate_for_export:.3f}"
            )
        if cfg.use_tablet_curve_validation:
            if metrics.get("tablet_curve_recall", 0.0) < cfg.min_tablet_curve_recall_for_export:
                failures.append(
                    f"tablet_curve_recall={metrics.get('tablet_curve_recall', 0.0):.3f} "
                    f"< {cfg.min_tablet_curve_recall_for_export:.3f}"
                )
            if metrics.get("tablet_curve_median_frames", 0.0) < cfg.min_tablet_curve_median_frames_for_export:
                failures.append(
                    f"tablet_curve_med_frames={metrics.get('tablet_curve_median_frames', 0.0):.1f} "
                    f"< {cfg.min_tablet_curve_median_frames_for_export:.1f}"
                )
            if metrics.get("tablet_curve_median_span_ms", 0.0) < cfg.min_tablet_curve_median_span_ms_for_export:
                failures.append(
                    f"tablet_curve_med_span={metrics.get('tablet_curve_median_span_ms', 0.0):.0f}ms "
                    f"< {cfg.min_tablet_curve_median_span_ms_for_export:.0f}ms"
                )
            if metrics.get("tablet_curve_confirmation_rate", 0.0) < cfg.min_tablet_curve_confirmation_rate_for_export:
                failures.append(
                    f"tablet_curve_confirm={metrics.get('tablet_curve_confirmation_rate', 0.0):.3f} "
                    f"< {cfg.min_tablet_curve_confirmation_rate_for_export:.3f}"
                )
    if math.isnan(metrics.get("loss", math.nan)):
        failures.append("val_loss is NaN")
    return failures


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
            float(metrics.get("recall_at_0_5", 0.0)),
            -float(metrics.get("threshold", 1.0)),
            -float(metrics["fp_per_hour"]),
            -float(metrics.get("fp_per_hour_at_0_5", math.inf)),
            float(metrics.get("positive_median_score", 0.0)),
            float(metrics.get("tablet_curve_confirmation_rate", 0.0)),
            float(metrics.get("tablet_curve_median_frames", 0.0)),
            float(metrics.get("tablet_curve_median_peak", 0.0)),
            float(metrics.get("curve_confirmation_rate", 0.0)),
            float(metrics.get("curve_median_frames", 0.0)),
            float(metrics.get("curve_median_peak", 0.0)),
            float(metrics.get("recall_at_p95", 0.0)),
            loss_score,
        )
    return (
        0,
        -float(len(_export_gate_failures(metrics, cfg))),
        float(metrics.get("recall_at_target_fp", 0.0)),
        float(metrics.get("recall_at_0_5", 0.0)),
        -float(metrics.get("threshold", 1.0)),
        -float(metrics.get("fp_per_hour_at_0_5", math.inf)),
        float(metrics.get("positive_median_score", 0.0)),
        float(metrics.get("tablet_curve_confirmation_rate", 0.0)),
        float(metrics.get("tablet_curve_median_frames", 0.0)),
        float(metrics.get("tablet_curve_median_peak", 0.0)),
        float(metrics.get("curve_confirmation_rate", 0.0)),
        float(metrics.get("curve_median_frames", 0.0)),
        float(metrics.get("curve_median_peak", 0.0)),
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
    curve_val: CurveValidationSet | None,
    device: torch.device,
    out_dir: Path,
    history: list[dict],
    best_score: tuple | None,
    best_metrics: dict[str, float] | None,
    best_step: int,
    best_exportable: bool,
    cancel_flag=None,
) -> tuple[tuple | None, dict[str, float] | None, int, bool]:
    if cfg.hard_negative_finetune_steps <= 0:
        bus.log("Skipping hard-negative fine-tune (0 steps configured).")
        return best_score, best_metrics, best_step, best_exportable

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
        loss = _weighted_loss(
            probs,
            y,
            cfg,
            negative_weight=float(cfg.max_negative_loss_weight),
        )
        loss.backward()
        optimizer.step()

        if ft_step % cfg.val_every_n_steps == 0 or ft_step == cfg.hard_negative_finetune_steps:
            metrics = _evaluate(model, val_loader, device, cfg)
            metrics.update(_evaluate_positive_curves(model, curve_val, device, cfg, metrics["threshold"]))
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
                val_recall_at_0_5=metrics["recall_at_0_5"],
                val_fp_per_hour_at_0_5=metrics["fp_per_hour_at_0_5"],
                positive_median_score=metrics["positive_median_score"],
                positive_p10_score=metrics["positive_p10_score"],
                curve_recall=metrics.get("curve_recall", 0.0),
                curve_median_peak=metrics.get("curve_median_peak", 0.0),
                curve_median_frames=metrics.get("curve_median_frames", 0.0),
                curve_confirmation_rate=metrics.get("curve_confirmation_rate", 0.0),
                threshold=metrics["threshold"],
            )
            bus.log(
                f"hard-neg step={ft_step} val_loss={metrics['loss']:.4f} "
                f"recall@p95={metrics['recall_at_p95']:.3f} "
                f"recall@targetFP={metrics['recall_at_target_fp']:.3f} "
                f"FP/hr={metrics['fp_per_hour']:.2f} "
                f"recall@0.5={metrics['recall_at_0_5']:.3f} "
                f"rawFP/hr@0.5={metrics['fp_per_hour_at_0_5']:.2f} "
                f"pos_med={metrics['positive_median_score']:.3f} "
                f"pos_p10={metrics['positive_p10_score']:.3f} "
                f"curve={metrics.get('curve_recall', 0.0):.3f}/"
                f"{metrics.get('curve_median_peak', 0.0):.3f}/"
                f"{metrics.get('curve_median_frames', 0.0):.1f}f/"
                f"{metrics.get('curve_confirmation_rate', 0.0):.3f} "
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


def _refresh_hard_negatives(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainingConfig,
    train_ds: FeatureMemmapDataset,
    val_loader: DataLoader,
    curve_val: CurveValidationSet | None,
    device: torch.device,
    refresh_index: int,
    global_step: int,
    cancel_flag=None,
) -> tuple[int, dict[str, float] | None]:
    """Briefly train on mined hard negatives, then re-evaluate.

    Random negative sampling is dominated by easy negatives, but FP/hour is set
    by the rare high-scoring tail. On a plateau, mine that tail from the train
    shard and run a short corrective burst instead of letting BCE keep improving
    on easy examples while the operating metric degrades.
    """

    if (
        not cfg.hard_negative_refresh_on_plateau
        or cfg.hard_negative_refresh_steps <= 0
        or cfg.hard_negative_refresh_top_k <= 0
    ):
        return 0, None

    labels = np.asarray(train_ds.labels[:], dtype=np.int64)
    pos_indices = np.flatnonzero(labels == 1)
    if pos_indices.size == 0:
        return 0, None

    bus.log(
        "Plateau hard-negative refresh "
        f"{refresh_index}/{cfg.max_hard_negative_refreshes}: mining top "
        f"{cfg.hard_negative_refresh_top_k:,} negatives from training shard..."
    )
    hard_neg_indices = _mine_hard_negatives(
        model,
        train_ds,
        device,
        top_k=cfg.hard_negative_refresh_top_k,
        batch_size=max(4096, cfg.batch_size * 2),
    )
    if hard_neg_indices.size == 0:
        bus.log("No hard negatives found for plateau refresh.", level="warning")
        return 0, None

    rng = np.random.default_rng(cfg.seed + 101 + refresh_index)
    pos_per_batch = max(
        1,
        min(
            cfg.batch_size - 1,
            int(cfg.batch_size * cfg.hard_negative_refresh_positive_fraction),
        ),
    )
    neg_per_batch = cfg.batch_size - pos_per_batch
    steps_done = 0
    for local_step in range(1, cfg.hard_negative_refresh_steps + 1):
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
        loss = _weighted_loss(
            probs,
            y,
            cfg,
            negative_weight=float(cfg.max_negative_loss_weight),
        )
        loss.backward()
        optimizer.step()
        steps_done += 1

        if local_step % 100 == 0 or local_step == cfg.hard_negative_refresh_steps:
            bus.metric(
                step=global_step + steps_done,
                max_steps=cfg.max_steps,
                train_loss=float(loss.item()),
            )
            bus.progress(
                "train",
                min(1.0, (global_step + steps_done) / cfg.max_steps),
                detail=(
                    f"hard-negative refresh {refresh_index} "
                    f"{local_step}/{cfg.hard_negative_refresh_steps}"
                ),
            )

    if steps_done == 0:
        return 0, None

    metrics = _evaluate(model, val_loader, device, cfg)
    metrics.update(_evaluate_positive_curves(model, curve_val, device, cfg, metrics["threshold"]))
    metrics["step"] = global_step + steps_done
    metrics["phase"] = "hard_negative_refresh"
    bus.metric(
        step=metrics["step"],
        val_loss=metrics["loss"],
        val_accuracy=metrics["accuracy"],
        val_recall_at_p95=metrics["recall_at_p95"],
        val_recall_at_target_fp=metrics["recall_at_target_fp"],
        val_fp_per_hour=metrics["fp_per_hour"],
        val_recall_at_0_5=metrics["recall_at_0_5"],
        val_fp_per_hour_at_0_5=metrics["fp_per_hour_at_0_5"],
        positive_median_score=metrics["positive_median_score"],
        positive_p10_score=metrics["positive_p10_score"],
        curve_recall=metrics.get("curve_recall", 0.0),
        curve_median_peak=metrics.get("curve_median_peak", 0.0),
        curve_median_frames=metrics.get("curve_median_frames", 0.0),
        curve_confirmation_rate=metrics.get("curve_confirmation_rate", 0.0),
        threshold=metrics["threshold"],
    )
    bus.log(
        f"hard-refresh step={metrics['step']} val_loss={metrics['loss']:.4f} "
        f"recall@p95={metrics['recall_at_p95']:.3f} "
        f"recall@targetFP={metrics['recall_at_target_fp']:.3f} "
        f"FP/hr={metrics['fp_per_hour']:.2f} "
        f"recall@0.5={metrics['recall_at_0_5']:.3f} "
        f"rawFP/hr@0.5={metrics['fp_per_hour_at_0_5']:.2f} "
        f"pos_med={metrics['positive_median_score']:.3f} "
        f"pos_p10={metrics['positive_p10_score']:.3f} "
        f"curve={metrics.get('curve_recall', 0.0):.3f}/"
        f"{metrics.get('curve_median_peak', 0.0):.3f}/"
        f"{metrics.get('curve_median_frames', 0.0):.1f}f/"
        f"{metrics.get('curve_confirmation_rate', 0.0):.3f} "
        f"threshold={metrics['threshold']:.5f}"
    )
    return steps_done, metrics


def train(
    cfg: TrainingConfig,
    train_ds: FeatureMemmapDataset,
    val_ds: FeatureMemmapDataset,
    out_dir: Path,
    workers: int = 4,
    curve_val: CurveValidationSet | None = None,
    cancel_flag=None,
) -> TrainResult:
    """Run the training loop. `cancel_flag.is_set()` aborts gracefully."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bus.log(f"Device: {device}")

    model = build_model(cfg.model_type, cfg.layer_dim, cfg.n_blocks).to(device)
    current_lr = float(cfg.learning_rate)
    lr_scale = 1.0
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=current_lr,
        weight_decay=float(cfg.weight_decay),
    )
    lr_reductions = 0
    hard_negative_refreshes = 0

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
    bus.log(
        "Training objective: "
        f"positive_sample_fraction={cfg.positive_sample_fraction:.2f}, "
        f"negative_loss_weight=1.00->{cfg.max_negative_loss_weight:.0f}, "
        f"hard_negative_threshold={cfg.hard_negative_threshold:.2f}, "
        f"hard_negative_loss_weight={cfg.hard_negative_loss_weight:.2f}, "
        f"focal_loss={cfg.use_focal_loss}, "
        f"label_smoothing={cfg.label_smoothing:.3f}, "
        f"mixup_alpha={cfg.mixup_alpha:.2f}, "
        f"positive_confidence={cfg.positive_confidence_weight:.2f}@{cfg.positive_confidence_target:.2f}, "
        f"negative_confidence={cfg.negative_confidence_weight:.2f}@{cfg.negative_confidence_target:.2f}, "
        f"separation={cfg.separation_loss_weight:.2f}/margin={cfg.separation_margin:.2f}, "
        f"export_gates=threshold<={cfg.max_calibration_threshold_for_export:.2f}, "
        f"raw_recall@0.5>={cfg.min_recall_at_0_5_for_export:.2f}, "
        f"rawFP/hr@0.5<={cfg.max_fp_per_hour_at_0_5_for_export:.2f}, "
        f"pos_median>={cfg.min_positive_median_score_for_export:.2f}, "
        f"pos_p10>={cfg.min_positive_p10_score_for_export:.2f}, "
        f"curve_validation={cfg.use_positive_curve_validation}, "
        f"curve_recall>={cfg.min_curve_recall_for_export:.2f}, "
        f"curve_med_peak>={cfg.min_curve_median_peak_for_export:.2f}, "
        f"curve_p10_peak>={cfg.min_curve_p10_peak_for_export:.2f}, "
        f"curve_med_frames>={cfg.min_curve_median_frames_for_export}, "
        f"curve_med_span>={cfg.min_curve_median_span_ms_for_export:.0f}ms, "
        f"curve_confirm>={cfg.min_curve_confirmation_rate_for_export:.2f}, "
        f"tablet_curve_validation={cfg.use_tablet_curve_validation}, "
        f"tablet_curve_recall>={cfg.min_tablet_curve_recall_for_export:.2f}, "
        f"tablet_curve_med_peak>={cfg.min_tablet_curve_median_peak_for_export:.2f}, "
        f"tablet_curve_med_frames>={cfg.min_tablet_curve_median_frames_for_export}, "
        f"tablet_curve_med_span>={cfg.min_tablet_curve_median_span_ms_for_export:.0f}ms, "
        f"tablet_curve_confirm>={cfg.min_tablet_curve_confirmation_rate_for_export:.2f}, "
        f"positive_temporal_windows={cfg.positive_temporal_windows}x"
        f"{cfg.positive_temporal_stride_embeddings}, "
        f"positive_context={cfg.positive_context_seconds:.1f}s, "
        f"quality_extension_steps={cfg.exportable_quality_extension_steps}, "
        f"weight_decay={cfg.weight_decay:.2e}, "
        f"lr={current_lr:.2e}, "
        "lr_schedule=warmup_hold_cosine, "
        f"lr_reduce_on_plateau={cfg.lr_reduce_on_plateau}"
    )

    history: list[dict] = []
    best_metrics: dict[str, float] | None = None
    best_score: tuple | None = None
    best_step = 0
    best_exportable = False
    no_improve = 0
    exportable_quality_until_step = int(cfg.early_stop_min_steps)

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

        if cfg.mixup_alpha > 0:
            lam = torch.distributions.Beta(
                float(cfg.mixup_alpha), float(cfg.mixup_alpha)
            ).sample().to(device)
            perm = torch.randperm(x.size(0), device=device)
            x = lam * x + (1.0 - lam) * x[perm]
            y = lam * y + (1.0 - lam) * y[perm]

        current_lr = _cosine_warmup_lr(
            step,
            cfg.max_steps,
            float(cfg.learning_rate),
            float(cfg.lr_warmup_fraction),
            float(cfg.lr_hold_fraction),
        )
        current_lr = max(float(cfg.min_learning_rate), current_lr * lr_scale)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        negative_weight = _scheduled_negative_weight(cfg, step)

        model.train()
        optimizer.zero_grad()
        p = model(x).squeeze(-1)
        loss = _weighted_loss(p, y, cfg, negative_weight=negative_weight)
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
            metrics.update(_evaluate_positive_curves(model, curve_val, device, cfg, metrics["threshold"]))
            metrics["step"] = step
            history.append(metrics)
            bus.metric(
                step=step,
                val_loss=metrics["loss"],
                val_accuracy=metrics["accuracy"],
                val_recall_at_p95=metrics["recall_at_p95"],
                val_recall_at_target_fp=metrics["recall_at_target_fp"],
                val_fp_per_hour=metrics["fp_per_hour"],
                val_recall_at_0_5=metrics["recall_at_0_5"],
                val_fp_per_hour_at_0_5=metrics["fp_per_hour_at_0_5"],
                positive_median_score=metrics["positive_median_score"],
                positive_p10_score=metrics["positive_p10_score"],
                curve_recall=metrics.get("curve_recall", 0.0),
                curve_median_peak=metrics.get("curve_median_peak", 0.0),
                curve_p10_peak=metrics.get("curve_p10_peak", 0.0),
                curve_median_frames=metrics.get("curve_median_frames", 0.0),
                curve_median_span_ms=metrics.get("curve_median_span_ms", 0.0),
                curve_confirmation_rate=metrics.get("curve_confirmation_rate", 0.0),
                tablet_curve_recall=metrics.get("tablet_curve_recall", 0.0),
                tablet_curve_median_peak=metrics.get("tablet_curve_median_peak", 0.0),
                tablet_curve_p10_peak=metrics.get("tablet_curve_p10_peak", 0.0),
                tablet_curve_median_frames=metrics.get("tablet_curve_median_frames", 0.0),
                tablet_curve_median_span_ms=metrics.get("tablet_curve_median_span_ms", 0.0),
                tablet_curve_confirmation_rate=metrics.get("tablet_curve_confirmation_rate", 0.0),
                threshold=metrics["threshold"],
            )
            bus.log(
                f"step={step} val_loss={metrics['loss']:.4f} "
                f"recall@p95={metrics['recall_at_p95']:.3f} "
                f"recall@targetFP={metrics['recall_at_target_fp']:.3f} "
                f"FP/hr={metrics['fp_per_hour']:.2f} "
                f"recall@0.5={metrics['recall_at_0_5']:.3f} "
                f"rawFP/hr@0.5={metrics['fp_per_hour_at_0_5']:.2f} "
                f"pos_med={metrics['positive_median_score']:.3f} "
                f"pos_p10={metrics['positive_p10_score']:.3f} "
                f"curve={metrics.get('curve_recall', 0.0):.3f}/"
                f"{metrics.get('curve_median_peak', 0.0):.3f}/"
                f"{metrics.get('curve_median_frames', 0.0):.1f}f/"
                f"{metrics.get('curve_median_span_ms', 0.0):.0f}ms/"
                f"{metrics.get('curve_confirmation_rate', 0.0):.3f} "
                f"tablet_curve={metrics.get('tablet_curve_recall', 0.0):.3f}/"
                f"{metrics.get('tablet_curve_median_peak', 0.0):.3f}/"
                f"{metrics.get('tablet_curve_median_frames', 0.0):.1f}f/"
                f"{metrics.get('tablet_curve_median_span_ms', 0.0):.0f}ms/"
                f"{metrics.get('tablet_curve_confirmation_rate', 0.0):.3f} "
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
                    if int(cfg.exportable_quality_extension_steps) > 0:
                        next_quality_until = min(
                            int(cfg.max_steps),
                            step + int(cfg.exportable_quality_extension_steps),
                        )
                        if next_quality_until > exportable_quality_until_step:
                            exportable_quality_until_step = next_quality_until
                            bus.log(
                                "Extended exportable quality window: "
                                f"continue until step={exportable_quality_until_step} "
                                f"(best exportable at step={step})."
                            )
                    bus.log(
                        "New exportable checkpoint: "
                        f"step={step} recall@targetFP={metrics['recall_at_target_fp']:.3f} "
                        f"FP/hr={metrics['fp_per_hour']:.2f}"
                    )
                else:
                    failures = _export_gate_failures(metrics, cfg)
                    if failures:
                        bus.log(
                            "Best candidate is not exportable yet: "
                            + "; ".join(failures),
                            level="warning",
                        )
            else:
                no_improve += 1
                can_reduce_lr = (
                    cfg.lr_reduce_on_plateau
                    and best_score is not None
                    and no_improve >= cfg.lr_reduce_patience
                    and lr_reductions < cfg.max_lr_reductions
                    and current_lr > cfg.min_learning_rate
                    and (out_dir / "best_candidate.pt").exists()
                )
                if can_reduce_lr:
                    next_lr = max(
                        float(cfg.min_learning_rate),
                        current_lr * float(cfg.lr_reduce_factor),
                    )
                    if next_lr < current_lr:
                        # The monitored metric can degrade while BCE loss keeps
                        # improving because the metric is dominated by the
                        # long-tail negative scores. Rewind to the best metric
                        # checkpoint and reset Adam so stale moments from the
                        # degraded weights do not push us back off the cliff.
                        model.load_state_dict(
                            torch.load(out_dir / "best_candidate.pt", map_location=device)
                        )
                        lr_scale *= float(cfg.lr_reduce_factor)
                        current_lr = next_lr
                        optimizer = torch.optim.AdamW(
                            model.parameters(),
                            lr=current_lr,
                            weight_decay=float(cfg.weight_decay),
                        )
                        lr_reductions += 1
                        no_improve = 0
                        train_iter = iter(train_loader)
                        bus.log(
                            "Validation metric plateaued; restored best "
                            f"checkpoint from step={best_step}, reduced LR to "
                            f"{current_lr:.2e} ({lr_reductions}/"
                            f"{cfg.max_lr_reductions})."
                        )
                        if (
                            cfg.hard_negative_refresh_on_plateau
                            and hard_negative_refreshes < cfg.max_hard_negative_refreshes
                        ):
                            hard_negative_refreshes += 1
                            refresh_steps, refresh_metrics = _refresh_hard_negatives(
                                model,
                                optimizer,
                                cfg,
                                train_ds,
                                val_loader,
                                curve_val,
                                device,
                                hard_negative_refreshes,
                                step,
                                cancel_flag=cancel_flag,
                            )
                            step += refresh_steps
                            if refresh_metrics is not None:
                                history.append(refresh_metrics)
                                score = _checkpoint_score(refresh_metrics, cfg)
                                if best_score is None or score > best_score:
                                    best_score = score
                                    best_metrics = refresh_metrics.copy()
                                    best_step = int(refresh_metrics["step"])
                                    best_exportable = _is_exportable(refresh_metrics, cfg)
                                    torch.save(model.state_dict(), out_dir / "best_candidate.pt")
                                    if best_exportable:
                                        torch.save(model.state_dict(), out_dir / "best.pt")
                                        if int(cfg.exportable_quality_extension_steps) > 0:
                                            next_quality_until = min(
                                                int(cfg.max_steps),
                                                best_step + int(cfg.exportable_quality_extension_steps),
                                            )
                                            if next_quality_until > exportable_quality_until_step:
                                                exportable_quality_until_step = next_quality_until
                                                bus.log(
                                                    "Extended exportable quality window: "
                                                    f"continue until step={exportable_quality_until_step} "
                                                    f"(best exportable at step={best_step})."
                                                )
                                        bus.log(
                                            "New exportable hard-refresh checkpoint: "
                                            f"step={best_step} "
                                            "recall@targetFP="
                                            f"{refresh_metrics['recall_at_target_fp']:.3f} "
                                            f"FP/hr={refresh_metrics['fp_per_hour']:.2f}"
                                        )
                        continue
                min_stop_step = (
                    exportable_quality_until_step
                    if best_exportable
                    else int(cfg.early_stop_min_steps)
                )
                if no_improve >= cfg.early_stop_patience and step >= min_stop_step:
                    bus.log(
                        f"Early stop after {no_improve} evals without improvement "
                        f"(min_steps={min_stop_step})",
                        level="warning",
                    )
                    break

            if _is_exportable(metrics, cfg) and step >= exportable_quality_until_step:
                bus.log("Hit target FP/hour with strong recall. Stopping.", level="info")
                break
            if _is_exportable(metrics, cfg) and step < exportable_quality_until_step:
                bus.log(
                    "Checkpoint is exportable; continuing until "
                    f"quality_window_step={exportable_quality_until_step} "
                    "to improve quality.",
                    level="info",
                )

    # Restore best weights and export.
    best_path = out_dir / "best.pt"
    if not best_exportable and (out_dir / "best_candidate.pt").exists():
        model.load_state_dict(torch.load(out_dir / "best_candidate.pt", map_location=device))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=current_lr,
            weight_decay=float(cfg.weight_decay),
        )
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
            curve_val,
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
        failures = _export_gate_failures(candidate, cfg)
        failure_detail = "; ".join(failures) if failures else "unknown export gate failure"
        msg = (
            "No exportable checkpoint found. "
            f"Best candidate: step={best_step}, "
            f"recall@targetFP={candidate.get('recall_at_target_fp', 0.0):.3f} "
            f"(required >= {cfg.min_recall_at_target_fp_for_export:.3f}), "
            f"FP/hr={candidate['fp_per_hour']:.2f} "
            f"(required <= {cfg.target_false_positives_per_hour:.2f}), "
            f"threshold={candidate.get('threshold', 1.0):.3f} "
            f"(required <= {cfg.max_calibration_threshold_for_export:.3f}), "
            f"raw_recall@0.5={candidate.get('recall_at_0_5', 0.0):.3f} "
            f"(required >= {cfg.min_recall_at_0_5_for_export:.3f}), "
            f"rawFP/hr@0.5={candidate.get('fp_per_hour_at_0_5', math.inf):.2f} "
            f"(required <= {cfg.max_fp_per_hour_at_0_5_for_export:.2f}), "
            f"pos_median={candidate.get('positive_median_score', 0.0):.3f} "
            f"(required >= {cfg.min_positive_median_score_for_export:.3f}), "
            f"pos_p10={candidate.get('positive_p10_score', 0.0):.3f} "
            f"(required >= {cfg.min_positive_p10_score_for_export:.3f}), "
            f"curve_recall={candidate.get('curve_recall', 0.0):.3f} "
            f"(required >= {cfg.min_curve_recall_for_export:.3f}), "
            f"curve_med_peak={candidate.get('curve_median_peak', 0.0):.3f} "
            f"(required >= {cfg.min_curve_median_peak_for_export:.3f}), "
            f"curve_p10_peak={candidate.get('curve_p10_peak', 0.0):.3f} "
            f"(required >= {cfg.min_curve_p10_peak_for_export:.3f}), "
            f"curve_med_frames={candidate.get('curve_median_frames', 0.0):.1f} "
            f"(required >= {cfg.min_curve_median_frames_for_export:.1f}), "
            f"curve_med_span={candidate.get('curve_median_span_ms', 0.0):.0f}ms "
            f"(required >= {cfg.min_curve_median_span_ms_for_export:.0f}ms), "
            f"curve_confirm={candidate.get('curve_confirmation_rate', 0.0):.3f} "
            f"(required >= {cfg.min_curve_confirmation_rate_for_export:.3f}), "
            f"tablet_curve_recall={candidate.get('tablet_curve_recall', 0.0):.3f} "
            f"(required >= {cfg.min_tablet_curve_recall_for_export:.3f}), "
            f"tablet_curve_med_peak={candidate.get('tablet_curve_median_peak', 0.0):.3f} "
            f"(required >= {cfg.min_tablet_curve_median_peak_for_export:.3f}), "
            f"tablet_curve_med_frames={candidate.get('tablet_curve_median_frames', 0.0):.1f} "
            f"(required >= {cfg.min_tablet_curve_median_frames_for_export:.1f}), "
            f"tablet_curve_med_span={candidate.get('tablet_curve_median_span_ms', 0.0):.0f}ms "
            f"(required >= {cfg.min_tablet_curve_median_span_ms_for_export:.0f}ms), "
            f"tablet_curve_confirm={candidate.get('tablet_curve_confirmation_rate', 0.0):.3f} "
            f"(required >= {cfg.min_tablet_curve_confirmation_rate_for_export:.3f}). "
            f"Failed gates: {failure_detail}. "
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
        f"raw_recall@0.5={best_metrics['recall_at_0_5']:.3f}, "
        f"rawFP/hr@0.5={best_metrics['fp_per_hour_at_0_5']:.2f}, "
        f"pos_med={best_metrics['positive_median_score']:.3f}, "
        f"pos_p10={best_metrics['positive_p10_score']:.3f}, "
        f"curve={best_metrics.get('curve_recall', 0.0):.3f}/"
        f"{best_metrics.get('curve_median_peak', 0.0):.3f}/"
        f"{best_metrics.get('curve_median_frames', 0.0):.1f}f/"
        f"{best_metrics.get('curve_median_span_ms', 0.0):.0f}ms/"
        f"{best_metrics.get('curve_confirmation_rate', 0.0):.3f}, "
        f"tablet_curve={best_metrics.get('tablet_curve_recall', 0.0):.3f}/"
        f"{best_metrics.get('tablet_curve_median_peak', 0.0):.3f}/"
        f"{best_metrics.get('tablet_curve_median_frames', 0.0):.1f}f/"
        f"{best_metrics.get('tablet_curve_median_span_ms', 0.0):.0f}ms/"
        f"{best_metrics.get('tablet_curve_confirmation_rate', 0.0):.3f}, "
        f"onnx_max_abs_diff={parity_diff:.2e})"
    )

    return TrainResult(
        onnx_path=onnx_path,
        best_val_loss=best_metrics["loss"],
        best_val_recall=best_metrics["recall_at_p95"],
        best_val_recall_at_target_fp=best_metrics["recall_at_target_fp"],
        best_val_fp_per_hour=best_metrics["fp_per_hour"],
        best_val_recall_at_0_5=best_metrics["recall_at_0_5"],
        best_val_fp_per_hour_at_0_5=best_metrics["fp_per_hour_at_0_5"],
        best_positive_median_score=best_metrics["positive_median_score"],
        best_positive_p10_score=best_metrics["positive_p10_score"],
        best_curve_recall=best_metrics.get("curve_recall", 0.0),
        best_curve_median_peak=best_metrics.get("curve_median_peak", 0.0),
        best_curve_p10_peak=best_metrics.get("curve_p10_peak", 0.0),
        best_curve_median_frames=best_metrics.get("curve_median_frames", 0.0),
        best_curve_median_span_ms=best_metrics.get("curve_median_span_ms", 0.0),
        best_curve_confirmation_rate=best_metrics.get("curve_confirmation_rate", 0.0),
        best_tablet_curve_recall=best_metrics.get("tablet_curve_recall", 0.0),
        best_tablet_curve_median_peak=best_metrics.get("tablet_curve_median_peak", 0.0),
        best_tablet_curve_p10_peak=best_metrics.get("tablet_curve_p10_peak", 0.0),
        best_tablet_curve_median_frames=best_metrics.get("tablet_curve_median_frames", 0.0),
        best_tablet_curve_median_span_ms=best_metrics.get("tablet_curve_median_span_ms", 0.0),
        best_tablet_curve_confirmation_rate=best_metrics.get("tablet_curve_confirmation_rate", 0.0),
        best_threshold=best_metrics["threshold"],
        best_step=best_step,
        history=history,
    )
