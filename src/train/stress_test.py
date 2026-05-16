"""False-positive stress tests for exported wake-word ONNX models.

The training loop already computes FP/hr, but this module makes the same kind
of event-based negative sweep available after export. It can run against a
session's cached feature shards and the official openWakeWord negative feature
banks without retraining.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import onnxruntime as ort

from src.data.features import CLASSIFIER_WINDOW_EMBEDDINGS, EMBEDDING_DIM
from src.train.trainer import FP_EVENT_REFRACTORY_WINDOWS

WINDOW_SECONDS = 0.08


def _providers(use_cuda: bool) -> list[str]:
    available = set(ort.get_available_providers())
    providers: list[str] = []
    if use_cuda and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def _session(model_path: Path, use_cuda: bool) -> ort.InferenceSession:
    return ort.InferenceSession(str(model_path), providers=_providers(use_cuda))


def _score_batch(sess: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    input_name = sess.get_inputs()[0].name
    try:
        out = sess.run(None, {input_name: x.astype(np.float32, copy=False)})[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)[: x.shape[0]]
    except Exception:
        # Some exported ONNX graphs are fixed-batch. Keep the stress test useful
        # instead of failing the whole run.
        scores = []
        for i in range(x.shape[0]):
            out = sess.run(None, {input_name: x[i : i + 1].astype(np.float32, copy=False)})[0]
            scores.append(float(np.asarray(out).reshape(-1)[0]))
        return np.asarray(scores, dtype=np.float32)


def _count_events(
    scores: np.ndarray,
    threshold: float,
    source_ids: np.ndarray | None = None,
) -> int:
    above = np.flatnonzero(scores >= threshold)
    if above.size == 0:
        return 0
    starts = np.ones(above.size, dtype=bool)
    if above.size > 1:
        if source_ids is None:
            starts[1:] = (above[1:] - above[:-1]) > FP_EVENT_REFRACTORY_WINDOWS
        else:
            src = source_ids[above]
            starts[1:] = (
                (src[1:] != src[:-1])
                | ((above[1:] - above[:-1]) > FP_EVENT_REFRACTORY_WINDOWS)
            )
    return int(starts.sum())


def _score_iter(
    sess: ort.InferenceSession,
    batches: Iterable[np.ndarray],
) -> np.ndarray:
    scores: list[np.ndarray] = []
    for x in batches:
        if x.size:
            scores.append(_score_batch(sess, x))
    if not scores:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(scores).astype(np.float32, copy=False)


def _percentiles(scores: np.ndarray) -> dict[str, float]:
    if scores.size == 0:
        return {}
    return {
        "min": float(np.min(scores)),
        "p10": float(np.percentile(scores, 10)),
        "p50": float(np.percentile(scores, 50)),
        "p90": float(np.percentile(scores, 90)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
        "p99_9": float(np.percentile(scores, 99.9)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
    }


def _negative_report(
    name: str,
    scores: np.ndarray,
    threshold: float,
    source_ids: np.ndarray | None = None,
) -> dict:
    hours = (int(scores.size) * WINDOW_SECONDS) / 3600.0
    events = _count_events(scores, threshold, source_ids=source_ids)
    return {
        "name": name,
        "kind": "negative",
        "windows": int(scores.size),
        "hours": float(hours),
        "threshold": float(threshold),
        "events": int(events),
        "fp_per_hour": float(events / hours) if hours > 0 else 0.0,
        "windows_above_threshold": int((scores >= threshold).sum()),
        "score": _percentiles(scores),
    }


def _positive_report(name: str, scores: np.ndarray, threshold: float, source_ids: np.ndarray | None) -> dict:
    if source_ids is not None and source_ids.size == scores.size and (source_ids >= 0).any():
        valid = source_ids >= 0
        src = source_ids[valid]
        vals = scores[valid]
        order = np.argsort(src)
        sorted_src = src[order]
        sorted_scores = vals[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_src)) + 1]
        clip_scores = np.maximum.reduceat(sorted_scores, starts)
        unit = "clips"
    else:
        clip_scores = scores
        unit = "windows"
    return {
        "name": name,
        "kind": "positive",
        unit: int(clip_scores.size),
        "threshold": float(threshold),
        "recall": float((clip_scores >= threshold).sum() / max(1, clip_scores.size)),
        "score": _percentiles(clip_scores),
    }


def _session_batches(features: np.memmap, indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, indices.size, batch_size):
        yield np.asarray(features[indices[start : start + batch_size]], dtype=np.float32)


def _stress_session_split(
    sess: ort.InferenceSession,
    run_dir: Path,
    split: str,
    threshold: float,
    batch_size: int,
    max_windows: int | None,
) -> list[dict]:
    labels_path = run_dir / f"{split}_labels.npy"
    features_path = run_dir / f"{split}_features.bin"
    source_ids_path = run_dir / f"{split}_source_ids.npy"
    if not labels_path.exists() or not features_path.exists():
        return []

    labels = np.load(labels_path, mmap_mode="r")
    n = int(labels.shape[0])
    if max_windows is not None:
        n = min(n, max_windows)
    labels_view = np.asarray(labels[:n], dtype=np.uint8)
    features = np.memmap(
        features_path,
        dtype=np.float32,
        mode="r",
        shape=(int(labels.shape[0]), CLASSIFIER_WINDOW_EMBEDDINGS, EMBEDDING_DIM),
    )
    source_ids = (
        np.asarray(np.load(source_ids_path, mmap_mode="r")[:n], dtype=np.int64)
        if source_ids_path.exists()
        else None
    )

    reports: list[dict] = []
    for label, label_name in ((0, "negative"), (1, "positive")):
        indices = np.flatnonzero(labels_view == label)
        if indices.size == 0:
            continue
        scores = _score_iter(sess, _session_batches(features, indices, batch_size))
        src = source_ids[indices] if source_ids is not None else None
        name = f"{run_dir.name}:{split}:{label_name}"
        if label == 0:
            reports.append(_negative_report(name, scores, threshold, src))
        else:
            reports.append(_positive_report(name, scores, threshold, src))
    return reports


def _external_batches(features: np.ndarray, n_windows: int, batch_size: int) -> Iterable[np.ndarray]:
    if features.ndim == 3:
        for start in range(0, n_windows, batch_size):
            end = min(n_windows, start + batch_size)
            yield np.asarray(features[start:end], dtype=np.float32)
        return

    for start in range(0, n_windows, batch_size):
        end = min(n_windows, start + batch_size)
        out = np.empty((end - start, CLASSIFIER_WINDOW_EMBEDDINGS, EMBEDDING_DIM), dtype=np.float32)
        for i, idx in enumerate(range(start, end)):
            out[i] = np.asarray(features[idx : idx + CLASSIFIER_WINDOW_EMBEDDINGS], dtype=np.float32)
        yield out


def _stress_external_features(
    sess: ort.InferenceSession,
    path: Path,
    threshold: float,
    batch_size: int,
    max_windows: int | None,
) -> dict:
    features = np.load(path, mmap_mode="r")
    if features.ndim == 3 and features.shape[1:] == (CLASSIFIER_WINDOW_EMBEDDINGS, EMBEDDING_DIM):
        n_windows = int(features.shape[0])
    elif features.ndim == 2 and features.shape[1] == EMBEDDING_DIM:
        n_windows = max(0, int(features.shape[0]) - CLASSIFIER_WINDOW_EMBEDDINGS + 1)
    else:
        raise RuntimeError(
            f"Unexpected feature shape at {path}: {features.shape}; expected "
            f"(N, {CLASSIFIER_WINDOW_EMBEDDINGS}, {EMBEDDING_DIM}) or (N, {EMBEDDING_DIM})"
        )
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)
    scores = _score_iter(sess, _external_batches(features, n_windows, batch_size))
    return _negative_report(path.name, scores, threshold)


def run(
    model_path: Path,
    threshold: float,
    run_dir: Path | None,
    external_features: list[Path],
    batch_size: int,
    max_windows: int | None,
    use_cuda: bool,
) -> dict:
    sess = _session(model_path, use_cuda=use_cuda)
    reports: list[dict] = []
    if run_dir is not None:
        for split in ("val", "train"):
            reports.extend(
                _stress_session_split(
                    sess,
                    run_dir,
                    split,
                    threshold,
                    batch_size,
                    max_windows,
                )
            )
    for path in external_features:
        reports.append(
            _stress_external_features(
                sess,
                path,
                threshold,
                batch_size,
                max_windows,
            )
        )

    return {
        "model": str(model_path),
        "threshold": float(threshold),
        "providers": sess.get_providers(),
        "window_seconds": WINDOW_SECONDS,
        "event_refractory_windows": FP_EVENT_REFRACTORY_WINDOWS,
        "reports": reports,
    }


def _print_human(report: dict) -> None:
    print(f"Model: {report['model']}")
    print(f"Providers: {', '.join(report['providers'])}")
    print(f"Threshold: {report['threshold']:.3f}")
    print()
    for item in report["reports"]:
        score = item.get("score", {})
        if item["kind"] == "negative":
            print(
                f"{item['name']}: {item['windows']:,} negative windows "
                f"({item['hours']:.2f} h), events={item['events']:,}, "
                f"FP/hr={item['fp_per_hour']:.3f}, "
                f"above={item['windows_above_threshold']:,}, "
                f"p99={score.get('p99', 0.0):.4f}, "
                f"p99.9={score.get('p99_9', 0.0):.4f}, "
                f"max={score.get('max', 0.0):.4f}"
            )
        else:
            units = "clips" if "clips" in item else "windows"
            print(
                f"{item['name']}: {item[units]:,} positive {units}, "
                f"recall={item['recall']:.3f}, "
                f"p10={score.get('p10', 0.0):.4f}, "
                f"p50={score.get('p50', 0.0):.4f}, "
                f"max={score.get('max', 0.0):.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test an exported ONNX model for false positives.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--external-feature", type=Path, action="append", default=[])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-windows", type=int, default=0, help="Limit each source to the first N windows; 0 means all.")
    parser.add_argument("--cpu", action="store_true", help="Disable CUDAExecutionProvider even when available.")
    parser.add_argument("--json", type=Path, help="Write the full report as JSON.")
    args = parser.parse_args()

    report = run(
        model_path=args.model,
        threshold=args.threshold,
        run_dir=args.run_dir,
        external_features=args.external_feature,
        batch_size=args.batch_size,
        max_windows=args.max_windows or None,
        use_cuda=not args.cpu,
    )
    _print_human(report)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
