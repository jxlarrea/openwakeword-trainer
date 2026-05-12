"""Cheap diagnostics for trained wake-word ONNX models.

This module intentionally works from already-built feature shards so it can
catch unusable models without re-running synthesis, augmentation, or training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.data.features import CLASSIFIER_WINDOW_EMBEDDINGS, EMBEDDING_DIM


def _session(model_path: Path) -> ort.InferenceSession:
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(model_path), providers=providers)


def _score_batch(sess: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    input_name = sess.get_inputs()[0].name
    try:
        return sess.run(None, {input_name: x.astype(np.float32)})[0].reshape(-1)
    except Exception:
        scores = []
        for i in range(x.shape[0]):
            out = sess.run(None, {input_name: x[i : i + 1].astype(np.float32)})[0]
            scores.append(float(out.reshape(-1)[0]))
        return np.asarray(scores, dtype=np.float32)


def _summary(name: str, scores: np.ndarray, threshold: float) -> dict:
    if scores.size == 0:
        return {"name": name, "n": 0}
    return {
        "name": name,
        "n": int(scores.size),
        "min": float(np.min(scores)),
        "p50": float(np.percentile(scores, 50)),
        "p95": float(np.percentile(scores, 95)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "above_threshold": int((scores >= threshold).sum()),
    }


def _sample_split(
    sess: ort.InferenceSession,
    run_dir: Path,
    split: str,
    limit_per_label: int,
    threshold: float,
    seed: int,
) -> list[dict]:
    labels_path = run_dir / f"{split}_labels.npy"
    features_path = run_dir / f"{split}_features.bin"
    if not labels_path.exists() or not features_path.exists():
        return []

    labels = np.load(labels_path, mmap_mode="r")
    features = np.memmap(
        features_path,
        dtype=np.float32,
        mode="r",
        shape=(int(labels.shape[0]), CLASSIFIER_WINDOW_EMBEDDINGS, EMBEDDING_DIM),
    )
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for label in (0, 1):
        ndcs = np.flatnonzero(np.asarray(labels[:]) == label)
        if ndcs.size == 0:
            continue
        if ndcs.size > limit_per_label:
            ndcs = rng.choice(ndcs, size=limit_per_label, replace=False)
        scores = _score_batch(sess, np.asarray(features[ndcs], dtype=np.float32))
        out.append(_summary(f"{split}_label_{label}", scores, threshold))
    return out


def run(model_path: Path, run_dir: Path | None, threshold: float, limit: int) -> dict:
    sess = _session(model_path)
    rng = np.random.default_rng(0)
    probes = {
        "feature_zeros": np.zeros((1, CLASSIFIER_WINDOW_EMBEDDINGS, EMBEDDING_DIM), dtype=np.float32),
        "feature_small_noise": (
            0.01
            * rng.standard_normal((1, CLASSIFIER_WINDOW_EMBEDDINGS, EMBEDDING_DIM))
        ).astype(np.float32),
        "feature_random_normal": rng.standard_normal(
            (limit, CLASSIFIER_WINDOW_EMBEDDINGS, EMBEDDING_DIM)
        ).astype(np.float32),
    }

    report = {
        "model": str(model_path),
        "threshold": threshold,
        "onnx_inputs": [
            {"name": i.name, "shape": i.shape, "type": i.type}
            for i in sess.get_inputs()
        ],
        "onnx_outputs": [
            {"name": o.name, "shape": o.shape, "type": o.type}
            for o in sess.get_outputs()
        ],
        "probes": [
            _summary(name, _score_batch(sess, x), threshold)
            for name, x in probes.items()
        ],
        "shards": [],
    }

    if run_dir is not None:
        report["run_dir"] = str(run_dir)
        for split in ("train", "val"):
            report["shards"].extend(
                _sample_split(sess, run_dir, split, limit, threshold, seed=42)
            )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=512)
    args = parser.parse_args()

    print(json.dumps(run(args.model, args.run_dir, args.threshold, args.limit), indent=2))


if __name__ == "__main__":
    main()
