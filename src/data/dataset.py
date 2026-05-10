"""Pre-computed feature dataset for training.

Workflow per training run:
  1. Generate positive WAVs (Piper + optional ElevenLabs)
  2. Generate adversarial WAVs (same TTS path)
  3. Build the augmenter (RIRs + background-noise pools)
  4. For each clip, augment N times, extract (M, 16, 96) classifier inputs,
     append to a memmap. Label per row.
  5. Save train / val / test memmap shards.

The memmap layout is one big float32 array of shape (total_windows, 16, 96)
plus a uint8 labels array of shape (total_windows,).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset

from src.data.features import (
    CLASSIFIER_WINDOW_EMBEDDINGS,
    EMBEDDING_DIM,
    FeatureExtractor,
    float32_to_int16,
)

logger = logging.getLogger(__name__)


@dataclass
class ShardManifest:
    features_path: Path
    labels_path: Path
    n_windows: int
    label_counts: dict[int, int]


class FeatureMemmapDataset(Dataset):
    """Reads precomputed (windows, 16, 96) float32 + (windows,) uint8 memmaps."""

    def __init__(self, features_path: Path, labels_path: Path) -> None:
        labels = np.load(labels_path, mmap_mode="r")
        n = labels.shape[0]
        self.features = np.memmap(
            features_path,
            dtype=np.float32,
            mode="r",
            shape=(n, CLASSIFIER_WINDOW_EMBEDDINGS, EMBEDDING_DIM),
        )
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        x = np.asarray(self.features[idx], dtype=np.float32)
        y = float(self.labels[idx])
        return x, y


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype="PCM_16")


def build_features_from_wavs(
    wav_paths: list[Path],
    label: int,
    extractor: FeatureExtractor,
    augmenter,
    augmentations_per_clip: int,
    out_features: np.memmap,
    out_labels: np.ndarray,
    write_offset: int,
) -> int:
    """Read each wav, augment + extract features, write to the memmap.

    Returns the new write_offset.
    """
    from src.augment.augmenter import augment_clip

    cursor = write_offset
    for path in wav_paths:
        try:
            audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16_000:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, 16_000)
            audio = resample_poly(audio, 16_000 // g, sr // g).astype(np.float32)

        for variant in augment_clip(audio, 16_000, augmenter, n_variants=augmentations_per_clip):
            int16 = float32_to_int16(variant)
            windows = extractor.classifier_inputs(int16)
            if windows.shape[0] == 0:
                continue
            n_new = windows.shape[0]
            if cursor + n_new > out_features.shape[0]:
                # Truncate to fit the pre-allocated memmap.
                n_new = out_features.shape[0] - cursor
                windows = windows[:n_new]
            out_features[cursor : cursor + n_new] = windows
            out_labels[cursor : cursor + n_new] = label
            cursor += n_new
            if cursor >= out_features.shape[0]:
                break
        if cursor >= out_features.shape[0]:
            break

    return cursor


def estimate_window_count(n_clips: int, augmentations_per_clip: int, avg_windows_per_clip: int = 8) -> int:
    """Conservative estimate of how many classifier-windows a clip set will yield."""
    return n_clips * augmentations_per_clip * avg_windows_per_clip


def allocate_memmap(path: Path, n_windows: int) -> tuple[np.memmap, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.memmap(
        path,
        dtype=np.float32,
        mode="w+",
        shape=(n_windows, CLASSIFIER_WINDOW_EMBEDDINGS, EMBEDDING_DIM),
    )
    return arr, path


def save_manifest(run_dir: Path, manifests: dict[str, ShardManifest]) -> None:
    out = {
        split: {
            "features": str(m.features_path.relative_to(run_dir)),
            "labels": str(m.labels_path.relative_to(run_dir)),
            "n_windows": m.n_windows,
            "label_counts": m.label_counts,
        }
        for split, m in manifests.items()
    }
    (run_dir / "shards.json").write_text(json.dumps(out, indent=2))
