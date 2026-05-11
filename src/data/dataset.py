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

# Pad short clips to this many samples (3 s at 16 kHz). The openWakeWord
# classifier window covers ~1.28 s, so we need clips at least ~2 s to yield even
# one window. Padding with silence on both sides also adds positional variability.
MIN_CLIP_SAMPLES = 16_000 * 3


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
        n = int(labels.shape[0])
        if n == 0:
            raise RuntimeError(
                f"Empty dataset shard at {features_path}. "
                "No classifier-windows were produced - check that audio clips are long enough "
                "(>= 2 s) and that augmentations are not silencing them."
            )
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


def _pad_to_min_length(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pad audio with silence so the result is at least MIN_CLIP_SAMPLES long.

    The original clip is placed at a random offset; trailing samples are zeros.
    Returns audio unchanged if it is already long enough.
    """
    if audio.size >= MIN_CLIP_SAMPLES:
        return audio
    out = np.zeros(MIN_CLIP_SAMPLES, dtype=np.float32)
    max_offset = MIN_CLIP_SAMPLES - audio.size
    offset = int(rng.integers(0, max_offset + 1))
    out[offset : offset + audio.size] = audio.astype(np.float32)
    return out


def build_features_from_wavs(
    wav_paths: list[Path],
    label: int,
    extractor: FeatureExtractor,
    augmenter,
    augmentations_per_clip: int,
    out_features: np.memmap,
    out_labels: np.ndarray,
    write_offset: int,
    cancel_flag=None,
) -> int:
    """Read each wav, augment + extract features, write to the memmap.

    Returns the new write_offset. Polls cancel_flag (a threading.Event) every
    clip so the orchestrator can interrupt long extraction passes.
    """
    from src.augment.augmenter import augment_clip

    cursor = write_offset
    rng = np.random.default_rng(write_offset + label)
    for path in wav_paths:
        if cancel_flag is not None and cancel_flag.is_set():
            break
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

        # Ensure each clip is long enough to produce at least one classifier window.
        audio = _pad_to_min_length(audio, rng)

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


def estimate_window_count(n_clips: int, augmentations_per_clip: int, avg_windows_per_clip: int = 6) -> int:
    """Conservative estimate of how many classifier-windows a clip set will yield.

    With clips padded to MIN_CLIP_SAMPLES (3 s) and 12.5 ms mel hop:
      ~240 mel-frames -> 21 embedding windows -> 6 classifier-windows per clip.
    """
    return max(1, n_clips * augmentations_per_clip * avg_windows_per_clip)


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
