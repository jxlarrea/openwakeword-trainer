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


def _pad_to_min_length(
    audio: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, int, int]:
    """Pad audio with silence so the result is at least MIN_CLIP_SAMPLES long.

    Returns (padded_audio, speech_start_samples, speech_end_samples). Speech
    range is the slice of the padded array that contains the original audio
    (the rest is zero-pad). If audio was already long enough, speech spans
    the whole array.
    """
    audio = audio.astype(np.float32)
    if audio.size >= MIN_CLIP_SAMPLES:
        return audio, 0, audio.size
    out = np.zeros(MIN_CLIP_SAMPLES, dtype=np.float32)
    max_offset = MIN_CLIP_SAMPLES - audio.size
    offset = int(rng.integers(0, max_offset + 1))
    out[offset : offset + audio.size] = audio
    return out, offset, offset + audio.size


# Each classifier window covers 16 embeddings x 8 mel-frame hop x 200 samples
# per mel frame at 16 kHz = 25,600 samples (1.6 s). Adjacent classifier windows
# differ by one embedding hop = 8 mel-frames = 1,600 samples (0.1 s).
_SAMPLES_PER_CLASSIFIER_WINDOW = CLASSIFIER_WINDOW_EMBEDDINGS * 8 * 200
_SAMPLES_PER_CLASSIFIER_HOP = 8 * 200


def _filter_speech_windows(
    windows: np.ndarray, speech_start: int, speech_end: int
) -> np.ndarray:
    """Keep only classifier-windows whose audio span overlaps the speech region.

    A positive clip padded to 3 s contains long stretches of silence. Without
    this filter, those silent windows get labeled positive and the model
    learns to fire on silence. We keep a window if at least half of the
    original speech is inside it; if none qualify (very short speech), we
    fall back to the single window closest to the speech center.
    """
    if windows.shape[0] == 0:
        return windows
    speech_len = max(1, speech_end - speech_start)
    keep: list[int] = []
    for k in range(windows.shape[0]):
        ws = k * _SAMPLES_PER_CLASSIFIER_HOP
        we = ws + _SAMPLES_PER_CLASSIFIER_WINDOW
        overlap = max(0, min(we, speech_end) - max(ws, speech_start))
        if overlap / speech_len >= 0.5:
            keep.append(k)
    if not keep:
        speech_center = (speech_start + speech_end) // 2
        closest = max(0, min(windows.shape[0] - 1, speech_center // _SAMPLES_PER_CLASSIFIER_HOP))
        keep = [closest]
    return windows[keep]


# ============================================================================
# Worker-pool parallelism for the CPU-heavy augmentation phase.
#
# Audiomentations' RIR convolution, MP3 codec, pitch shift, and time stretch
# each cost ~50-150 ms per clip and run single-threaded on CPU. Run them in a
# spawn-based process pool; results stream back to the main process which
# runs ONNX inference (mel + embedding) on GPU and writes to the memmap.
# ============================================================================

_WORKER_AUGMENTER = None     # populated by _augment_worker_init
_WORKER_EXTRACTOR = None     # populated by _augment_worker_init


def _augment_worker_init(
    aug_cfg_dump: dict,
    rir_dir_strs: list[str],
    bg_dir_strs: list[str],
    use_gpu: bool,
) -> None:
    """Pool initializer. Builds the augmenter and a FeatureExtractor per worker.

    Each worker gets its own ORT session (CUDA or CPU) so feature extraction
    runs end-to-end inside the worker; the parent only writes results to the
    memmap. This removes the per-variant Python/IPC overhead that was capping
    GPU utilization at ~18% in main-process-only mode.
    """
    global _WORKER_AUGMENTER, _WORKER_EXTRACTOR
    import warnings as _warnings

    from src.augment.augmenter import build_augmenter
    from src.config_schema import AugmentationConfig
    from src.data.features import FeatureExtractor

    _warnings.filterwarnings(
        "ignore",
        message=".*had to be resampled from .* Hz to .* Hz.*",
        category=UserWarning,
    )

    aug_cfg = AugmentationConfig(**aug_cfg_dump)
    rir_dirs = [Path(p) for p in rir_dir_strs]
    bg_dirs = [Path(p) for p in bg_dir_strs]
    _WORKER_AUGMENTER = build_augmenter(
        aug_cfg, rir_dirs=rir_dirs, background_noise_dirs=bg_dirs
    )

    providers = None  # auto-detect (prefers CUDA)
    if not use_gpu:
        providers = ["CPUExecutionProvider"]
    _WORKER_EXTRACTOR = FeatureExtractor(providers=providers)


def _augment_one_clip(task: dict) -> dict | None:
    """Worker entry point: read, pad, augment N times, run ONNX, filter windows.

    Returns a dict with serialized concatenated classifier-windows ready for
    the parent to drop straight into the memmap.
    """
    from math import gcd as _gcd

    import numpy as _np
    import soundfile as _sf
    from scipy.signal import resample_poly as _resample_poly

    try:
        wav_path = task["wav_path"]
        label = int(task["label"])
        n_augs = task["augmentations_per_clip"]
        seed = task["seed"]

        audio, sr = _sf.read(wav_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16_000:
            g = _gcd(sr, 16_000)
            audio = _resample_poly(audio, 16_000 // g, sr // g).astype(_np.float32)

        rng = _np.random.default_rng(seed)
        audio, speech_start, speech_end = _pad_to_min_length(audio, rng)

        all_windows: list[_np.ndarray] = []
        for _ in range(n_augs):
            try:
                aug = _WORKER_AUGMENTER(samples=audio.astype(_np.float32), sample_rate=16_000)
                _np.clip(aug, -1.0, 1.0, out=aug)
            except Exception:
                aug = audio
            int16 = float32_to_int16(aug)
            windows = _WORKER_EXTRACTOR.classifier_inputs(int16)
            if label == 1:
                windows = _filter_speech_windows(windows, speech_start, speech_end)
            if windows.shape[0] > 0:
                all_windows.append(windows)

        if not all_windows:
            return {"n_windows": 0, "windows_bytes": b""}

        combined = _np.concatenate(all_windows, axis=0).astype(_np.float32, copy=False)
        return {
            "n_windows": int(combined.shape[0]),
            "windows_bytes": combined.tobytes(),
        }
    except Exception:
        return None


def build_features_from_wavs_parallel(
    wav_paths: list[Path],
    label: int,
    extractor: FeatureExtractor,                  # kept for API compat; unused
    augmentation_cfg,                             # AugmentationConfig
    rir_dirs: list[Path],
    bg_dirs: list[Path],
    augmentations_per_clip: int,
    out_features: np.memmap,
    out_labels: np.ndarray,
    write_offset: int,
    workers: int,
    cancel_flag=None,
    use_gpu_in_workers: bool = True,
    progress_label: str = "features:extract",
    progress_fraction_base: float = 0.0,
    progress_fraction_span: float = 1.0,
) -> int:
    """Full end-to-end pipeline in workers (augment + mel + embedding + filter).

    Main process only writes pre-computed classifier windows to the memmap.
    With CUDA available in each worker (one CUDA context per process), this
    keeps the GPU fed continuously rather than waiting on Python overhead in
    a single drainer thread.
    """
    import time as _time
    from multiprocessing import get_context

    from src.train.progress import bus

    cursor = write_offset
    total = len(wav_paths)
    if total == 0:
        return cursor

    tasks = [
        {
            "wav_path": str(p),
            "label": label,
            "augmentations_per_clip": augmentations_per_clip,
            "seed": (hash((str(p), label, write_offset)) & 0xFFFFFFFF),
        }
        for p in wav_paths
    ]

    aug_cfg_dump = augmentation_cfg.model_dump()
    rir_dir_strs = [str(p) for p in rir_dirs]
    bg_dir_strs = [str(p) for p in bg_dirs]

    bus.log(
        f"features: parallel mode, {workers} workers (GPU in workers: {use_gpu_in_workers}), "
        f"{total} clips, {augmentations_per_clip} augs each"
    )

    ctx = get_context("spawn")
    completed = 0
    last_progress_t = _time.monotonic()
    feature_shape_tail = out_features.shape[1:]  # (16, 96)

    with ctx.Pool(
        processes=workers,
        initializer=_augment_worker_init,
        initargs=(aug_cfg_dump, rir_dir_strs, bg_dir_strs, use_gpu_in_workers),
    ) as pool:
        for result in pool.imap_unordered(_augment_one_clip, tasks, chunksize=2):
            if cancel_flag is not None and cancel_flag.is_set():
                pool.terminate()
                break
            completed += 1

            if result is None:
                continue

            n_new = result["n_windows"]
            if n_new == 0:
                continue

            windows_flat = np.frombuffer(result["windows_bytes"], dtype=np.float32)
            windows = windows_flat.reshape((n_new,) + feature_shape_tail)

            if cursor + n_new > out_features.shape[0]:
                n_new = out_features.shape[0] - cursor
                windows = windows[:n_new]
            out_features[cursor : cursor + n_new] = windows
            out_labels[cursor : cursor + n_new] = label
            cursor += n_new

            now = _time.monotonic()
            if completed % 100 == 0 or (now - last_progress_t) > 5.0:
                done_frac = completed / max(1, total)
                global_frac = progress_fraction_base + progress_fraction_span * done_frac
                bus.progress(
                    progress_label,
                    global_frac,
                    detail=f"{completed}/{total} clips, {cursor:,} windows",
                )
                bus.log(
                    f"features: {completed}/{total} clips ({100.0 * completed / total:.1f}%), "
                    f"{cursor:,} windows written"
                )
                last_progress_t = now

            if cursor >= out_features.shape[0]:
                break

    return cursor


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
    progress_label: str = "features:extract",
    progress_fraction_base: float = 0.0,
    progress_fraction_span: float = 1.0,
) -> int:
    """Read each wav, augment + extract features, write to the memmap.

    Returns the new write_offset. Polls cancel_flag (a threading.Event) every
    clip so the orchestrator can interrupt long extraction passes. Emits
    progress + log events through the EventBus every 50 clips so the Web UI
    sees the phase advancing.
    """
    import time as _time

    from src.augment.augmenter import augment_clip
    from src.train.progress import bus

    cursor = write_offset
    rng = np.random.default_rng(write_offset + label)
    total = len(wav_paths)
    last_progress_t = _time.monotonic()

    for clip_idx, path in enumerate(wav_paths):
        if cancel_flag is not None and cancel_flag.is_set():
            break

        # Periodic progress + log, every 50 clips or 5 seconds (whichever first).
        now = _time.monotonic()
        if clip_idx > 0 and (clip_idx % 50 == 0 or (now - last_progress_t) > 5.0):
            done_frac = clip_idx / max(1, total)
            global_frac = progress_fraction_base + progress_fraction_span * done_frac
            bus.progress(
                progress_label,
                global_frac,
                detail=f"{clip_idx}/{total} clips, {cursor:,} windows",
            )
            bus.log(
                f"features: {clip_idx}/{total} clips processed "
                f"({cursor:,} windows written)"
            )
            last_progress_t = now
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
        # Track the speech region so we can drop silent-padding windows for positives.
        audio, speech_start, speech_end = _pad_to_min_length(audio, rng)

        for variant in augment_clip(audio, 16_000, augmenter, n_variants=augmentations_per_clip):
            int16 = float32_to_int16(variant)
            windows = extractor.classifier_inputs(int16)
            # CRITICAL: for positive clips, drop classifier windows that are
            # mostly silent padding. Without this, the model learns
            # "silence = wake word" because each positive's 6 windows include
            # 3-5 silence-only ones all labeled 1. Negatives keep all windows
            # since silence-labeled-negative is a useful training signal.
            if label == 1:
                windows = _filter_speech_windows(windows, speech_start, speech_end)
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
