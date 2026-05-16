"""Pre-computed feature dataset for training.

Workflow per training run:
  1. Generate positive WAVs (Piper + Kokoro)
  2. Generate adversarial WAVs (same TTS path)
  3. Build the augmenter (RIRs + background-noise pools)
  4. For each clip, augment N times, align to one fixed ~2 s training window,
     extract one (16, 96) classifier input, append to a memmap. Label per row.
  5. Save train / val / test memmap shards.

The memmap layout is one big float32 array of shape (total_examples, 16, 96)
plus a uint8 labels array of shape (total_examples,).
"""
from __future__ import annotations

import json
import logging
import hashlib
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

# Reference openWakeWord-style training uses fixed clips. Positives can fan
# out to adjacent trailing windows so the model learns a wider activation
# region; negatives stay single-window to avoid making random audio easier to
# trigger.
TRAINING_CLIP_SAMPLES = 16_000 * 2
_POSITIVE_END_JITTER_SAMPLES = int(0.2 * 16_000)
DEFAULT_POSITIVE_CONTEXT_SAMPLES = 16_000 * 3


@dataclass
class ShardManifest:
    features_path: Path
    labels_path: Path
    n_windows: int
    label_counts: dict[int, int]
    source_ids_path: Path | None = None


class FeatureMemmapDataset(Dataset):
    """Reads precomputed (windows, 16, 96) float32 + (windows,) uint8 memmaps."""

    def __init__(
        self,
        features_path: Path,
        labels_path: Path,
        source_ids_path: Path | None = None,
    ) -> None:
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
        if source_ids_path is None:
            inferred = labels_path.with_name(
                labels_path.name.replace("_labels.npy", "_source_ids.npy")
            )
            source_ids_path = inferred if inferred.exists() else None
        self.source_ids = (
            np.load(source_ids_path, mmap_mode="r")
            if source_ids_path is not None and source_ids_path.exists()
            else None
        )

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        x = np.asarray(self.features[idx], dtype=np.float32)
        y = float(self.labels[idx])
        return x, y


class ExternalNegativeFeatureDataset(Dataset):
    """Reads an openWakeWord `.npy` feature bank as label-0 windows."""

    def __init__(self, features_path: Path) -> None:
        self.features = np.load(features_path, mmap_mode="r")
        if self.features.ndim == 3 and self.features.shape[1:] == (
            CLASSIFIER_WINDOW_EMBEDDINGS,
            EMBEDDING_DIM,
        ):
            self._mode = "windows"
            n = int(self.features.shape[0])
        elif self.features.ndim == 2 and self.features.shape[1] == EMBEDDING_DIM:
            self._mode = "embeddings"
            n = max(0, int(self.features.shape[0]) - CLASSIFIER_WINDOW_EMBEDDINGS + 1)
        else:
            raise RuntimeError(
                f"Unexpected external feature shape at {features_path}: "
                f"{self.features.shape}; expected "
                f"(N, {CLASSIFIER_WINDOW_EMBEDDINGS}, {EMBEDDING_DIM}) windows "
                f"or (N, {EMBEDDING_DIM}) embedding frames"
            )
        if n <= 0:
            raise RuntimeError(f"External feature bank is empty or too short: {features_path}")
        self.labels = np.zeros(n, dtype=np.uint8)
        self.source_ids = None

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        if self._mode == "windows":
            x = np.asarray(self.features[idx], dtype=np.float32)
        else:
            x = np.asarray(
                self.features[idx : idx + CLASSIFIER_WINDOW_EMBEDDINGS],
                dtype=np.float32,
            )
        return x, 0.0

    def get_features(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        if self._mode == "windows":
            return np.asarray(self.features[indices], dtype=np.float32)
        out = np.empty(
            (indices.size, CLASSIFIER_WINDOW_EMBEDDINGS, EMBEDDING_DIM),
            dtype=np.float32,
        )
        for i, idx in enumerate(indices):
            out[i] = np.asarray(
                self.features[idx : idx + CLASSIFIER_WINDOW_EMBEDDINGS],
                dtype=np.float32,
            )
        return out


class CombinedFeatureDataset(Dataset):
    """Concatenate feature datasets while preserving labels/source IDs."""

    def __init__(self, datasets: list[Dataset]) -> None:
        self.datasets = [d for d in datasets if len(d) > 0]
        if not self.datasets:
            raise RuntimeError("CombinedFeatureDataset requires at least one non-empty dataset")
        lengths = [len(d) for d in self.datasets]
        self._offsets = np.cumsum([0] + lengths)
        self.labels = np.concatenate(
            [np.asarray(getattr(d, "labels")[:], dtype=np.uint8) for d in self.datasets]
        )
        source_parts: list[np.ndarray] = []
        source_base = 0
        for d in self.datasets:
            n = len(d)
            src = getattr(d, "source_ids", None)
            labels = np.asarray(getattr(d, "labels")[:], dtype=np.uint8)
            if src is None:
                part = np.full(n, -1, dtype=np.int64)
            else:
                part = np.asarray(src[:], dtype=np.int64)
                valid = part >= 0
                # Keep positive source IDs unique across child datasets.
                part = part.copy()
                part[valid] += source_base
                if valid.any():
                    source_base = int(part[valid].max()) + 1
            source_parts.append(part)
        self.source_ids = np.concatenate(source_parts)

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def _locate(self, idx: int) -> tuple[Dataset, int]:
        if idx < 0:
            idx += len(self)
        ds_idx = int(np.searchsorted(self._offsets, idx, side="right") - 1)
        return self.datasets[ds_idx], int(idx - self._offsets[ds_idx])

    def __getitem__(self, idx: int):
        ds, local_idx = self._locate(idx)
        return ds[local_idx]

    def get_features(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        out = np.empty(
            (indices.size, CLASSIFIER_WINDOW_EMBEDDINGS, EMBEDDING_DIM),
            dtype=np.float32,
        )
        for ds_idx, ds in enumerate(self.datasets):
            start = self._offsets[ds_idx]
            end = self._offsets[ds_idx + 1]
            mask = (indices >= start) & (indices < end)
            if not mask.any():
                continue
            local = indices[mask] - start
            out[mask] = _dataset_features_at(ds, local)
        return out


def _dataset_features_at(dataset: Dataset, indices: np.ndarray) -> np.ndarray:
    if hasattr(dataset, "get_features"):
        return dataset.get_features(indices)
    return np.asarray(getattr(dataset, "features")[indices], dtype=np.float32)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype="PCM_16")


def _active_audio_bounds(audio: np.ndarray) -> tuple[int, int]:
    """Return a conservative non-silence region for a synthesized clip."""
    if audio.size == 0:
        return 0, 0
    abs_audio = np.abs(audio.astype(np.float32))
    peak = float(abs_audio.max(initial=0.0))
    if peak <= 1e-5:
        return 0, audio.size
    threshold = max(1e-4, peak * 0.02)
    active = np.flatnonzero(abs_audio >= threshold)
    if active.size == 0:
        return 0, audio.size

    # Include a little edge context so quiet consonants do not get clipped out
    # of the timing estimate.
    margin = int(0.08 * 16_000)
    start = max(0, int(active[0]) - margin)
    end = min(audio.size, int(active[-1]) + margin + 1)
    return start, max(start + 1, end)


def _align_training_clip(
    audio: np.ndarray,
    label: int,
    rng: np.random.Generator,
    active_start: int | None = None,
    active_end: int | None = None,
    positive_context_samples: int = DEFAULT_POSITIVE_CONTEXT_SAMPLES,
) -> tuple[np.ndarray, int, int]:
    """Return one fixed-size training clip and speech bounds inside it.

    Positives are end-aligned with small jitter so the model learns the
    streaming question: "did the wake word just finish?" Negatives are centered
    when short and randomly cropped when long.
    """
    audio = audio.astype(np.float32)
    if active_start is None or active_end is None:
        active_start, active_end = _active_audio_bounds(audio)
    active_start = max(0, min(int(active_start), audio.size))
    active_end = max(active_start + 1, min(int(active_end), audio.size))

    if label == 1:
        speech = audio[active_start:active_end]
        if speech.size >= TRAINING_CLIP_SAMPLES:
            speech = speech[-TRAINING_CLIP_SAMPLES:]
        out = np.zeros(TRAINING_CLIP_SAMPLES, dtype=np.float32)
        jitter = int(rng.integers(0, _POSITIVE_END_JITTER_SAMPLES + 1))
        end_pos = max(1, TRAINING_CLIP_SAMPLES - jitter)
        start_pos = max(0, end_pos - speech.size)
        clip_start = max(0, speech.size - (end_pos - start_pos))
        used = speech[clip_start : clip_start + (end_pos - start_pos)]
        out[start_pos : start_pos + used.size] = used
        return out, start_pos, start_pos + used.size

    if audio.size >= TRAINING_CLIP_SAMPLES:
        max_start = audio.size - TRAINING_CLIP_SAMPLES
        if active_end - active_start < TRAINING_CLIP_SAMPLES:
            center = (active_start + active_end) // 2
            start = center - TRAINING_CLIP_SAMPLES // 2
            jitter = int(rng.integers(-TRAINING_CLIP_SAMPLES // 8, TRAINING_CLIP_SAMPLES // 8 + 1))
            start += jitter
            start = max(0, min(max_start, start))
        else:
            start = int(rng.integers(active_start, max(active_start + 1, active_end - TRAINING_CLIP_SAMPLES + 1)))
            start = max(0, min(max_start, start))
        out = audio[start : start + TRAINING_CLIP_SAMPLES].astype(np.float32, copy=False)
        return out, max(0, active_start - start), max(1, min(TRAINING_CLIP_SAMPLES, active_end - start))

    out = np.zeros(TRAINING_CLIP_SAMPLES, dtype=np.float32)
    offset = (TRAINING_CLIP_SAMPLES - audio.size) // 2
    out[offset : offset + audio.size] = audio
    return out, offset + active_start, offset + active_end


def _place_positive_streaming_clip_with_bounds(
    audio: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, int]:
    """Place a positive phrase into a streaming-style 3 s buffer.

    The fixed trailing windows teach the ideal "wake word just finished"
    alignment. These streaming placements teach the model to stay active as a
    rolling inference buffer advances across a spoken phrase.
    """
    active_start, active_end = _active_audio_bounds(audio)
    speech = audio[active_start:active_end].astype(np.float32, copy=False)
    if speech.size == 0:
        speech = audio.astype(np.float32, copy=False)
    lead_samples = int(rng.integers(int(0.4 * 16_000), int(1.2 * 16_000) + 1))
    tail_samples = int(1.2 * 16_000)
    total_samples = max(DEFAULT_POSITIVE_CONTEXT_SAMPLES, lead_samples + int(speech.size) + tail_samples)
    out = np.zeros(total_samples, dtype=np.float32)
    end = min(total_samples, lead_samples + int(speech.size))
    out[lead_samples:end] = speech[: max(0, end - lead_samples)]
    return out, lead_samples, end


def _place_positive_streaming_clip(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out, _, _ = _place_positive_streaming_clip_with_bounds(audio, rng)
    return out


def _select_positive_streaming_windows(
    windows: np.ndarray,
    speech_end_sample: int,
    n_windows: int,
    stride_embeddings: int,
) -> np.ndarray:
    """Select streaming positive windows starting at wake-word completion.

    classifier_inputs() returns every rolling 1.28 s window from the streaming
    clip indexed by win_start. The window whose TRAILING EDGE lands AT
    speech_end has win_start = speech_end_embedding (because the trailing
    1.28 s ends exactly at speech_end). The model needs to recognize "wake
    word just ended" for several adjacent rolling positions: walk FORWARD
    from this anchor into the post-speech tail, where the wake word sits
    increasingly toward the lead of the rolling 1.28 s buffer until it slides
    out. Walking backward (the previous behaviour) trained the classifier on
    windows whose trailing edge had not yet reached the wake word, i.e.
    silence-as-positive.
    """
    n_windows = max(1, int(n_windows))
    stride_embeddings = max(1, int(stride_embeddings))
    if windows.shape[0] == 0:
        return windows

    hop_samples = int(0.08 * 16_000)
    anchor_index = int(round(speech_end_sample / hop_samples)) - CLASSIFIER_WINDOW_EMBEDDINGS
    anchor_index = max(0, min(windows.shape[0] - 1, anchor_index))
    indices: list[int] = []
    for k in range(n_windows):
        idx = anchor_index + k * stride_embeddings
        if idx >= windows.shape[0]:
            break
        if not indices or idx != indices[-1]:
            indices.append(idx)
    if not indices:
        return windows[:1]
    return windows[np.asarray(indices, dtype=np.int64)]


FEATURE_LABELING_VERSION = "fixed_end_aligned_examples_v2"


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
_WORKER_AUG_CFG = None       # populated by _augment_worker_init
_WORKER_POSITIVE_TEMPORAL_WINDOWS = 1
_WORKER_POSITIVE_TEMPORAL_STRIDE = 1
_WORKER_POSITIVE_CONTEXT_SAMPLES = DEFAULT_POSITIVE_CONTEXT_SAMPLES


def _augment_worker_init(
    aug_cfg_dump: dict,
    rir_dir_strs: list[str],
    bg_dir_strs: list[str],
    use_gpu: bool,
    positive_temporal_windows: int = 1,
    positive_temporal_stride_embeddings: int = 1,
    positive_context_seconds: float = 3.0,
) -> None:
    """Pool initializer. Builds the augmenter and a FeatureExtractor per worker.

    Each worker gets its own ORT session (CUDA or CPU) so feature extraction
    runs end-to-end inside the worker; the parent only writes results to the
    memmap. This removes the per-variant Python/IPC overhead that was capping
    GPU utilization at ~18% in main-process-only mode.
    """
    global _WORKER_AUGMENTER, _WORKER_EXTRACTOR, _WORKER_AUG_CFG
    global _WORKER_POSITIVE_TEMPORAL_WINDOWS, _WORKER_POSITIVE_TEMPORAL_STRIDE
    global _WORKER_POSITIVE_CONTEXT_SAMPLES
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
    _WORKER_AUG_CFG = aug_cfg
    rir_dirs = [Path(p) for p in rir_dir_strs]
    bg_dirs = [Path(p) for p in bg_dir_strs]
    _WORKER_AUGMENTER = build_augmenter(
        aug_cfg, rir_dirs=rir_dirs, background_noise_dirs=bg_dirs
    )

    providers = None  # auto-detect (prefers CUDA)
    if not use_gpu:
        providers = ["CPUExecutionProvider"]
    _WORKER_EXTRACTOR = FeatureExtractor(providers=providers)
    _WORKER_POSITIVE_TEMPORAL_WINDOWS = max(1, int(positive_temporal_windows))
    _WORKER_POSITIVE_TEMPORAL_STRIDE = max(1, int(positive_temporal_stride_embeddings))
    _WORKER_POSITIVE_CONTEXT_SAMPLES = max(
        TRAINING_CLIP_SAMPLES,
        int(float(positive_context_seconds) * 16_000),
    )


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
        source_id = int(task.get("source_id", -1))
        task_index = int(task.get("task_index", -1))

        audio, sr = _sf.read(wav_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16_000:
            g = _gcd(sr, 16_000)
            audio = _resample_poly(audio, 16_000 // g, sr // g).astype(_np.float32)

        rng = _np.random.default_rng(seed)
        speech_start = 0
        speech_end = 0
        audio, speech_start, speech_end = _align_training_clip(
            audio,
            label,
            rng,
            positive_context_samples=_WORKER_POSITIVE_CONTEXT_SAMPLES,
        )

        all_windows: list[_np.ndarray] = []
        for _ in range(n_augs):
            try:
                aug = _WORKER_AUGMENTER(samples=audio.astype(_np.float32), sample_rate=16_000)
                from src.augment.augmenter import apply_tablet_far_field_effect

                aug = apply_tablet_far_field_effect(aug, 16_000, _WORKER_AUG_CFG)
                _np.clip(aug, -1.0, 1.0, out=aug)
            except Exception:
                aug = audio
            int16 = float32_to_int16(aug)
            feature = _WORKER_EXTRACTOR.fixed_classifier_input(int16)[None, ...]
            all_windows.append(feature)

        if not all_windows:
            return {"n_windows": 0, "windows_bytes": b"", "task_index": task_index}

        combined = _np.concatenate(all_windows, axis=0).astype(_np.float32, copy=False)
        return {
            "n_windows": int(combined.shape[0]),
            "windows_bytes": combined.tobytes(),
            "source_id": source_id,
            "task_index": task_index,
        }
    except Exception:
        return {
            "n_windows": 0,
            "windows_bytes": b"",
            "source_id": int(task.get("source_id", -1)),
            "task_index": int(task.get("task_index", -1)),
        }


def _stable_feature_seed(path: Path, label: int, source_id: int, augmentations_per_clip: int) -> int:
    blob = f"{path}|{label}|{source_id}|{augmentations_per_clip}".encode("utf-8")
    return int(hashlib.sha1(blob).hexdigest()[:8], 16)


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
    out_source_ids: np.ndarray | None,
    write_offset: int,
    workers: int,
    cancel_flag=None,
    use_gpu_in_workers: bool = True,
    progress_label: str = "features:extract",
    progress_fraction_base: float = 0.0,
    progress_fraction_span: float = 1.0,
    source_id_base: int = 0,
    completed_clip_indices: set[int] | None = None,
    checkpoint_callback=None,
    checkpoint_every: int = 500,
    positive_temporal_windows: int = 1,
    positive_temporal_stride_embeddings: int = 1,
    positive_context_seconds: float = 3.0,
) -> int:
    """Full end-to-end pipeline in workers (align + augment + mel + embedding).

    Main process only writes pre-computed fixed classifier examples to the memmap.
    With CUDA available in each worker (one CUDA context per process), this
    keeps the GPU fed continuously rather than waiting on Python overhead in
    a single drainer thread.
    """
    import time as _time
    from multiprocessing import TimeoutError as PoolTimeoutError
    from multiprocessing import get_context

    from src.train.progress import bus

    cursor = write_offset
    total = len(wav_paths)
    if total == 0:
        return cursor
    completed_indices = completed_clip_indices if completed_clip_indices is not None else set()
    completed = len(completed_indices)
    if completed >= total:
        return cursor

    tasks = [
        {
            "wav_path": str(p),
            "label": label,
            "augmentations_per_clip": augmentations_per_clip,
            "seed": _stable_feature_seed(p, label, source_id_base + i, augmentations_per_clip),
            "source_id": source_id_base + i,
            "task_index": i,
        }
        for i, p in enumerate(wav_paths)
        if i not in completed_indices
    ]

    aug_cfg_dump = augmentation_cfg.model_dump()
    rir_dir_strs = [str(p) for p in rir_dirs]
    bg_dir_strs = [str(p) for p in bg_dirs]

    bus.log(
        f"features: parallel mode, {workers} workers (GPU in workers: {use_gpu_in_workers}), "
        f"{total} clips, {augmentations_per_clip} augs each"
    )

    ctx = get_context("spawn")
    last_progress_t = _time.monotonic()
    feature_shape_tail = out_features.shape[1:]  # (16, 96)
    last_checkpoint_completed = completed

    with ctx.Pool(
        processes=workers,
        initializer=_augment_worker_init,
        initargs=(
            aug_cfg_dump,
            rir_dir_strs,
            bg_dir_strs,
            use_gpu_in_workers,
            positive_temporal_windows,
            positive_temporal_stride_embeddings,
            positive_context_seconds,
        ),
    ) as pool:
        iterator = pool.imap_unordered(_augment_one_clip, tasks, chunksize=1)
        while True:
            if cancel_flag is not None and cancel_flag.is_set():
                pool.terminate()
                break

            try:
                result = iterator.next(timeout=1.0)
            except StopIteration:
                break
            except PoolTimeoutError:
                continue

            task_index = int(result.get("task_index", -1)) if result is not None else -1
            n_new = int(result.get("n_windows", 0)) if result is not None else 0
            if n_new == 0:
                windows = None
            else:
                windows_flat = np.frombuffer(result["windows_bytes"], dtype=np.float32)
                windows = windows_flat.reshape((n_new,) + feature_shape_tail)

                if cursor + n_new > out_features.shape[0]:
                    pool.terminate()
                    break
                out_features[cursor : cursor + n_new] = windows
                out_labels[cursor : cursor + n_new] = label
                if out_source_ids is not None:
                    out_source_ids[cursor : cursor + n_new] = int(result.get("source_id", -1))
                cursor += n_new

            if task_index >= 0:
                completed_indices.add(task_index)
            completed = len(completed_indices)

            if (
                checkpoint_callback is not None
                and completed - last_checkpoint_completed >= max(1, checkpoint_every)
            ):
                checkpoint_callback(cursor, completed_indices, completed)
                last_checkpoint_completed = completed

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

    if checkpoint_callback is not None and completed != last_checkpoint_completed:
        checkpoint_callback(cursor, completed_indices, completed)

    return cursor


def build_features_from_wavs(
    wav_paths: list[Path],
    label: int,
    extractor: FeatureExtractor,
    augmenter,
    augmentations_per_clip: int,
    out_features: np.memmap,
    out_labels: np.ndarray,
    out_source_ids: np.ndarray | None,
    write_offset: int,
    cancel_flag=None,
    progress_label: str = "features:extract",
    progress_fraction_base: float = 0.0,
    progress_fraction_span: float = 1.0,
    source_id_base: int = 0,
    positive_temporal_windows: int = 1,
    positive_temporal_stride_embeddings: int = 1,
    positive_context_seconds: float = 3.0,
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

        # Place positives with leading silence + speech + trailing tail so the
        # fan-out can walk forward from speech_end into the tail. Negatives
        # are centered or randomly cropped to a single fixed window.
        audio, speech_start, speech_end = _align_training_clip(
            audio,
            label,
            rng,
            positive_context_samples=max(TRAINING_CLIP_SAMPLES, int(float(positive_context_seconds) * 16_000)),
        )

        for variant in augment_clip(
            audio,
            16_000,
            augmenter,
            augmentation_cfg=augmentation_cfg,
            n_variants=augmentations_per_clip,
        ):
            int16 = float32_to_int16(variant)
            windows = extractor.fixed_classifier_input(int16)[None, ...]
            n_new = int(windows.shape[0])
            if cursor + n_new > out_features.shape[0]:
                # Truncate to fit the pre-allocated memmap.
                n_new = out_features.shape[0] - cursor
                windows = windows[:n_new]
            out_features[cursor : cursor + n_new] = windows
            out_labels[cursor : cursor + n_new] = label
            if out_source_ids is not None:
                out_source_ids[cursor : cursor + n_new] = source_id_base + clip_idx
            cursor += n_new
            if cursor >= out_features.shape[0]:
                break
        if cursor >= out_features.shape[0]:
            break

    return cursor


def estimate_window_count(n_clips: int, augmentations_per_clip: int, avg_windows_per_clip: int = 1) -> int:
    """Estimate fixed examples produced by the reference-style extractor.

    Every augmented WAV contributes exactly one (16, 96) classifier input.
    """
    return max(1, n_clips * augmentations_per_clip * avg_windows_per_clip)


def allocate_memmap(path: Path, n_windows: int, mode: str = "w+") -> tuple[np.memmap, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.memmap(
        path,
        dtype=np.float32,
        mode=mode,
        shape=(n_windows, CLASSIFIER_WINDOW_EMBEDDINGS, EMBEDDING_DIM),
    )
    return arr, path


def save_manifest(run_dir: Path, manifests: dict[str, ShardManifest]) -> None:
    out = {
        split: {
            "features": str(m.features_path.relative_to(run_dir)),
            "labels": str(m.labels_path.relative_to(run_dir)),
            "source_ids": (
                str(m.source_ids_path.relative_to(run_dir))
                if m.source_ids_path is not None
                else None
            ),
            "n_windows": m.n_windows,
            "label_counts": m.label_counts,
        }
        for split, m in manifests.items()
    }
    (run_dir / "shards.json").write_text(json.dumps(out, indent=2))
