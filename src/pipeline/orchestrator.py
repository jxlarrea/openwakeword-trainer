"""End-to-end training run.

Phases:
  0. Prepare run directory + persist config
  1. Generate positive samples (Piper + Kokoro)
  2. Generate negative / adversarial samples
  3. Ensure augmentation corpora are downloaded
  4. Build augmenter
  5. Extract features into memmaps (train / val split)
  6. Train classifier
  7. Export ONNX

The whole thing runs in a worker thread; progress is reported via the EventBus.
"""
from __future__ import annotations

import json
import logging
import hashlib
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from src.augment.augmenter import (
    apply_tablet_far_field_effect,
    build_augmenter,
    collect_background_noise_dirs,
    collect_rir_dirs,
)
from src.augment.downloader import ensure_corpora
from src.augment.downloader import ensure_openwakeword_feature_files
from src.config_schema import TrainRunConfig
from src.data.adversarial import build_adversarial_phrases
from src.data.dataset import (
    CombinedFeatureDataset,
    ExternalNegativeFeatureDataset,
    FEATURE_LABELING_VERSION,
    FeatureMemmapDataset,
    ShardManifest,
    allocate_memmap,
    build_features_from_wavs,
    build_features_from_wavs_parallel,
    estimate_window_count,
    save_manifest,
    write_wav,
)
from src.data.features import (
    CLASSIFIER_WINDOW_EMBEDDINGS,
    EMBEDDING_DIM,
    FeatureExtractor,
    float32_to_int16,
)
from src.settings import get_settings
from src.train.progress import bus
from src.train.trainer import CurveValidationSet, train as run_training
from src.tts.kokoro_generator import KokoroGenerator
from src.tts.piper_generator import PiperGenerator

logger = logging.getLogger(__name__)

FEATURE_RESUME_FILENAME = "features_resume.json"
FEATURE_RESUME_VERSION = 1
FEATURE_CHECKPOINT_EVERY_CLIPS = 500
CURVE_VALIDATION_FILENAME = "positive_curve_validation.json"
CURVE_VALIDATION_VERSION = 5


class RunState:
    """Shared mutable state for a single in-flight training run."""

    def __init__(self) -> None:
        self.run_id: str | None = None
        self.run_dir: Path | None = None
        self.config: TrainRunConfig | None = None
        self.status: str = "idle"  # idle | running | succeeded | failed | cancelled
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.onnx_path: Path | None = None
        self.error: str | None = None
        self.cancel_flag = threading.Event()
        self.thread: threading.Thread | None = None

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "onnx_path": str(self.onnx_path) if self.onnx_path else None,
            "error": self.error,
            "wake_word": self.config.wake_word if self.config else None,
        }


# Single in-process run state. The UI prevents starting a new run while one is active.
state = RunState()


def _make_run_dir(cfg: TrainRunConfig) -> tuple[str, Path]:
    settings = get_settings()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = cfg.run_name or f"{cfg.slug()}_{ts}"
    run_dir = settings.runs_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return name, run_dir


def _save_config(run_dir: Path, cfg: TrainRunConfig) -> None:
    (run_dir / "config.json").write_text(cfg.model_dump_json(indent=2))


def _write_model_package(
    *,
    run_id: str,
    run_dir: Path,
    final_onnx_path: Path,
    result_path: Path,
    package_path: Path,
) -> Path:
    """Create a portable zip with the model, exact config, and checkpoint."""

    package_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = package_path.with_suffix(package_path.suffix + ".tmp")
    tmp_path.unlink(missing_ok=True)

    checkpoint_path = run_dir / "best.pt"
    config_path = run_dir / "config.json"
    if not checkpoint_path.exists():
        raise RuntimeError(f"Cannot package model; missing checkpoint {checkpoint_path}")
    if not config_path.exists():
        raise RuntimeError(f"Cannot package model; missing config {config_path}")

    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(final_onnx_path, arcname=f"{run_id}.onnx")
        zf.write(config_path, arcname="training_config.json")
        zf.write(checkpoint_path, arcname="checkpoint.pt")
        if result_path.exists():
            zf.write(result_path, arcname="checkpoint_metadata.json")

    tmp_path.replace(package_path)
    return package_path


def _phrases_signature(
    phrases: list[str],
    n_per_phrase_per_voice: int,
    piper_voices: list,
) -> str:
    """Stable hash of everything that determines what WAVs get synthesized."""
    import hashlib
    import json

    blob = json.dumps(
        {
            "phrases": sorted(phrases),
            "n": n_per_phrase_per_voice,
            "piper": sorted(getattr(v, "voice_key", str(v)) for v in piper_voices),
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def _generate_samples(
    phrases: list[str],
    out_dir: Path,
    cfg: TrainRunConfig,
    n_per_phrase_per_voice: int,
    label: str,
) -> list[Path]:
    """Generate Piper samples. Returns wav paths.

    Skips generation entirely if a sentinel for this label exists AND its
    content matches a hash of the current phrases + voice selection. If the
    phrase list (or voices) changed since the last run, the sentinel is
    invalidated and synthesis re-runs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = out_dir / f".generated_{label}"
    existing_patterns = (f"piper_{label}_*.wav",)

    def existing_wavs() -> list[Path]:
        files: list[Path] = []
        for pattern in existing_patterns:
            files.extend(out_dir.glob(pattern))
        return sorted(files)

    def purge_existing_wavs(reason: str) -> None:
        old = existing_wavs()
        if old:
            bus.log(
                f"Removing {len(old):,} stale cached WAVs for label={label} ({reason})",
                level="warning",
            )
        for old_wav in old:
            try:
                old_wav.unlink()
            except FileNotFoundError:
                pass

    sig = _phrases_signature(
        phrases,
        n_per_phrase_per_voice,
        cfg.generation.piper_voices,
    )

    settings = get_settings()
    piper = PiperGenerator(use_cuda=settings.piper_use_cuda)
    piper_tasks = (
        piper.build_tasks(
            phrases=phrases,
            voice_selections=cfg.generation.piper_voices,
            n_per_phrase_per_voice=n_per_phrase_per_voice,
            cfg=cfg.generation,
            seed=hash((cfg.wake_word, label)) & 0xFFFFFFFF,
        )
        if cfg.generation.piper_voices
        else []
    )
    expected_total = len(piper_tasks)

    if sentinel.exists():
        cached = sentinel.read_text().strip()
        if cached == sig:
            existing = existing_wavs()
            if existing and len(existing) == expected_total:
                bus.log(
                    f"Reusing {len(existing)} cached WAVs for label={label} "
                    f"(signature match)"
                )
                return existing
            if existing:
                if len(existing) > expected_total:
                    stale = existing[expected_total:]
                    bus.log(
                        f"Trimming {len(stale):,} extra cached WAVs for label={label} "
                        f"(expected {expected_total:,}, found {len(existing):,})",
                        level="warning",
                    )
                    for old_wav in stale:
                        try:
                            old_wav.unlink()
                        except FileNotFoundError:
                            pass
                    existing = existing_wavs()
                    if len(existing) == expected_total:
                        return existing
                purge_existing_wavs(
                    f"expected {expected_total:,}, found {len(existing):,}"
                )
        else:
            bus.log(
                f"Phrases/voices changed since last run for label={label} "
                f"- regenerating (sig {cached[:8]}.. -> {sig[:8]}..)",
                level="warning",
            )
            purge_existing_wavs("signature changed")
    else:
        purge_existing_wavs("missing signature")

    written: list[Path] = []

    def write_sample_metadata(wav_path: Path, sample, engine: str) -> None:
        metadata_path = wav_path.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(
                {
                    "engine": engine,
                    "label": label,
                    "text": sample.text,
                    "voice": sample.voice,
                    "sample_rate": sample.sample_rate,
                    "metadata": sample.metadata,
                },
                indent=2,
                sort_keys=True,
            )
        )

    # Piper - parallel across a process pool.
    if piper_tasks:
        target_total = len(piper_tasks)
        workers = settings.resolved_generation_workers()
        bus.phase(
            f"generate:piper:{label}",
            detail=f"{len(phrases)} phrases, {target_total} synths, {workers} workers",
        )
        bus.log(
            f"Piper {label}: {target_total} synths across {workers} workers"
        )
        i = 0
        loop_start_t = time.monotonic()
        last_log_t = loop_start_t
        for wav_path in piper.iter_parallel_to_wavs(
            piper_tasks,
            workers=workers,
            out_dir=out_dir,
            label=label,
        ):
            written.append(wav_path)
            i += 1
            if i % 25 == 0:
                bus.progress(
                    f"generate:piper:{label}",
                    i / max(1, target_total),
                    detail=f"{i}/{target_total}",
                )
            # Heartbeat log every ~10s so the live-log keeps ticking on long
            # generation passes (Piper adversarial is often 100k+ synths).
            # Both timestamps come from time.monotonic() so the subtraction
            # is in the same time domain (state.started_at uses time.time()).
            now = time.monotonic()
            if (now - last_log_t) > 10.0:
                elapsed = max(1.0, now - loop_start_t)
                rate = i / elapsed
                bus.log(
                    f"Piper {label}: {i:,}/{target_total:,} synths "
                    f"({100.0 * i / max(1, target_total):.1f}%, ~{rate:.1f}/s avg)"
                )
                last_log_t = now
            if state.cancel_flag.is_set():
                # Breaking out of the iterator closes the pool via __exit__.
                break
        # Final progress emit so the UI shows 100% even when:
        #   - some synths failed silently (i < target_total)
        #   - the last batch did not hit the i % 25 == 0 boundary
        # If the run was cancelled, leave the bar at its last partial value.
        if not state.cancel_flag.is_set():
            skipped = target_total - i
            detail = f"{i}/{target_total}" + (f" ({skipped} skipped)" if skipped else "")
            bus.progress(f"generate:piper:{label}", 1.0, detail=detail)

    # Drop sentinel only if we successfully reached this point without cancel.
    # Content = phrase-list signature so a phrase change forces regeneration.
    if not state.cancel_flag.is_set() and written:
        if len(written) < expected_total:
            bus.log(
                f"Generated {len(written):,}/{expected_total:,} WAVs for label={label}; "
                "leaving cache unsatisfied so the next run can retry missing synths.",
                level="warning",
            )
            sentinel.unlink(missing_ok=True)
            return written
        sentinel.write_text(sig)

    return written


def _kokoro_signature(
    phrases: list[str],
    n_per_phrase_per_voice: int,
    voices: list[str],
    cfg: TrainRunConfig,
) -> str:
    import hashlib
    import json

    blob = json.dumps(
        {
            "phrases": sorted(phrases),
            "n": n_per_phrase_per_voice,
            "voices": sorted(voices),
            "speed_min": cfg.generation.kokoro_speed_min,
            "speed_max": cfg.generation.kokoro_speed_max,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def _generate_kokoro_samples(
    phrases: list[str],
    out_dir: Path,
    cfg: TrainRunConfig,
    n_per_phrase_per_voice: int,
    label: str,
    kokoro: KokoroGenerator | None = None,
) -> list[Path]:
    """Generate Kokoro samples with a separate cache sentinel from Piper."""
    if not cfg.generation.use_kokoro or not cfg.generation.kokoro_voices:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = out_dir / f".generated_kokoro_{label}"
    sig = _kokoro_signature(
        phrases,
        n_per_phrase_per_voice,
        cfg.generation.kokoro_voices,
        cfg,
    )
    target_total = (
        len(phrases)
        * len(cfg.generation.kokoro_voices)
        * n_per_phrase_per_voice
    )
    if sentinel.exists():
        cached = sentinel.read_text().strip()
        if cached == sig:
            existing = sorted(out_dir.glob(f"kokoro_{label}_*.wav"))
            if existing and len(existing) == target_total:
                bus.log(
                    f"Reusing {len(existing)} cached Kokoro WAVs for label={label} "
                    f"(signature match)"
                )
                return existing
            if existing:
                if len(existing) > target_total:
                    stale = existing[target_total:]
                    bus.log(
                        f"Trimming {len(stale):,} extra cached Kokoro WAVs for label={label} "
                        f"(expected {target_total:,}, found {len(existing):,})",
                        level="warning",
                    )
                    for old_wav in stale:
                        try:
                            old_wav.unlink()
                        except FileNotFoundError:
                            pass
                    existing = sorted(out_dir.glob(f"kokoro_{label}_*.wav"))
                    if len(existing) == target_total:
                        return existing
                bus.log(
                    f"Removing {len(existing):,} stale cached Kokoro WAVs for label={label} "
                    f"(expected {target_total:,})",
                    level="warning",
                )
                for old_wav in existing:
                    try:
                        old_wav.unlink()
                    except FileNotFoundError:
                        pass
        else:
            bus.log(
                f"Kokoro phrases/voices changed since last run for label={label} "
                f"- regenerating (sig {cached[:8]}.. -> {sig[:8]}..)",
                level="warning",
            )

    for old_wav in out_dir.glob(f"kokoro_{label}_*.wav"):
        try:
            old_wav.unlink()
        except FileNotFoundError:
            pass

    written: list[Path] = []
    if target_total <= 0:
        return written

    bus.phase(
        f"generate:kokoro:{label}",
        detail=(
            f"{len(phrases)} phrases, {len(cfg.generation.kokoro_voices)} voices, "
            f"{target_total} synths"
        ),
    )
    bus.log(f"Kokoro {label}: {target_total} synths")
    kokoro = kokoro or KokoroGenerator()
    i = 0
    loop_start_t = time.monotonic()
    last_log_t = loop_start_t
    import hashlib

    seed = int(hashlib.sha1(f"{cfg.wake_word}|{label}|kokoro".encode()).hexdigest()[:8], 16)
    for wav_path in kokoro.iter_samples_to_wavs(
        phrases=phrases,
        voice_keys=cfg.generation.kokoro_voices,
        n_per_phrase_per_voice=n_per_phrase_per_voice,
        cfg=cfg.generation,
        out_dir=out_dir,
        label=label,
        seed=seed,
    ):
        written.append(wav_path)
        i += 1
        if i % 10 == 0:
            bus.progress(
                f"generate:kokoro:{label}",
                i / max(1, target_total),
                detail=f"{i}/{target_total}",
            )
        now = time.monotonic()
        if (now - last_log_t) > 10.0:
            elapsed = max(1.0, now - loop_start_t)
            rate = i / elapsed
            bus.log(
                f"Kokoro {label}: {i:,}/{target_total:,} synths "
                f"({100.0 * i / max(1, target_total):.1f}%, ~{rate:.1f}/s avg)"
            )
            last_log_t = now
        if state.cancel_flag.is_set():
            break

    if not state.cancel_flag.is_set():
        skipped = target_total - i
        detail = f"{i}/{target_total}" + (f" ({skipped} skipped)" if skipped else "")
        bus.progress(f"generate:kokoro:{label}", 1.0, detail=detail)
        if i >= target_total and written:
            sentinel.write_text(sig)
        elif written:
            bus.log(
                f"Kokoro {label}: generated {i:,}/{target_total:,}; "
                "leaving cache unsatisfied so the next run can retry missing synths.",
                level="warning",
            )

    return written


def _resolve_positive_generation_counts(
    cfg: TrainRunConfig,
    phrases: list[str],
) -> tuple[int, int]:
    """Resolve Piper/Kokoro positive render counts from an optional total budget."""
    phrase_count = max(1, len(phrases))
    piper_voice_count = len(cfg.generation.piper_voices)
    kokoro_voice_count = (
        len(cfg.generation.kokoro_voices)
        if cfg.generation.use_kokoro
        else 0
    )
    budget = int(cfg.generation.positive_sample_budget or 0)
    if budget <= 0:
        return (
            cfg.generation.n_positive_per_phrase_per_voice,
            cfg.generation.n_kokoro_positive_per_phrase_per_voice,
        )

    if piper_voice_count <= 0 and kokoro_voice_count <= 0:
        return (0, 0)

    if piper_voice_count and kokoro_voice_count:
        piper_budget = round(budget * 0.9)
        kokoro_budget = max(0, budget - piper_budget)
    elif piper_voice_count:
        piper_budget = budget
        kokoro_budget = 0
    else:
        piper_budget = 0
        kokoro_budget = budget

    def ceil_div(value: int, denom: int) -> int:
        if value <= 0 or denom <= 0:
            return 0
        return max(1, (value + denom - 1) // denom)

    piper_n = ceil_div(piper_budget, phrase_count * piper_voice_count)
    kokoro_n = ceil_div(kokoro_budget, phrase_count * kokoro_voice_count)
    resolved_total = (
        phrase_count * piper_voice_count * piper_n
        + phrase_count * kokoro_voice_count * kokoro_n
    )
    bus.log(
        "Positive sample budget resolved: "
        f"target={budget:,}, actual={resolved_total:,}, "
        f"piper_per_phrase_per_voice={piper_n}, "
        f"kokoro_per_phrase_per_voice={kokoro_n}"
    )
    return (piper_n, kokoro_n)


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def _feature_input_signature(
    cfg: TrainRunConfig,
    positive_wavs: list[Path],
    negative_wavs: list[Path],
    cv_clips: list[Path],
    background_negative_clips: list[Path],
) -> str:
    """Hash the inputs that affect extracted features and train/val splits."""
    h = hashlib.sha256()
    payload = {
        "version": FEATURE_RESUME_VERSION,
        "feature_labeling_version": FEATURE_LABELING_VERSION,
        "wake_word": cfg.wake_word,
        "generation": cfg.generation.model_dump(mode="json"),
        "augmentation": cfg.augmentation.model_dump(mode="json"),
        "datasets": cfg.datasets.model_dump(mode="json"),
        "positive_temporal_windows": cfg.training.positive_temporal_windows,
        "positive_temporal_stride_embeddings": cfg.training.positive_temporal_stride_embeddings,
        "positive_context_seconds": cfg.training.positive_context_seconds,
        "split_seed": cfg.training.seed,
    }
    h.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    for group_name, paths in (
        ("positive", positive_wavs),
        ("negative", negative_wavs),
        ("common_voice", cv_clips),
        ("background_negatives", background_negative_clips),
    ):
        h.update(group_name.encode("utf-8"))
        h.update(str(len(paths)).encode("ascii"))
        for path in paths:
            h.update(str(path).encode("utf-8"))
            try:
                st = path.stat()
            except OSError:
                h.update(b":missing")
                continue
            h.update(f":{st.st_size}:{st.st_mtime_ns}".encode("ascii"))
    return h.hexdigest()


def _collect_background_negative_clips(bg_dirs: list[Path], limit: int, seed: int) -> list[Path]:
    """Use environmental corpora as standalone no-wake-word negative clips.

    MUSAN/FSD50K are already used as background overlays. Adding a bounded,
    deterministic subset as direct negatives gives the classifier explicit
    examples of household/media/noise audio where no wake word is present.
    """
    if limit <= 0:
        return []
    suffixes = {".wav", ".flac", ".ogg"}
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in bg_dirs:
        if not root or not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes and path not in seen:
                seen.add(path)
                candidates.append(path)
    if len(candidates) <= limit:
        return sorted(candidates)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(candidates), size=limit, replace=False)
    return sorted(candidates[int(i)] for i in indices)


def _curve_validation_signature(cfg: TrainRunConfig, val_pos: list[Path]) -> str:
    h = hashlib.sha256()
    payload = {
        "version": CURVE_VALIDATION_VERSION,
        "wake_word": cfg.wake_word,
        "seed": cfg.training.seed,
        "max_positive_clips": cfg.training.curve_validation_max_positive_clips,
        "tablet_curve_validation": cfg.training.use_tablet_curve_validation,
        "tablet_variants_per_clip": cfg.training.tablet_curve_validation_variants_per_clip,
        "tablet_far_field_probability": cfg.augmentation.tablet_far_field_probability,
        "rir_probability": cfg.augmentation.rir_probability,
        "background_noise_probability": cfg.augmentation.background_noise_probability,
    }
    h.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    for path in val_pos:
        h.update(str(path).encode("utf-8"))
        try:
            st = path.stat()
        except OSError:
            h.update(b":missing")
            continue
        h.update(f":{st.st_size}:{st.st_mtime_ns}".encode("ascii"))
    return h.hexdigest()


def _load_audio_16k(path: Path) -> np.ndarray:
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16_000:
        g = gcd(sr, 16_000)
        audio = resample_poly(audio, 16_000 // g, sr // g).astype(np.float32)
    else:
        audio = audio.astype(np.float32, copy=False)
    return audio


def _place_positive_for_curve_validation(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Place a phrase-only synthetic WAV into a streaming-style test clip."""
    lead_samples = int(rng.integers(int(0.4 * 16_000), int(1.2 * 16_000) + 1))
    tail_samples = int(1.2 * 16_000)
    total_samples = max(16_000 * 3, lead_samples + int(audio.size) + tail_samples)
    out = np.zeros(total_samples, dtype=np.float32)
    end = min(total_samples, lead_samples + int(audio.size))
    out[lead_samples:end] = audio[: max(0, end - lead_samples)]
    return out


def _build_curve_validation_features(
    *,
    run_dir: Path,
    val_pos: list[Path],
    cfg: TrainRunConfig,
    extractor: FeatureExtractor,
    rir_dirs: list[Path],
    bg_dirs: list[Path],
) -> CurveValidationSet | None:
    """Precompute full-audio sliding positive curves for export validation."""
    if not cfg.training.use_positive_curve_validation or not val_pos:
        return None

    limit = max(1, int(cfg.training.curve_validation_max_positive_clips))
    selected = list(val_pos)
    if len(selected) > limit:
        rng = np.random.default_rng(cfg.training.seed + 313)
        indices = sorted(int(i) for i in rng.choice(len(selected), size=limit, replace=False))
        selected = [selected[i] for i in indices]

    meta_path = run_dir / CURVE_VALIDATION_FILENAME
    features_path = run_dir / "positive_curve_features.bin"
    source_ids_path = run_dir / "positive_curve_source_ids.npy"
    tablet_features_path = run_dir / "tablet_positive_curve_features.bin"
    tablet_source_ids_path = run_dir / "tablet_positive_curve_source_ids.npy"
    signature = _curve_validation_signature(cfg, selected)
    if meta_path.exists() and features_path.exists() and source_ids_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            tablet_ready = (
                not cfg.training.use_tablet_curve_validation
                or (
                    tablet_features_path.exists()
                    and tablet_source_ids_path.exists()
                    and int(meta.get("tablet_n_windows", 0)) > 0
                    and int(meta.get("tablet_n_clips", 0)) > 0
                )
            )
            if (
                meta.get("version") == CURVE_VALIDATION_VERSION
                and meta.get("signature") == signature
                and int(meta.get("n_windows", 0)) > 0
                and int(meta.get("n_clips", 0)) > 0
                and tablet_ready
            ):
                bus.log(
                    "Reusing full-audio positive curve validation cache: "
                    f"{int(meta['n_clips']):,} clean clips, {int(meta['n_windows']):,} windows; "
                    f"{int(meta.get('tablet_n_clips', 0)):,} tablet clips, "
                    f"{int(meta.get('tablet_n_windows', 0)):,} windows"
                )
                return CurveValidationSet(
                    features_path=features_path,
                    source_ids_path=source_ids_path,
                    n_windows=int(meta["n_windows"]),
                    n_clips=int(meta["n_clips"]),
                    tablet_features_path=tablet_features_path if cfg.training.use_tablet_curve_validation else None,
                    tablet_source_ids_path=tablet_source_ids_path if cfg.training.use_tablet_curve_validation else None,
                    tablet_n_windows=int(meta.get("tablet_n_windows", 0)),
                    tablet_n_clips=int(meta.get("tablet_n_clips", 0)),
                )
        except Exception:
            bus.log("Ignoring unreadable positive curve validation cache", level="warning")

    windows: list[np.ndarray] = []
    source_ids: list[np.ndarray] = []
    tablet_windows: list[np.ndarray] = []
    tablet_source_ids: list[np.ndarray] = []
    rng = np.random.default_rng(cfg.training.seed + 719)
    # Use the per-clip training probabilities directly instead of stacking
    # 100% tablet + 90% RIR + 50% bg on every clip. Worst-case stacking made
    # the tablet curve gate effectively unreachable: most clips dropped to
    # ~0.3 calibrated peaks because every variant got all three degradations
    # at once. Real deployment is one channel at a time at the configured
    # rate; generate a few variants per clip so the metric is stable.
    tablet_aug_cfg = cfg.augmentation.model_copy(
        update={
            "use_tablet_far_field_augmentation": True,
        }
    )
    tablet_augmenter = (
        build_augmenter(tablet_aug_cfg, rir_dirs=rir_dirs, background_noise_dirs=bg_dirs)
        if cfg.training.use_tablet_curve_validation
        else None
    )
    tablet_variants_per_clip = max(1, int(cfg.training.tablet_curve_validation_variants_per_clip))
    for clip_id, path in enumerate(selected):
        try:
            audio = _load_audio_16k(path)
            audio = _place_positive_for_curve_validation(audio, rng)
            clip_windows = extractor.classifier_inputs(float32_to_int16(audio))
        except Exception as exc:
            bus.log(f"Skipping curve validation clip {path.name}: {exc}", level="warning")
            continue
        if clip_windows.shape[0] == 0:
            continue
        windows.append(clip_windows.astype(np.float32, copy=False))
        source_ids.append(np.full(clip_windows.shape[0], clip_id, dtype=np.int32))
        if tablet_augmenter is not None:
            for variant_idx in range(tablet_variants_per_clip):
                try:
                    tablet_audio = tablet_augmenter(samples=audio.astype(np.float32), sample_rate=16_000)
                    tablet_audio = apply_tablet_far_field_effect(tablet_audio, 16_000, tablet_aug_cfg)
                    np.clip(tablet_audio, -1.0, 1.0, out=tablet_audio)
                    tw = extractor.classifier_inputs(float32_to_int16(tablet_audio))
                except Exception as exc:
                    bus.log(
                        f"Skipping tablet curve validation variant {path.name}: {exc}",
                        level="warning",
                    )
                    continue
                if tw.shape[0] == 0:
                    continue
                tablet_windows.append(tw.astype(np.float32, copy=False))
                tablet_source_ids.append(
                    np.full(
                        tw.shape[0],
                        clip_id * tablet_variants_per_clip + variant_idx,
                        dtype=np.int32,
                    )
                )

    if not windows:
        bus.log("No positive curve validation windows were produced", level="warning")
        return None

    all_windows = np.concatenate(windows, axis=0)
    all_sources = np.concatenate(source_ids, axis=0)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(
        features_path,
        dtype=np.float32,
        mode="w+",
        shape=all_windows.shape,
    )
    mm[:] = all_windows
    mm.flush()
    del mm
    np.save(source_ids_path, all_sources)
    tablet_n_windows = 0
    tablet_n_clips = 0
    if cfg.training.use_tablet_curve_validation and tablet_windows:
        all_tablet_windows = np.concatenate(tablet_windows, axis=0)
        all_tablet_sources = np.concatenate(tablet_source_ids, axis=0)
        tmm = np.memmap(
            tablet_features_path,
            dtype=np.float32,
            mode="w+",
            shape=all_tablet_windows.shape,
        )
        tmm[:] = all_tablet_windows
        tmm.flush()
        del tmm
        np.save(tablet_source_ids_path, all_tablet_sources)
        tablet_n_windows = int(all_tablet_windows.shape[0])
        tablet_n_clips = int(np.unique(all_tablet_sources).size)
    elif cfg.training.use_tablet_curve_validation:
        bus.log("No tablet/off-axis positive curve validation windows were produced", level="warning")
    meta = {
        "version": CURVE_VALIDATION_VERSION,
        "signature": signature,
        "n_windows": int(all_windows.shape[0]),
        "n_clips": int(len(windows)),
        "tablet_n_windows": tablet_n_windows,
        "tablet_n_clips": tablet_n_clips,
        "clips": [str(p) for p in selected],
    }
    _atomic_write_json(meta_path, meta)
    bus.log(
        "Built full-audio positive curve validation set: "
        f"{len(windows):,} clean clips, {all_windows.shape[0]:,} windows; "
        f"{tablet_n_clips:,} tablet clips, {tablet_n_windows:,} windows"
    )
    return CurveValidationSet(
        features_path=features_path,
        source_ids_path=source_ids_path,
        n_windows=int(all_windows.shape[0]),
        n_clips=int(len(windows)),
        tablet_features_path=tablet_features_path if tablet_n_windows else None,
        tablet_source_ids_path=tablet_source_ids_path if tablet_n_windows else None,
        tablet_n_windows=tablet_n_windows,
        tablet_n_clips=tablet_n_clips,
    )


def _load_completed_feature_cache(
    run_dir: Path,
    signature: str,
) -> tuple[FeatureMemmapDataset, FeatureMemmapDataset, dict[str, ShardManifest]] | None:
    resume_path = run_dir / FEATURE_RESUME_FILENAME
    manifest_path = run_dir / "shards.json"
    if not resume_path.exists() or not manifest_path.exists():
        return None
    try:
        resume = json.loads(resume_path.read_text())
        manifest_data = json.loads(manifest_path.read_text())
    except Exception:
        return None
    if (
        resume.get("version") != FEATURE_RESUME_VERSION
        or resume.get("signature") != signature
        or not resume.get("complete")
    ):
        return None

    manifests: dict[str, ShardManifest] = {}
    for split in ("train", "val"):
        entry = manifest_data.get(split)
        if not entry:
            return None
        features_path = run_dir / entry["features"]
        labels_path = run_dir / entry["labels"]
        source_ids_path = (
            run_dir / entry["source_ids"]
            if entry.get("source_ids")
            else None
        )
        if not features_path.exists() or not labels_path.exists():
            return None
        if source_ids_path is not None and not source_ids_path.exists():
            return None
        manifests[split] = ShardManifest(
            features_path=features_path,
            labels_path=labels_path,
            source_ids_path=source_ids_path,
            n_windows=int(entry["n_windows"]),
            label_counts={int(k): int(v) for k, v in entry["label_counts"].items()},
        )

    bus.log("Reusing completed feature cache for matching session/config")
    return (
        FeatureMemmapDataset(
            manifests["train"].features_path,
            manifests["train"].labels_path,
            manifests["train"].source_ids_path,
        ),
        FeatureMemmapDataset(
            manifests["val"].features_path,
            manifests["val"].labels_path,
            manifests["val"].source_ids_path,
        ),
        manifests,
    )


def _feature_partial_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "train_features": run_dir / "train_features.bin",
        "val_features": run_dir / "val_features.bin",
        "train_labels": run_dir / "train_labels.partial.bin",
        "val_labels": run_dir / "val_labels.partial.bin",
        "train_source_ids": run_dir / "train_source_ids.partial.bin",
        "val_source_ids": run_dir / "val_source_ids.partial.bin",
    }


def _feature_partial_files_match(
    paths: dict[str, Path],
    train_capacity: int,
    val_capacity: int,
) -> bool:
    bytes_per_window = CLASSIFIER_WINDOW_EMBEDDINGS * EMBEDDING_DIM * 4
    expected = {
        "train_features": train_capacity * bytes_per_window,
        "val_features": val_capacity * bytes_per_window,
        "train_labels": train_capacity,
        "val_labels": val_capacity,
        "train_source_ids": train_capacity * 4,
        "val_source_ids": val_capacity * 4,
    }
    for key, size in expected.items():
        path = paths[key]
        if not path.exists() or path.stat().st_size != size:
            return False
    return True


def _initial_feature_resume_state(
    signature: str,
    train_capacity: int,
    val_capacity: int,
) -> dict:
    return {
        "version": FEATURE_RESUME_VERSION,
        "signature": signature,
        "complete": False,
        "created_at": time.time(),
        "updated_at": time.time(),
        "capacities": {"train": train_capacity, "val": val_capacity},
        "split_cursors": {"train": 0, "val": 0},
        "passes": {},
    }


def _load_partial_feature_resume(
    run_dir: Path,
    signature: str,
    train_capacity: int,
    val_capacity: int,
) -> tuple[dict, bool]:
    resume_path = run_dir / FEATURE_RESUME_FILENAME
    partial_paths = _feature_partial_paths(run_dir)
    if not resume_path.exists():
        return _initial_feature_resume_state(signature, train_capacity, val_capacity), False
    try:
        resume = json.loads(resume_path.read_text())
    except Exception:
        bus.log("Ignoring unreadable feature resume checkpoint", level="warning")
        return _initial_feature_resume_state(signature, train_capacity, val_capacity), False
    if (
        resume.get("version") != FEATURE_RESUME_VERSION
        or resume.get("signature") != signature
        or resume.get("complete")
        or resume.get("capacities") != {"train": train_capacity, "val": val_capacity}
        or not _feature_partial_files_match(partial_paths, train_capacity, val_capacity)
    ):
        bus.log("Feature resume checkpoint does not match this run; rebuilding features")
        return _initial_feature_resume_state(signature, train_capacity, val_capacity), False
    bus.log("Resuming feature extraction from saved checkpoint")
    return resume, True


def _build_features(
    positive_wavs: list[Path],
    negative_wavs: list[Path],
    common_voice_dir: Path | None,
    cfg: TrainRunConfig,
    run_dir: Path,
) -> tuple[FeatureMemmapDataset, FeatureMemmapDataset, dict[str, ShardManifest], CurveValidationSet | None]:
    """Augment + extract features into train/val memmaps."""
    extractor = FeatureExtractor()

    rir_dirs = collect_rir_dirs(
        use_mit_rirs=cfg.datasets.use_mit_rirs,
        use_but_reverbdb=cfg.datasets.use_but_reverbdb,
    )
    bg_dirs = collect_background_noise_dirs(
        use_musan_noise=cfg.datasets.use_musan_noise,
        use_musan_music=cfg.datasets.use_musan_music,
        use_fsd50k=cfg.datasets.use_fsd50k,
    )
    augmenter = build_augmenter(
        cfg.augmentation,
        rir_dirs=rir_dirs,
        background_noise_dirs=bg_dirs,
    )

    # Add Common Voice clips as additional negatives.
    cv_clips: list[Path] = []
    if common_voice_dir and cfg.datasets.use_common_voice_negatives:
        cv_clips = sorted(common_voice_dir.rglob("*.wav"))[: cfg.datasets.common_voice_subset]

    background_negative_clips: list[Path] = []
    if cfg.datasets.use_background_corpus_negatives:
        background_negative_clips = _collect_background_negative_clips(
            bg_dirs,
            int(cfg.datasets.background_corpus_negative_subset),
            seed=cfg.training.seed,
        )
        if background_negative_clips:
            bus.log(
                f"Added background-corpus standalone negatives: "
                f"{len(background_negative_clips):,} clips"
            )

    all_negatives = negative_wavs + cv_clips + background_negative_clips

    # 90/10 split; track at the source-clip level so windows from the same clip
    # don't leak across train/val.
    rng = np.random.default_rng(cfg.training.seed)
    pos_idx = rng.permutation(len(positive_wavs))
    neg_idx = rng.permutation(len(all_negatives))

    pos_split = max(1, int(0.9 * len(positive_wavs)))
    neg_split = max(1, int(0.9 * len(all_negatives)))

    train_pos = [positive_wavs[i] for i in pos_idx[:pos_split]]
    val_pos = [positive_wavs[i] for i in pos_idx[pos_split:]]
    train_neg = [all_negatives[i] for i in neg_idx[:neg_split]]
    val_neg = [all_negatives[i] for i in neg_idx[neg_split:]]
    curve_val = _build_curve_validation_features(
        run_dir=run_dir,
        val_pos=val_pos,
        cfg=cfg,
        extractor=extractor,
        rir_dirs=rir_dirs,
        bg_dirs=bg_dirs,
    )

    aug_per = cfg.augmentation.augmentations_per_clip
    feature_signature = _feature_input_signature(
        cfg,
        positive_wavs,
        negative_wavs,
        cv_clips,
        background_negative_clips,
    )
    cached = _load_completed_feature_cache(run_dir, feature_signature)
    if cached is not None:
        train_ds, val_ds, manifests = cached
        return train_ds, val_ds, manifests, curve_val

    pos_windows_per_aug = max(1, int(cfg.training.positive_temporal_windows)) * 2
    train_capacity = (
        estimate_window_count(len(train_pos), aug_per, pos_windows_per_aug)
        + estimate_window_count(len(train_neg), aug_per)
    ) or 1024
    val_capacity = (
        estimate_window_count(len(val_pos), aug_per, pos_windows_per_aug)
        + estimate_window_count(len(val_neg), aug_per)
    ) or 256

    bus.log(f"Allocating train memmap up to {train_capacity:,} windows; val {val_capacity:,}")

    train_features_path = run_dir / "train_features.bin"
    val_features_path = run_dir / "val_features.bin"
    train_labels_path = run_dir / "train_labels.npy"
    val_labels_path = run_dir / "val_labels.npy"
    train_source_ids_path = run_dir / "train_source_ids.npy"
    val_source_ids_path = run_dir / "val_source_ids.npy"
    partial_paths = _feature_partial_paths(run_dir)
    resume_path = run_dir / FEATURE_RESUME_FILENAME
    resume_state, is_resuming = _load_partial_feature_resume(
        run_dir,
        feature_signature,
        train_capacity,
        val_capacity,
    )
    memmap_mode = "r+" if is_resuming else "w+"

    train_arr, _ = allocate_memmap(train_features_path, train_capacity, mode=memmap_mode)
    val_arr, _ = allocate_memmap(val_features_path, val_capacity, mode=memmap_mode)
    train_labels = np.memmap(
        partial_paths["train_labels"],
        dtype=np.uint8,
        mode=memmap_mode,
        shape=(train_capacity,),
    )
    val_labels = np.memmap(
        partial_paths["val_labels"],
        dtype=np.uint8,
        mode=memmap_mode,
        shape=(val_capacity,),
    )
    train_source_ids = np.memmap(
        partial_paths["train_source_ids"],
        dtype=np.int32,
        mode=memmap_mode,
        shape=(train_capacity,),
    )
    val_source_ids = np.memmap(
        partial_paths["val_source_ids"],
        dtype=np.int32,
        mode=memmap_mode,
        shape=(val_capacity,),
    )
    if not is_resuming:
        train_labels[:] = 0
        val_labels[:] = 0
        train_source_ids[:] = -1
        val_source_ids[:] = -1
        train_labels.flush()
        val_labels.flush()
        train_source_ids.flush()
        val_source_ids.flush()

    # Split the global progress bar across 4 sub-passes weighted by clip count.
    total_clips = max(1, len(train_pos) + len(train_neg) + len(val_pos) + len(val_neg))
    bus.log(
        f"features:extract starting - {total_clips:,} source clips "
        f"x {aug_per} augs each"
    )

    # Feature extraction uses a process pool. Each worker does augmentation
    # + mel + embedding + window filter end-to-end; main writes the memmap.
    workers = get_settings().resolved_feature_workers()
    bus.log(
        f"features:extract using {workers} worker processes "
        f"(each with its own CUDA session)"
    )

    split_cursors = {
        "train": int(resume_state.get("split_cursors", {}).get("train", 0)),
        "val": int(resume_state.get("split_cursors", {}).get("val", 0)),
    }

    def save_feature_checkpoint(
        pass_name: str,
        split: str,
        label: int,
        total: int,
        cursor: int,
        completed_indices: set[int],
        complete: bool = False,
    ) -> None:
        train_arr.flush()
        val_arr.flush()
        train_labels.flush()
        val_labels.flush()
        train_source_ids.flush()
        val_source_ids.flush()
        resume_state["updated_at"] = time.time()
        resume_state["complete"] = False
        resume_state.setdefault("split_cursors", {})[split] = int(cursor)
        resume_state.setdefault("passes", {})[pass_name] = {
            "split": split,
            "label": label,
            "total": total,
            "cursor": int(cursor),
            "completed_count": len(completed_indices),
            "completed_indices": sorted(int(i) for i in completed_indices),
            "complete": complete,
        }
        _atomic_write_json(resume_path, resume_state)

    def run_feature_pass(
        pass_name: str,
        split: str,
        label: int,
        wavs: list[Path],
        out_features: np.memmap,
        out_labels: np.memmap,
        out_source_ids: np.memmap,
        base: float,
        span: float,
        source_id_base: int,
    ) -> int:
        entry = resume_state.get("passes", {}).get(pass_name, {})
        if (
            entry.get("complete")
            and entry.get("total") == len(wavs)
            and entry.get("split") == split
            and entry.get("label") == label
        ):
            cursor = int(entry.get("cursor", split_cursors[split]))
            split_cursors[split] = cursor
            bus.log(
                f"Reusing completed feature pass {pass_name}: "
                f"{len(wavs):,}/{len(wavs):,} clips"
            )
            bus.progress(
                "features:extract",
                base + span,
                detail=f"{len(wavs)}/{len(wavs)} clips, {cursor:,} windows",
            )
            return cursor

        completed_indices = {
            int(i)
            for i in entry.get("completed_indices", [])
            if 0 <= int(i) < len(wavs)
        }
        cursor = int(entry.get("cursor", split_cursors[split]))
        if completed_indices:
            bus.log(
                f"Resuming feature pass {pass_name}: "
                f"{len(completed_indices):,}/{len(wavs):,} clips already written"
            )
            bus.progress(
                "features:extract",
                base + span * (len(completed_indices) / max(1, len(wavs))),
                detail=f"{len(completed_indices)}/{len(wavs)} clips, {cursor:,} windows",
            )

        bus.phase("features:extract", detail=f"{pass_name.replace('_', ' ')} ({len(wavs)} clips)")

        def checkpoint(new_cursor: int, done: set[int], _done_count: int) -> None:
            save_feature_checkpoint(
                pass_name,
                split,
                label,
                len(wavs),
                new_cursor,
                done,
            )

        new_cursor = build_features_from_wavs_parallel(
            wavs, label, extractor, cfg.augmentation, rir_dirs, bg_dirs, aug_per,
            out_features, out_labels, out_source_ids, cursor,
            workers=workers,
            cancel_flag=state.cancel_flag,
            progress_label="features:extract",
            progress_fraction_base=base,
            progress_fraction_span=span,
            source_id_base=source_id_base,
            completed_clip_indices=completed_indices,
            checkpoint_callback=checkpoint,
            checkpoint_every=FEATURE_CHECKPOINT_EVERY_CLIPS,
            positive_temporal_windows=cfg.training.positive_temporal_windows,
            positive_temporal_stride_embeddings=cfg.training.positive_temporal_stride_embeddings,
            positive_context_seconds=cfg.training.positive_context_seconds,
        )
        split_cursors[split] = new_cursor
        save_feature_checkpoint(
            pass_name,
            split,
            label,
            len(wavs),
            new_cursor,
            completed_indices,
            complete=False,
        )
        if state.cancel_flag.is_set():
            raise RuntimeError("cancelled")
        if len(completed_indices) < len(wavs):
            raise RuntimeError(
                f"Feature capacity exhausted during {pass_name}: "
                f"{len(completed_indices):,}/{len(wavs):,} clips completed. "
                "Increase the feature capacity estimate before retrying."
            )
        save_feature_checkpoint(
            pass_name,
            split,
            label,
            len(wavs),
            new_cursor,
            completed_indices,
            complete=True,
        )
        return new_cursor

    base = 0.0
    span = len(train_pos) / total_clips
    train_cursor = run_feature_pass(
        "train_positives",
        "train",
        1,
        train_pos,
        train_arr,
        train_labels,
        train_source_ids,
        base,
        span,
        source_id_base=0,
    )

    base += span
    span = len(train_neg) / total_clips
    train_cursor = run_feature_pass(
        "train_negatives",
        "train",
        0,
        train_neg,
        train_arr,
        train_labels,
        train_source_ids,
        base,
        span,
        source_id_base=len(train_pos),
    )

    base += span
    span = len(val_pos) / total_clips
    val_cursor = run_feature_pass(
        "val_positives",
        "val",
        1,
        val_pos,
        val_arr,
        val_labels,
        val_source_ids,
        base,
        span,
        source_id_base=0,
    )

    base += span
    span = len(val_neg) / total_clips
    val_cursor = run_feature_pass(
        "val_negatives",
        "val",
        0,
        val_neg,
        val_arr,
        val_labels,
        val_source_ids,
        base,
        span,
        source_id_base=len(val_pos),
    )
    bus.progress("features:extract", 1.0,
                 detail=f"done - train {train_cursor:,} windows, val {val_cursor:,} windows")

    # Truncate to actual cursor and re-create memmaps + labels at exact size.
    train_arr.flush()
    val_arr.flush()
    del train_arr, val_arr

    # Resize via memmap "view" + label array
    np.save(train_labels_path, train_labels[:train_cursor])
    np.save(val_labels_path, val_labels[:val_cursor])
    np.save(train_source_ids_path, train_source_ids[:train_cursor])
    np.save(val_source_ids_path, val_source_ids[:val_cursor])
    # Truncate the binary files so the dataset memmap reads the right shape.
    bytes_per_window = CLASSIFIER_WINDOW_EMBEDDINGS * EMBEDDING_DIM * 4
    with open(train_features_path, "r+b") as f:
        f.truncate(train_cursor * bytes_per_window)
    with open(val_features_path, "r+b") as f:
        f.truncate(val_cursor * bytes_per_window)

    train_ds = FeatureMemmapDataset(train_features_path, train_labels_path, train_source_ids_path)
    val_ds = FeatureMemmapDataset(val_features_path, val_labels_path, val_source_ids_path)

    label_counts_train = {
        int(k): int(v)
        for k, v in zip(*np.unique(train_labels[:train_cursor], return_counts=True))
    }
    label_counts_val = {
        int(k): int(v)
        for k, v in zip(*np.unique(val_labels[:val_cursor], return_counts=True))
    }

    manifests = {
        "train": ShardManifest(
            features_path=train_features_path,
            labels_path=train_labels_path,
            source_ids_path=train_source_ids_path,
            n_windows=train_cursor,
            label_counts=label_counts_train,
        ),
        "val": ShardManifest(
            features_path=val_features_path,
            labels_path=val_labels_path,
            source_ids_path=val_source_ids_path,
            n_windows=val_cursor,
            label_counts=label_counts_val,
        ),
    }
    save_manifest(run_dir, manifests)
    resume_state["complete"] = True
    resume_state["completed_at"] = time.time()
    resume_state["updated_at"] = time.time()
    resume_state["split_cursors"] = {"train": int(train_cursor), "val": int(val_cursor)}
    _atomic_write_json(resume_path, resume_state)
    return train_ds, val_ds, manifests, curve_val


def _run(cfg: TrainRunConfig) -> None:
    settings = get_settings()
    state.config = cfg
    state.status = "running"
    state.started_at = time.time()
    state.finished_at = None
    state.onnx_path = None
    state.error = None
    state.cancel_flag.clear()

    try:
        run_id, run_dir = _make_run_dir(cfg)
        state.run_id = run_id
        state.run_dir = run_dir
        bus.start_run(run_id, run_dir)
        bus.publish("run_started", run_id=run_id, run_dir=str(run_dir), wake_word=cfg.wake_word)
        _save_config(run_dir, cfg)
        for stale_name in ("wakeword.onnx", "best.pt", "best_candidate.pt", "result.json"):
            stale_path = run_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()

        # 1. Download/verify corpora and external feature banks first. This
        # fails fast on missing credentials/disk and avoids spending hours on
        # TTS before discovering a dataset problem.
        bus.phase("download:corpora")
        corpora = ensure_corpora(
            use_mit_rirs=cfg.datasets.use_mit_rirs,
            use_but_reverbdb=cfg.datasets.use_but_reverbdb,
            use_musan=cfg.datasets.use_musan_noise or cfg.datasets.use_musan_music,
            use_fsd50k=cfg.datasets.use_fsd50k,
            use_common_voice=cfg.datasets.use_common_voice_negatives,
            common_voice_subset=cfg.datasets.common_voice_subset,
            progress=lambda name, frac: bus.progress(f"download:{name}", frac),
            cancel_flag=state.cancel_flag,
        )
        bus.log(f"Corpora ready: {list(corpora.keys())}")

        oww_feature_paths: dict[str, Path] = {}
        if (
            cfg.datasets.use_openwakeword_negative_features
            or cfg.datasets.use_openwakeword_validation_features
        ):
            bus.phase("download:openwakeword_features")
            oww_feature_paths = ensure_openwakeword_feature_files(
                use_training=cfg.datasets.use_openwakeword_negative_features,
                use_validation=cfg.datasets.use_openwakeword_validation_features,
                progress=lambda name, frac: bus.progress(f"download:{name}", frac),
            )
            bus.log(
                "openWakeWord feature banks ready: "
                + ", ".join(f"{k}={v.name}" for k, v in oww_feature_paths.items())
            )

        if state.cancel_flag.is_set():
            raise RuntimeError("cancelled")

        # 2. Positive phrases
        positive_phrases = cfg.generation.positive_phrases or [cfg.wake_word]
        piper_positive_count, kokoro_positive_count = _resolve_positive_generation_counts(
            cfg,
            positive_phrases,
        )
        kokoro = (
            KokoroGenerator()
            if cfg.generation.use_kokoro and cfg.generation.kokoro_voices
            else None
        )
        bus.phase("generate:positive", detail=f"{len(positive_phrases)} phrases")
        positive_wavs = _generate_samples(
            positive_phrases,
            run_dir / "wavs" / "positive",
            cfg,
            piper_positive_count,
            label="pos",
        )
        positive_wavs.extend(
            _generate_kokoro_samples(
                positive_phrases,
                run_dir / "wavs" / "positive",
                cfg,
                kokoro_positive_count,
                label="pos",
                kokoro=kokoro,
            )
        )
        bus.log(f"Positive WAVs generated: {len(positive_wavs)}")

        if state.cancel_flag.is_set():
            raise RuntimeError("cancelled")

        # 3a. Hard negatives (user-supplied phrases the model must NOT trigger on).
        # Synthesized with the same emphasis as positives so the model strongly
        # learns to reject them.
        negative_wavs: list[Path] = []
        positive_phrase_lowers = {p.lower().strip() for p in positive_phrases}
        hard_negative_phrases = [
            p
            for p in (cfg.generation.negative_phrases or [])
            if p.lower().strip() not in positive_phrase_lowers
        ]
        skipped_hard_negatives = len(cfg.generation.negative_phrases or []) - len(hard_negative_phrases)
        if skipped_hard_negatives:
            bus.log(
                f"Skipped {skipped_hard_negatives} hard negative phrase(s) "
                "because they exactly match configured positive phrases.",
                level="warning",
            )
        if hard_negative_phrases:
            bus.phase(
                "generate:hard_negatives",
                detail=f"{len(hard_negative_phrases)} phrases",
            )
            negative_wavs.extend(
                _generate_samples(
                    hard_negative_phrases,
                    run_dir / "wavs" / "negative",
                    cfg,
                    cfg.generation.n_negative_per_phrase_per_voice,
                    label="hard_neg",
                )
            )
            if cfg.generation.use_kokoro_for_negatives:
                negative_wavs.extend(
                    _generate_kokoro_samples(
                        hard_negative_phrases,
                        run_dir / "wavs" / "negative",
                        cfg,
                        cfg.generation.n_kokoro_negative_per_phrase_per_voice,
                        label="hard_neg",
                        kokoro=kokoro,
                    )
                )
            bus.log(f"Hard-negative WAVs generated: {len(negative_wavs)}")

        if state.cancel_flag.is_set():
            raise RuntimeError("cancelled")

        # 3b. Auto-generated adversarial phrases (phonetic neighbors + generic
        # conversational pool). These broaden the negative distribution.
        adv_phrases = build_adversarial_phrases(
            cfg.wake_word,
            cfg.generation.n_adversarial_phrases,
            seed=cfg.training.seed,
            forbidden_phrases=positive_phrases,
        )
        bus.phase("generate:adversarial", detail=f"{len(adv_phrases)} phrases")
        negative_wavs.extend(
            _generate_samples(
                adv_phrases,
                run_dir / "wavs" / "negative",
                cfg,
                cfg.generation.n_adversarial_per_phrase_per_voice,
                label="adv",
            )
        )
        bus.log(f"Total negative WAVs (hard + adversarial): {len(negative_wavs)}")

        if state.cancel_flag.is_set():
            raise RuntimeError("cancelled")

        # 4 + 5. Features
        bus.phase("features:build")
        train_ds, val_ds, _manifests, curve_val = _build_features(
            positive_wavs=positive_wavs,
            negative_wavs=negative_wavs,
            common_voice_dir=corpora.get("common_voice"),
            cfg=cfg,
            run_dir=run_dir,
        )
        if "acav100m" in oww_feature_paths:
            external_train = ExternalNegativeFeatureDataset(oww_feature_paths["acav100m"])
            train_ds = CombinedFeatureDataset([train_ds, external_train])
            bus.log(
                f"Added ACAV100M generic negatives to training: "
                f"{len(external_train):,} windows"
            )
        if "validation" in oww_feature_paths:
            external_val = ExternalNegativeFeatureDataset(oww_feature_paths["validation"])
            val_ds = CombinedFeatureDataset([val_ds, external_val])
            bus.log(
                f"Added official openWakeWord validation negatives: "
                f"{len(external_val):,} windows"
            )

        if state.cancel_flag.is_set():
            raise RuntimeError("cancelled")

        # 6. Train
        bus.phase("train")
        result = run_training(
            cfg.training,
            train_ds=train_ds,
            val_ds=val_ds,
            out_dir=run_dir,
            workers=settings.resolved_dataloader_workers(),
            curve_val=curve_val,
            cancel_flag=state.cancel_flag,
        )

        # 7. Publish to models dir
        final_path = settings.models_dir / f"{run_id}.onnx"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        from shutil import copy2

        copy2(result.onnx_path, final_path)
        state.onnx_path = final_path

        result_path = run_dir / "result.json"
        package_path = settings.models_dir / f"{run_id}.zip"
        result_path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "wake_word": cfg.wake_word,
                    "best_val_loss": result.best_val_loss,
                    "best_val_recall_at_p95": result.best_val_recall,
                    "best_val_recall_at_target_fp": result.best_val_recall_at_target_fp,
                    "best_val_fp_per_hour": result.best_val_fp_per_hour,
                    "best_val_recall_at_0_5": result.best_val_recall_at_0_5,
                    "best_val_fp_per_hour_at_0_5": result.best_val_fp_per_hour_at_0_5,
                    "best_positive_median_score": result.best_positive_median_score,
                    "best_positive_p10_score": result.best_positive_p10_score,
                    "best_curve_recall": result.best_curve_recall,
                    "best_curve_median_peak": result.best_curve_median_peak,
                    "best_curve_p10_peak": result.best_curve_p10_peak,
                    "best_curve_median_frames": result.best_curve_median_frames,
                    "best_curve_median_span_ms": result.best_curve_median_span_ms,
                    "best_curve_confirmation_rate": result.best_curve_confirmation_rate,
                    "best_tablet_curve_recall": result.best_tablet_curve_recall,
                    "best_tablet_curve_median_peak": result.best_tablet_curve_median_peak,
                    "best_tablet_curve_p10_peak": result.best_tablet_curve_p10_peak,
                    "best_tablet_curve_median_frames": result.best_tablet_curve_median_frames,
                    "best_tablet_curve_median_span_ms": result.best_tablet_curve_median_span_ms,
                    "best_tablet_curve_confirmation_rate": result.best_tablet_curve_confirmation_rate,
                    "calibration_threshold_raw": result.best_threshold,
                    "recommended_runtime_threshold": 0.5,
                    "recommended_threshold": 0.5,
                    "best_step": result.best_step,
                    "onnx_path": str(final_path),
                    "package_path": str(package_path),
                    "history": result.history,
                },
                indent=2,
            )
        )
        _write_model_package(
            run_id=run_id,
            run_dir=run_dir,
            final_onnx_path=final_path,
            result_path=result_path,
            package_path=package_path,
        )
        bus.log(f"Packaged model artifact -> {package_path}")

        state.status = "succeeded"
        bus.complete(run_id=run_id, onnx_path=str(final_path), package_path=str(package_path))
    except Exception as exc:
        if state.cancel_flag.is_set():
            state.status = "cancelled"
            bus.publish("cancelled", run_id=state.run_id)
        else:
            state.status = "failed"
            state.error = str(exc)
            logger.exception("Training pipeline failed")
            bus.error(str(exc))
    finally:
        state.finished_at = time.time()
        bus.finish_run()


def start_run(cfg: TrainRunConfig) -> bool:
    """Kick off a run in a background thread. Returns False if one is already running."""
    if state.status == "running":
        return False
    state.thread = threading.Thread(target=_run, args=(cfg,), name="oww-trainer", daemon=True)
    state.thread.start()
    return True


def cancel_run() -> bool:
    if state.status != "running":
        return False
    state.cancel_flag.set()
    bus.log("Cancellation requested", level="warning")
    return True
