"""End-to-end training run.

Phases:
  0. Prepare run directory + persist config
  1. Generate positive samples (Piper + optional ElevenLabs)
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
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from src.augment.augmenter import (
    build_augmenter,
    collect_background_noise_dirs,
    collect_rir_dirs,
)
from src.augment.downloader import ensure_corpora
from src.config_schema import TrainRunConfig
from src.data.adversarial import build_adversarial_phrases
from src.data.dataset import (
    FeatureMemmapDataset,
    ShardManifest,
    allocate_memmap,
    build_features_from_wavs,
    estimate_window_count,
    save_manifest,
    write_wav,
)
from src.data.features import (
    CLASSIFIER_WINDOW_EMBEDDINGS,
    EMBEDDING_DIM,
    FeatureExtractor,
)
from src.settings import get_settings
from src.train.progress import bus
from src.train.trainer import train as run_training
from src.tts.elevenlabs_generator import ElevenLabsGenerator
from src.tts.piper_generator import PiperGenerator

logger = logging.getLogger(__name__)


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


def _generate_samples(
    phrases: list[str],
    out_dir: Path,
    cfg: TrainRunConfig,
    n_per_phrase_per_voice: int,
    label: str,
) -> list[Path]:
    """Generate Piper + (optionally) ElevenLabs samples. Returns wav paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    settings = get_settings()
    piper = PiperGenerator(use_cuda=False)

    # Piper - parallel across a process pool.
    if cfg.generation.piper_voices:
        tasks = piper.build_tasks(
            phrases=phrases,
            voice_selections=cfg.generation.piper_voices,
            n_per_phrase_per_voice=n_per_phrase_per_voice,
            cfg=cfg.generation,
            seed=hash((cfg.wake_word, label)) & 0xFFFFFFFF,
        )
        target_total = len(tasks)
        workers = settings.resolved_generation_workers()
        bus.phase(
            f"generate:piper:{label}",
            detail=f"{len(phrases)} phrases, {target_total} synths, {workers} workers",
        )
        bus.log(
            f"Piper {label}: {target_total} synths across {workers} workers"
        )
        i = 0
        for sample in piper.iter_parallel(tasks, workers=workers):
            wav_path = out_dir / f"piper_{label}_{i:07d}.wav"
            write_wav(wav_path, sample.audio, sample.sample_rate)
            written.append(wav_path)
            i += 1
            if i % 25 == 0:
                bus.progress(
                    f"generate:piper:{label}",
                    i / max(1, target_total),
                    detail=f"{i}/{target_total}",
                )
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

    # ElevenLabs (optional)
    if cfg.generation.use_elevenlabs and cfg.generation.elevenlabs_voice_ids:
        api_key = settings.elevenlabs_api_key
        if not api_key:
            bus.log("ElevenLabs requested but ELEVENLABS_API_KEY not set; skipping", level="warning")
        else:
            bus.phase(f"generate:elevenlabs:{label}")
            el = ElevenLabsGenerator(api_key=api_key, model_id=cfg.generation.elevenlabs_model)
            target_total = (
                len(cfg.generation.elevenlabs_voice_ids)
                * len(phrases)
                * n_per_phrase_per_voice
            )
            j = 0
            for sample in el.iter_samples(
                phrases=phrases,
                voice_ids=cfg.generation.elevenlabs_voice_ids,
                n_per_phrase_per_voice=n_per_phrase_per_voice,
                cfg=cfg.generation,
            ):
                wav_path = out_dir / f"el_{label}_{j:07d}.wav"
                write_wav(wav_path, sample.audio, sample.sample_rate)
                written.append(wav_path)
                j += 1
                if j % 10 == 0:
                    bus.progress(
                        f"generate:elevenlabs:{label}",
                        j / max(1, target_total),
                        detail=f"{j}/{target_total}",
                    )
                if state.cancel_flag.is_set():
                    break
            if not state.cancel_flag.is_set():
                skipped = target_total - j
                detail = f"{j}/{target_total}" + (f" ({skipped} skipped)" if skipped else "")
                bus.progress(f"generate:elevenlabs:{label}", 1.0, detail=detail)

    return written


def _build_features(
    positive_wavs: list[Path],
    negative_wavs: list[Path],
    common_voice_dir: Path | None,
    cfg: TrainRunConfig,
    run_dir: Path,
) -> tuple[FeatureMemmapDataset, FeatureMemmapDataset, dict[str, ShardManifest]]:
    """Augment + extract features into train/val memmaps."""
    extractor = FeatureExtractor()

    rir_dirs = collect_rir_dirs() if cfg.datasets.use_mit_rirs else []
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

    all_negatives = negative_wavs + cv_clips

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

    aug_per = cfg.augmentation.augmentations_per_clip

    train_capacity = (
        estimate_window_count(len(train_pos), aug_per)
        + estimate_window_count(len(train_neg), aug_per)
    ) or 1024
    val_capacity = (
        estimate_window_count(len(val_pos), aug_per)
        + estimate_window_count(len(val_neg), aug_per)
    ) or 256

    bus.log(f"Allocating train memmap up to {train_capacity:,} windows; val {val_capacity:,}")

    train_features_path = run_dir / "train_features.bin"
    val_features_path = run_dir / "val_features.bin"
    train_labels_path = run_dir / "train_labels.npy"
    val_labels_path = run_dir / "val_labels.npy"

    train_arr, _ = allocate_memmap(train_features_path, train_capacity)
    val_arr, _ = allocate_memmap(val_features_path, val_capacity)
    train_labels = np.zeros(train_capacity, dtype=np.uint8)
    val_labels = np.zeros(val_capacity, dtype=np.uint8)

    bus.phase("features:extract", detail="train positives")
    train_cursor = build_features_from_wavs(
        train_pos, 1, extractor, augmenter, aug_per, train_arr, train_labels, 0
    )
    bus.progress("features:extract", 0.25)

    bus.phase("features:extract", detail="train negatives")
    train_cursor = build_features_from_wavs(
        train_neg, 0, extractor, augmenter, aug_per, train_arr, train_labels, train_cursor
    )
    bus.progress("features:extract", 0.5)

    bus.phase("features:extract", detail="val positives")
    val_cursor = build_features_from_wavs(
        val_pos, 1, extractor, augmenter, aug_per, val_arr, val_labels, 0
    )
    bus.progress("features:extract", 0.75)

    bus.phase("features:extract", detail="val negatives")
    val_cursor = build_features_from_wavs(
        val_neg, 0, extractor, augmenter, aug_per, val_arr, val_labels, val_cursor
    )
    bus.progress("features:extract", 1.0)

    # Truncate to actual cursor and re-create memmaps + labels at exact size.
    train_arr.flush()
    val_arr.flush()
    del train_arr, val_arr

    # Resize via memmap "view" + label array
    np.save(train_labels_path, train_labels[:train_cursor])
    np.save(val_labels_path, val_labels[:val_cursor])
    # Truncate the binary files so the dataset memmap reads the right shape.
    bytes_per_window = CLASSIFIER_WINDOW_EMBEDDINGS * EMBEDDING_DIM * 4
    with open(train_features_path, "r+b") as f:
        f.truncate(train_cursor * bytes_per_window)
    with open(val_features_path, "r+b") as f:
        f.truncate(val_cursor * bytes_per_window)

    train_ds = FeatureMemmapDataset(train_features_path, train_labels_path)
    val_ds = FeatureMemmapDataset(val_features_path, val_labels_path)

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
            n_windows=train_cursor,
            label_counts=label_counts_train,
        ),
        "val": ShardManifest(
            features_path=val_features_path,
            labels_path=val_labels_path,
            n_windows=val_cursor,
            label_counts=label_counts_val,
        ),
    }
    save_manifest(run_dir, manifests)
    return train_ds, val_ds, manifests


def _run(cfg: TrainRunConfig) -> None:
    settings = get_settings()
    state.config = cfg
    state.status = "running"
    state.started_at = time.time()
    state.error = None
    state.cancel_flag.clear()

    try:
        run_id, run_dir = _make_run_dir(cfg)
        state.run_id = run_id
        state.run_dir = run_dir
        bus.publish("run_started", run_id=run_id, run_dir=str(run_dir), wake_word=cfg.wake_word)
        _save_config(run_dir, cfg)

        # 1. Positive phrases
        positive_phrases = cfg.generation.positive_phrases or [cfg.wake_word]
        bus.phase("generate:positive", detail=f"{len(positive_phrases)} phrases")
        positive_wavs = _generate_samples(
            positive_phrases,
            run_dir / "wavs" / "positive",
            cfg,
            cfg.generation.n_positive_per_phrase_per_voice,
            label="pos",
        )
        bus.log(f"Positive WAVs generated: {len(positive_wavs)}")

        if state.cancel_flag.is_set():
            raise RuntimeError("cancelled")

        # 2a. Hard negatives (user-supplied phrases the model must NOT trigger on).
        # Synthesized with the same emphasis as positives so the model strongly
        # learns to reject them.
        negative_wavs: list[Path] = []
        hard_negative_phrases = cfg.generation.negative_phrases or []
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
            bus.log(f"Hard-negative WAVs generated: {len(negative_wavs)}")

        if state.cancel_flag.is_set():
            raise RuntimeError("cancelled")

        # 2b. Auto-generated adversarial phrases (phonetic neighbors + generic
        # conversational pool). These broaden the negative distribution.
        adv_phrases = build_adversarial_phrases(
            cfg.wake_word,
            cfg.generation.n_adversarial_phrases,
            seed=cfg.training.seed,
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

        # 3. Augmentation corpora
        bus.phase("download:corpora")
        corpora = ensure_corpora(
            use_mit_rirs=cfg.datasets.use_mit_rirs,
            use_musan=cfg.datasets.use_musan_noise or cfg.datasets.use_musan_music,
            use_fsd50k=cfg.datasets.use_fsd50k,
            use_common_voice=cfg.datasets.use_common_voice_negatives,
            common_voice_subset=cfg.datasets.common_voice_subset,
            progress=lambda name, frac: bus.progress(f"download:{name}", frac),
        )
        bus.log(f"Corpora ready: {list(corpora.keys())}")

        if state.cancel_flag.is_set():
            raise RuntimeError("cancelled")

        # 4 + 5. Features
        bus.phase("features:build")
        train_ds, val_ds, _manifests = _build_features(
            positive_wavs=positive_wavs,
            negative_wavs=negative_wavs,
            common_voice_dir=corpora.get("common_voice"),
            cfg=cfg,
            run_dir=run_dir,
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
            cancel_flag=state.cancel_flag,
        )

        # 7. Publish to models dir
        final_path = settings.models_dir / f"{run_id}.onnx"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        from shutil import copy2

        copy2(result.onnx_path, final_path)
        state.onnx_path = final_path

        (run_dir / "result.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "wake_word": cfg.wake_word,
                    "best_val_loss": result.best_val_loss,
                    "best_val_recall_at_p95": result.best_val_recall,
                    "best_step": result.best_step,
                    "onnx_path": str(final_path),
                    "history": result.history,
                },
                indent=2,
            )
        )

        state.status = "succeeded"
        bus.complete(run_id=run_id, onnx_path=str(final_path))
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
