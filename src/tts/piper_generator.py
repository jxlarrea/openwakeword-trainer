"""Piper TTS sample generator.

Two execution paths:

- ``PiperGenerator.synthesize_one`` / ``iter_samples`` - single-process, used
  for the audition endpoint and any one-shot synthesis.
- ``PiperGenerator.iter_parallel`` - process-pool fan-out used by the training
  pipeline. Workers lazy-load voices in their own address space so the parent
  is never blocked behind one ONNX session.

Why processes and not threads: Piper's voice runtime (onnxruntime under the
hood) uses internal threading and is not friendly to Python-level concurrency.
A spawn-based pool keeps each worker isolated.
"""
from __future__ import annotations

import logging
import random
from collections.abc import Iterator
from typing import Any

import numpy as np
from piper import PiperVoice, SynthesisConfig

from src.config_schema import GenerationConfig, VoiceSelection
from src.tts.base import GeneratedSample
from src.tts.voices import PiperVoiceInfo, ensure_voice_downloaded, get_voice_info

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000


def _resample_to_16k(audio_int16: np.ndarray, src_sr: int) -> np.ndarray:
    """Convert int16 PCM at src_sr to float32 mono at 16 kHz, range [-1, 1]."""
    audio = audio_int16.astype(np.float32) / 32768.0
    if src_sr == TARGET_SAMPLE_RATE:
        return audio
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(src_sr, TARGET_SAMPLE_RATE)
    up = TARGET_SAMPLE_RATE // g
    down = src_sr // g
    return resample_poly(audio, up, down).astype(np.float32)


# ----------------------------------------------------------------------------- #
# Worker-side state + entry points (module-level so multiprocessing can pickle)
# ----------------------------------------------------------------------------- #

# Populated per-worker after _worker_init is called.
_WORKER_VOICES: dict[str, tuple[PiperVoice, PiperVoiceInfo]] = {}
_WORKER_ORT_PATCHED = False


def _worker_apply_ort_thread_cap() -> None:
    """Cap onnxruntime intra-op threads on the session Piper is about to create.

    Without this cap, every Piper worker's onnxruntime session spawns one
    thread per physical core, so N workers x N cores threads collide on a
    fixed number of cores. The cap (default 2) keeps total active threads
    near the physical core count when paired with the default worker count.

    Implemented as a monkey-patch of ``onnxruntime.InferenceSession.__init__``
    because Piper does not expose a ``SessionOptions`` knob. Runs once per
    worker process.
    """
    global _WORKER_ORT_PATCHED
    if _WORKER_ORT_PATCHED:
        return

    import os

    import onnxruntime as ort

    n_threads = int(os.environ.get("OWW_PIPER_ORT_THREADS", "2"))
    if n_threads <= 0:
        _WORKER_ORT_PATCHED = True
        return

    _orig_init = ort.InferenceSession.__init__

    def _patched_init(self, *args, sess_options=None, providers=None, **kwargs):  # noqa: ANN001
        if sess_options is None:
            sess_options = ort.SessionOptions()
        # Only set if the caller hasn't customized threading already.
        if sess_options.intra_op_num_threads == 0:
            sess_options.intra_op_num_threads = n_threads
            sess_options.inter_op_num_threads = 1
        return _orig_init(self, *args, sess_options=sess_options, providers=providers, **kwargs)

    ort.InferenceSession.__init__ = _patched_init
    _WORKER_ORT_PATCHED = True


def _worker_init() -> None:
    """Pool initializer. Resets per-worker state."""
    global _WORKER_VOICES
    _WORKER_VOICES = {}
    _worker_apply_ort_thread_cap()
    # Avoid noisy duplicate handlers in spawned workers.
    logging.basicConfig(level=logging.WARNING, force=True)


def _worker_load_voice(key: str) -> tuple[PiperVoice, PiperVoiceInfo]:
    cached = _WORKER_VOICES.get(key)
    if cached is not None:
        return cached
    info = get_voice_info(key)
    onnx_path, cfg_path = ensure_voice_downloaded(info)
    voice = PiperVoice.load(str(onnx_path), config_path=str(cfg_path), use_cuda=False)
    _WORKER_VOICES[key] = (voice, info)
    return voice, info


def _worker_synth(task: dict[str, Any]) -> dict[str, Any] | None:
    """Synthesize one task in a worker. Returns dict or None on failure."""
    try:
        voice, info = _worker_load_voice(task["voice_key"])
        syn_cfg = SynthesisConfig(
            length_scale=task["length_scale"],
            noise_scale=task["noise_scale"],
            noise_w_scale=task["noise_w_scale"],
            volume=1.0,
            speaker_id=task["speaker_id"],
            normalize_audio=False,
        )
        chunks: list[np.ndarray] = []
        sample_rate = info.sample_rate
        for chunk in voice.synthesize(task["text"], syn_config=syn_cfg):
            sample_rate = chunk.sample_rate
            chunks.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))
        audio_int16 = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        audio = _resample_to_16k(audio_int16, sample_rate)
        return {
            "audio_bytes": audio.tobytes(),
            "text": task["text"],
            "voice": task["voice_key"],
            "metadata": {
                "speaker_id": task["speaker_id"],
                "length_scale": task["length_scale"],
                "noise_scale": task["noise_scale"],
                "noise_w_scale": task["noise_w_scale"],
            },
        }
    except Exception as exc:  # noqa: BLE001
        # Worker exceptions kill the pool unless caught. Log and return None.
        logger.warning(
            "Piper worker synth failed (voice=%s text=%r): %s",
            task.get("voice_key"),
            task.get("text"),
            exc,
        )
        return None


# ----------------------------------------------------------------------------- #
# Generator API
# ----------------------------------------------------------------------------- #


class PiperGenerator:
    """Lazy-loads Piper voices and yields synthesized samples."""

    def __init__(self, use_cuda: bool = False) -> None:
        self.use_cuda = use_cuda
        self._voices: dict[str, tuple[PiperVoice, PiperVoiceInfo]] = {}

    # ----- single-process API (audition / one-shot use) -----

    def _load_voice(self, key: str) -> tuple[PiperVoice, PiperVoiceInfo]:
        if key in self._voices:
            return self._voices[key]
        info = get_voice_info(key)
        onnx_path, cfg_path = ensure_voice_downloaded(info)
        logger.info("Loading Piper voice: %s (cuda=%s)", key, self.use_cuda)
        voice = PiperVoice.load(
            str(onnx_path),
            config_path=str(cfg_path),
            use_cuda=self.use_cuda,
        )
        self._voices[key] = (voice, info)
        return voice, info

    def synthesize_one(
        self,
        text: str,
        voice_key: str,
        speaker_id: int | None = None,
        rng: random.Random | None = None,
        cfg: GenerationConfig | None = None,
    ) -> GeneratedSample:
        rng = rng or random.Random()
        voice, info = self._load_voice(voice_key)

        if cfg is None:
            length_scale = 1.0
            noise_scale = 0.667
            noise_w_scale = 0.8
        else:
            length_scale = rng.uniform(cfg.length_scale_min, cfg.length_scale_max)
            noise_scale = rng.uniform(cfg.noise_scale_min, cfg.noise_scale_max)
            noise_w_scale = rng.uniform(cfg.noise_w_scale_min, cfg.noise_w_scale_max)

        syn_cfg = SynthesisConfig(
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w_scale,
            volume=1.0,
            speaker_id=speaker_id,
            normalize_audio=False,
        )

        chunks: list[np.ndarray] = []
        sample_rate = info.sample_rate
        for chunk in voice.synthesize(text, syn_config=syn_cfg):
            sample_rate = chunk.sample_rate
            arr = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            chunks.append(arr)

        audio_int16 = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        audio = _resample_to_16k(audio_int16, sample_rate)

        return GeneratedSample(
            audio=audio,
            sample_rate=TARGET_SAMPLE_RATE,
            text=text,
            voice=voice_key,
            metadata={
                "speaker_id": speaker_id,
                "length_scale": length_scale,
                "noise_scale": noise_scale,
                "noise_w_scale": noise_w_scale,
            },
        )

    def iter_samples(
        self,
        phrases: list[str],
        voice_selections: list[VoiceSelection],
        n_per_phrase_per_voice: int,
        cfg: GenerationConfig,
        seed: int = 0,
    ) -> Iterator[GeneratedSample]:
        """Single-process iterator. Kept for backward compat; use iter_parallel
        for batch work in the training pipeline."""
        rng = random.Random(seed)
        for sel in voice_selections:
            info = get_voice_info(sel.voice_key)
            speakers: list[int | None]
            if sel.speaker_ids:
                speakers = list(sel.speaker_ids)
            elif info.is_multi_speaker:
                speakers = list(range(info.n_speakers))
            else:
                speakers = [None]

            for phrase in phrases:
                for _ in range(n_per_phrase_per_voice):
                    speaker_id = rng.choice(speakers) if speakers else None
                    try:
                        yield self.synthesize_one(
                            text=phrase,
                            voice_key=sel.voice_key,
                            speaker_id=speaker_id,
                            rng=rng,
                            cfg=cfg,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Piper synth failed (voice=%s phrase=%r): %s",
                            sel.voice_key,
                            phrase,
                            exc,
                        )

    # ----- multi-process API (training pipeline) -----

    def build_tasks(
        self,
        phrases: list[str],
        voice_selections: list[VoiceSelection],
        n_per_phrase_per_voice: int,
        cfg: GenerationConfig,
        seed: int = 0,
    ) -> list[dict[str, Any]]:
        """Pre-compute the full task list deterministically.

        Each task is a plain dict so it pickles cleanly across the pool.
        """
        rng = random.Random(seed)
        tasks: list[dict[str, Any]] = []
        for sel in voice_selections:
            info = get_voice_info(sel.voice_key)
            if sel.speaker_ids:
                speakers: list[int | None] = list(sel.speaker_ids)
            elif info.is_multi_speaker:
                speakers = list(range(info.n_speakers))
            else:
                speakers = [None]

            for phrase in phrases:
                for _ in range(n_per_phrase_per_voice):
                    tasks.append(
                        {
                            "voice_key": sel.voice_key,
                            "text": phrase,
                            "speaker_id": rng.choice(speakers) if speakers else None,
                            "length_scale": rng.uniform(
                                cfg.length_scale_min, cfg.length_scale_max
                            ),
                            "noise_scale": rng.uniform(
                                cfg.noise_scale_min, cfg.noise_scale_max
                            ),
                            "noise_w_scale": rng.uniform(
                                cfg.noise_w_scale_min, cfg.noise_w_scale_max
                            ),
                        }
                    )
        return tasks

    def iter_parallel(
        self,
        tasks: list[dict[str, Any]],
        workers: int,
        chunksize: int = 4,
    ) -> Iterator[GeneratedSample]:
        """Synthesize tasks across a process pool, yielding samples as they complete.

        Tasks complete out of order (imap_unordered) - keep the work
        embarrassingly parallel and don't rely on task order downstream.
        """
        if not tasks:
            return
        if workers <= 1:
            yield from self._inline_iter(tasks)
            return

        from multiprocessing import get_context

        # `spawn` avoids fork-with-threads pitfalls in onnxruntime / numpy.
        ctx = get_context("spawn")
        with ctx.Pool(processes=workers, initializer=_worker_init) as pool:
            for result in pool.imap_unordered(_worker_synth, tasks, chunksize=chunksize):
                if result is None:
                    continue
                audio = np.frombuffer(result["audio_bytes"], dtype=np.float32)
                yield GeneratedSample(
                    audio=audio,
                    sample_rate=TARGET_SAMPLE_RATE,
                    text=result["text"],
                    voice=result["voice"],
                    metadata=result["metadata"],
                )

    def _inline_iter(self, tasks: list[dict[str, Any]]) -> Iterator[GeneratedSample]:
        """Inline (single-process) fallback. Same output contract as iter_parallel."""
        for task in tasks:
            result = _worker_synth(task)
            if result is None:
                continue
            audio = np.frombuffer(result["audio_bytes"], dtype=np.float32)
            yield GeneratedSample(
                audio=audio,
                sample_rate=TARGET_SAMPLE_RATE,
                text=result["text"],
                voice=result["voice"],
                metadata=result["metadata"],
            )
