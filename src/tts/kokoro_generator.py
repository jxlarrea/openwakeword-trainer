"""Kokoro TTS sample generator.

Kokoro is used as an additive high-quality local TTS source. Piper remains the
main broad-coverage/adversarial generator; Kokoro adds more natural positives
and optional hard negatives.
"""
from __future__ import annotations

import logging
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from src.config_schema import GenerationConfig
from src.tts.base import GeneratedSample

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000
KOKORO_SAMPLE_RATE = 24_000


@dataclass(frozen=True)
class KokoroVoiceInfo:
    key: str
    language_code: str
    accent: str
    gender: str
    quality: str


KOKORO_ENGLISH_VOICES: list[KokoroVoiceInfo] = [
    KokoroVoiceInfo("af_heart", "en_US", "American", "female", "A"),
    KokoroVoiceInfo("af_bella", "en_US", "American", "female", "A-"),
    KokoroVoiceInfo("af_nicole", "en_US", "American", "female", "B-"),
    KokoroVoiceInfo("af_aoede", "en_US", "American", "female", "C+"),
    KokoroVoiceInfo("af_kore", "en_US", "American", "female", "C+"),
    KokoroVoiceInfo("af_sarah", "en_US", "American", "female", "C+"),
    KokoroVoiceInfo("af_alloy", "en_US", "American", "female", "C"),
    KokoroVoiceInfo("af_nova", "en_US", "American", "female", "C"),
    KokoroVoiceInfo("af_sky", "en_US", "American", "female", "C-"),
    KokoroVoiceInfo("af_jessica", "en_US", "American", "female", "D"),
    KokoroVoiceInfo("af_river", "en_US", "American", "female", "D"),
    KokoroVoiceInfo("am_fenrir", "en_US", "American", "male", "C+"),
    KokoroVoiceInfo("am_michael", "en_US", "American", "male", "C+"),
    KokoroVoiceInfo("am_puck", "en_US", "American", "male", "C+"),
    KokoroVoiceInfo("am_adam", "en_US", "American", "male", "F+"),
    KokoroVoiceInfo("am_echo", "en_US", "American", "male", "D"),
    KokoroVoiceInfo("am_eric", "en_US", "American", "male", "D"),
    KokoroVoiceInfo("am_liam", "en_US", "American", "male", "D"),
    KokoroVoiceInfo("am_onyx", "en_US", "American", "male", "D"),
    KokoroVoiceInfo("am_santa", "en_US", "American", "male", "D-"),
    KokoroVoiceInfo("bf_emma", "en_GB", "British", "female", "B-"),
    KokoroVoiceInfo("bf_isabella", "en_GB", "British", "female", "C"),
    KokoroVoiceInfo("bf_alice", "en_GB", "British", "female", "D"),
    KokoroVoiceInfo("bf_lily", "en_GB", "British", "female", "D"),
    KokoroVoiceInfo("bm_daniel", "en_GB", "British", "male", "D"),
    KokoroVoiceInfo("bm_fable", "en_GB", "British", "male", "D"),
    KokoroVoiceInfo("bm_george", "en_GB", "British", "male", "D"),
    KokoroVoiceInfo("bm_lewis", "en_GB", "British", "male", "D"),
]


def list_kokoro_voices() -> list[KokoroVoiceInfo]:
    return list(KOKORO_ENGLISH_VOICES)


def default_kokoro_voice_keys() -> list[str]:
    return [
        v.key
        for v in KOKORO_ENGLISH_VOICES
        if v.quality in {"A", "A-", "B-", "C+"}
    ]


def _lang_code_for_voice(voice_key: str) -> str:
    return "b" if voice_key.startswith("b") else "a"


def _resample_float(audio: np.ndarray, src_sr: int, dst_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if src_sr == dst_sr:
        return audio
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(src_sr, dst_sr)
    return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)


class KokoroGenerator:
    """Lazy-loads Kokoro pipelines and yields synthesized samples."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device or os.getenv("OWW_KOKORO_DEVICE", "cuda")
        self._pipelines = {}

    def _pipeline(self, lang_code: str):
        cached = self._pipelines.get(lang_code)
        if cached is not None:
            return cached
        from kokoro import KPipeline
        from kokoro.model import KModel

        logger.info("Loading Kokoro pipeline lang=%s device=%s", lang_code, self.device)
        model = True
        if self.device.startswith("cuda"):
            # Kokoro's default TorchSTFT path uses CUDA complex kernels. On GB10
            # Torch 2.7 can JIT those with an unsupported arch, so use Kokoro's
            # custom real-valued STFT while keeping the model itself on CUDA.
            model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).to(self.device).eval()
        pipeline = KPipeline(
            lang_code=lang_code,
            repo_id="hexgrad/Kokoro-82M",
            model=model,
            device=self.device,
        )
        self._pipelines[lang_code] = pipeline
        return pipeline

    def synthesize_one(
        self,
        text: str,
        voice_key: str,
        speed: float = 1.0,
    ) -> GeneratedSample:
        lang_code = _lang_code_for_voice(voice_key)
        pipeline = self._pipeline(lang_code)
        chunks: list[np.ndarray] = []
        for result in pipeline(text, voice=voice_key, speed=speed, split_pattern=None):
            audio = result[-1] if isinstance(result, tuple) else result.audio
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        audio_24k = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        audio = _resample_float(audio_24k, KOKORO_SAMPLE_RATE, TARGET_SAMPLE_RATE)
        np.clip(audio, -1.0, 1.0, out=audio)
        return GeneratedSample(
            audio=audio,
            sample_rate=TARGET_SAMPLE_RATE,
            text=text,
            voice=f"kokoro:{voice_key}",
            metadata={"speed": speed, "lang_code": lang_code},
        )

    def iter_samples(
        self,
        phrases: list[str],
        voice_keys: list[str],
        n_per_phrase_per_voice: int,
        cfg: GenerationConfig,
        seed: int = 0,
    ) -> Iterator[GeneratedSample]:
        rng = random.Random(seed)
        for voice_key in voice_keys:
            for phrase in phrases:
                for _ in range(n_per_phrase_per_voice):
                    try:
                        speed = rng.uniform(cfg.kokoro_speed_min, cfg.kokoro_speed_max)
                        yield self.synthesize_one(phrase, voice_key, speed=speed)
                    except Exception as exc:
                        logger.warning(
                            "Kokoro synth failed (voice=%s phrase=%r): %s",
                            voice_key,
                            phrase,
                            exc,
                        )
