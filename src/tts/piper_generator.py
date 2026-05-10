"""Piper TTS sample generator."""
from __future__ import annotations

import logging
import random
from collections.abc import Iterator
from pathlib import Path

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
    # Use scipy.signal.resample_poly for clean integer-ratio resampling.
    from math import gcd

    g = gcd(src_sr, TARGET_SAMPLE_RATE)
    up = TARGET_SAMPLE_RATE // g
    down = src_sr // g
    from scipy.signal import resample_poly

    return resample_poly(audio, up, down).astype(np.float32)


class PiperGenerator:
    """Lazy-loads Piper voices and yields synthesized samples."""

    def __init__(self, use_cuda: bool = False) -> None:
        self.use_cuda = use_cuda
        self._voices: dict[str, tuple[PiperVoice, PiperVoiceInfo]] = {}

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
        rng = random.Random(seed)
        for sel in voice_selections:
            info = get_voice_info(sel.voice_key)
            speakers: list[int | None]
            if sel.speaker_ids:
                speakers = list(sel.speaker_ids)
            elif info.is_multi_speaker:
                # Sample from the full speaker space.
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
