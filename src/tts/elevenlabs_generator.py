"""ElevenLabs sample generator (optional path).

Uses output_format='pcm_16000' to skip resampling. Each call costs credits, so
the generator is rate-limited by the SDK and any failures are logged + skipped.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np

from src.config_schema import GenerationConfig
from src.tts.base import GeneratedSample

logger = logging.getLogger(__name__)


class ElevenLabsGenerator:
    """Wrapper around the ElevenLabs Python SDK."""

    def __init__(self, api_key: str, model_id: str = "eleven_multilingual_v2") -> None:
        from elevenlabs.client import ElevenLabs

        self._client = ElevenLabs(api_key=api_key)
        self.model_id = model_id

    def list_voices(self) -> list[dict]:
        """Return a flat list of {voice_id, name, labels} dicts."""
        try:
            response = self._client.voices.search()
        except Exception as exc:
            logger.error("ElevenLabs voices.search() failed: %s", exc)
            return []

        out: list[dict] = []
        for v in getattr(response, "voices", []) or []:
            out.append(
                {
                    "voice_id": getattr(v, "voice_id", ""),
                    "name": getattr(v, "name", ""),
                    "category": getattr(v, "category", ""),
                    "labels": dict(getattr(v, "labels", {}) or {}),
                }
            )
        return out

    def synthesize_one(self, text: str, voice_id: str) -> GeneratedSample:
        audio_iter = self._client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=self.model_id,
            output_format="pcm_16000",
        )
        chunks: list[bytes] = []
        for chunk in audio_iter:
            if chunk:
                chunks.append(chunk)
        raw = b"".join(chunks)
        audio_int16 = np.frombuffer(raw, dtype=np.int16)
        audio = audio_int16.astype(np.float32) / 32768.0
        return GeneratedSample(
            audio=audio,
            sample_rate=16_000,
            text=text,
            voice=f"elevenlabs:{voice_id}",
            metadata={"model_id": self.model_id},
        )

    def iter_samples(
        self,
        phrases: list[str],
        voice_ids: list[str],
        n_per_phrase_per_voice: int,
        cfg: GenerationConfig,
    ) -> Iterator[GeneratedSample]:
        for voice_id in voice_ids:
            for phrase in phrases:
                for _ in range(n_per_phrase_per_voice):
                    try:
                        yield self.synthesize_one(phrase, voice_id)
                    except Exception as exc:
                        logger.warning(
                            "ElevenLabs synth failed (voice=%s phrase=%r): %s",
                            voice_id,
                            phrase,
                            exc,
                        )
