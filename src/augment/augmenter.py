"""Audio augmentation pipeline.

Stacks audiomentations transforms in an order that matches the openWakeWord
reference notebook + a few extras. Order matters:

  1. ApplyImpulseResponse  (room reverb, applied to dry signal)
  2. AddBackgroundNoise    (mix in MUSAN/FSD50K background)
  3. AddGaussianSNR        (mic / preamp hiss)
  4. SevenBandParametricEQ (channel coloration)
  5. AirAbsorption         (high-frequency rolloff for distance)
  6. PitchShift            (small +/- semitone drift; do NOT shift speech a lot)
  7. TimeStretch           (small +/- 10% rate)
  8. Gain                  (overall level)
  9. Mp3Compression        (bandwidth-limited / coded audio)

This pipeline is mono, 16 kHz, float32 in [-1, 1].
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
from audiomentations import (
    AddBackgroundNoise,
    AddGaussianSNR,
    AirAbsorption,
    ApplyImpulseResponse,
    Compose,
    Gain,
    Mp3Compression,
    PitchShift,
    SevenBandParametricEQ,
    TimeStretch,
)

from src.config_schema import AugmentationConfig

logger = logging.getLogger(__name__)

# audiomentations spams a UserWarning per FSD50K clip ("had to be resampled
# from 44100 Hz to 16000 Hz") - it's informational, the actual resampling works
# fine, and FSD50K is 100% 44.1 kHz so every load triggers it.
warnings.filterwarnings(
    "ignore",
    message=".*had to be resampled from .* Hz to .* Hz.*",
    category=UserWarning,
)


def build_augmenter(
    cfg: AugmentationConfig,
    *,
    rir_dirs: list[Path] | None = None,
    background_noise_dirs: list[Path] | None = None,
) -> Compose:
    """Compose the full augmentation chain. Pass empty dirs to skip those steps."""
    transforms = []

    if rir_dirs:
        # audiomentations expects a single dir or list; we feed a flat list.
        ir_paths = [str(p) for p in rir_dirs if p.exists()]
        if ir_paths:
            transforms.append(
                ApplyImpulseResponse(
                    ir_path=ir_paths if len(ir_paths) > 1 else ir_paths[0],
                    p=cfg.rir_probability,
                )
            )

    if background_noise_dirs:
        bg_paths = [str(p) for p in background_noise_dirs if p.exists()]
        if bg_paths:
            transforms.append(
                AddBackgroundNoise(
                    sounds_path=bg_paths if len(bg_paths) > 1 else bg_paths[0],
                    min_snr_db=cfg.background_noise_min_snr_db,
                    max_snr_db=cfg.background_noise_max_snr_db,
                    p=cfg.background_noise_probability,
                )
            )

    transforms.extend(
        [
            AddGaussianSNR(
                min_snr_db=15.0, max_snr_db=40.0, p=cfg.gaussian_noise_probability
            ),
            SevenBandParametricEQ(
                min_gain_db=-6.0, max_gain_db=6.0, p=cfg.seven_band_eq_probability
            ),
            AirAbsorption(p=cfg.air_absorption_probability),
            PitchShift(
                min_semitones=-cfg.pitch_shift_semitones,
                max_semitones=cfg.pitch_shift_semitones,
                p=cfg.pitch_shift_probability,
            ),
            TimeStretch(
                min_rate=cfg.time_stretch_min_rate,
                max_rate=cfg.time_stretch_max_rate,
                p=cfg.time_stretch_probability,
                leave_length_unchanged=True,
            ),
            Gain(
                min_gain_db=cfg.gain_min_db,
                max_gain_db=cfg.gain_max_db,
                p=cfg.gain_probability,
            ),
            Mp3Compression(
                min_bitrate=32, max_bitrate=128, p=cfg.mp3_compression_probability
            ),
        ]
    )

    return Compose(transforms=transforms, shuffle=False)


def augment_clip(
    audio: np.ndarray,
    sample_rate: int,
    augmenter: Compose,
    n_variants: int = 1,
) -> list[np.ndarray]:
    """Return n_variants augmented copies of `audio`."""
    out: list[np.ndarray] = []
    for _ in range(n_variants):
        try:
            aug = augmenter(samples=audio.astype(np.float32), sample_rate=sample_rate)
            # Re-clip to [-1, 1] to keep downstream int16 conversion safe.
            np.clip(aug, -1.0, 1.0, out=aug)
            out.append(aug.astype(np.float32))
        except Exception as exc:
            logger.warning("Augmentation failed: %s", exc)
            out.append(audio.astype(np.float32))
    return out


def collect_rir_dirs() -> list[Path]:
    """Return RIR dirs that exist on disk."""
    from src.settings import get_settings

    settings = get_settings()
    candidates = [settings.rirs_dir, settings.rirs_dir / "mit"]
    return [p for p in candidates if p.exists() and any(p.rglob("*.wav"))]


def collect_background_noise_dirs(
    *, use_musan_noise: bool, use_musan_music: bool, use_fsd50k: bool
) -> list[Path]:
    """Return background-audio dirs that exist on disk."""
    from src.settings import get_settings

    settings = get_settings()
    out: list[Path] = []
    if use_musan_noise:
        d = settings.musan_dir / "musan" / "noise"
        if d.exists():
            out.append(d)
    if use_musan_music:
        d = settings.musan_dir / "musan" / "music"
        if d.exists():
            out.append(d)
    if use_fsd50k:
        # Two layouts supported:
        #   - HF mirror (Fhrozen/FSD50k):   clips/dev, clips/eval
        #   - Zenodo legacy:                 FSD50K.dev_audio, FSD50K.eval_audio
        # First match that has any wavs wins.
        for cand in (
            settings.fsd50k_dir / "clips" / "dev",
            settings.fsd50k_dir / "clips" / "eval",
            settings.fsd50k_dir / "FSD50K.dev_audio",
            settings.fsd50k_dir / "FSD50K.eval_audio",
        ):
            if cand.exists() and any(cand.glob("*.wav")):
                out.append(cand)
    return out
