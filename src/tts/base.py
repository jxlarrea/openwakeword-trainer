"""Common types for TTS sample generation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GeneratedSample:
    """A single TTS-generated audio clip."""

    audio: np.ndarray  # float32 mono, range [-1, 1]
    sample_rate: int
    text: str
    voice: str
    metadata: dict = field(default_factory=dict)
