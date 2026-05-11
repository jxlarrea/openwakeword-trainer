"""Per-run training configuration. Submitted from the Web UI."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class VoiceSelection(BaseModel):
    """A single Piper voice selection."""

    voice_key: str  # e.g. "en_US-libritts-high"
    speaker_ids: list[int] | None = None  # multi-speaker voices


class GenerationConfig(BaseModel):
    """How synthetic samples are produced."""

    positive_phrases: list[str] = Field(
        default_factory=list,
        description="Phrases that should trigger the wake word. Usually variants of the wake word itself.",
    )
    # Defaults tuned for the DGX Spark (GB10, 128 GB unified memory). Smaller
    # hosts may want to reduce sample volume and batch size.
    n_positive_per_phrase_per_voice: int = 4

    # User-supplied hard negatives - phrases the model must NOT trigger on.
    # Typical use: paste false-triggers observed in production
    # (e.g. "hey google", "hey man"). Each is synthesized on every selected
    # voice with the same emphasis as positives so the model strongly learns
    # to reject them.
    negative_phrases: list[str] = Field(
        default_factory=list,
        description="Hard-negative phrases observed to false-trigger. Same emphasis as positives.",
    )
    n_negative_per_phrase_per_voice: int = 4

    n_adversarial_phrases: int = 3000
    n_adversarial_per_phrase_per_voice: int = 1

    piper_voices: list[VoiceSelection] = Field(default_factory=list)
    use_elevenlabs: bool = False
    elevenlabs_voice_ids: list[str] = Field(default_factory=list)
    elevenlabs_model: str = "eleven_multilingual_v2"

    # Synthesis variability (Piper SynthesisConfig).
    length_scale_min: float = 0.85
    length_scale_max: float = 1.15
    noise_scale_min: float = 0.5
    noise_scale_max: float = 0.85
    noise_w_scale_min: float = 0.5
    noise_w_scale_max: float = 0.9


class AugmentationConfig(BaseModel):
    """Per-clip augmentation knobs."""

    rir_probability: float = 0.7
    background_noise_probability: float = 0.7
    background_noise_min_snr_db: float = 3.0
    background_noise_max_snr_db: float = 30.0
    gaussian_noise_probability: float = 0.3
    pitch_shift_probability: float = 0.3
    pitch_shift_semitones: float = 2.0
    time_stretch_probability: float = 0.3
    time_stretch_min_rate: float = 0.9
    time_stretch_max_rate: float = 1.1
    seven_band_eq_probability: float = 0.3
    air_absorption_probability: float = 0.2
    gain_probability: float = 0.5
    gain_min_db: float = -12.0
    gain_max_db: float = 3.0
    mp3_compression_probability: float = 0.2
    augmentations_per_clip: int = 5  # multiplies dataset size (DGX Spark default)


class DatasetConfig(BaseModel):
    """Which augmentation / negative-speech corpora to use."""

    use_mit_rirs: bool = True
    use_musan_noise: bool = True
    use_musan_music: bool = True
    use_fsd50k: bool = True
    use_common_voice_negatives: bool = True
    common_voice_subset: int = 15000  # DGX Spark default


class TrainingConfig(BaseModel):
    """Classifier head + optimizer."""

    # DGX Spark defaults: large batch, longer schedule, more patience.
    model_type: Literal["dnn", "rnn"] = "dnn"
    layer_dim: int = 128
    n_blocks: int = 1
    learning_rate: float = 1e-4
    batch_size: int = 2048
    max_steps: int = 75_000
    val_every_n_steps: int = 500
    target_false_positives_per_hour: float = 0.2
    early_stop_patience: int = 8
    seed: int = 42


class TrainRunConfig(BaseModel):
    """Top-level training-run config submitted from the UI."""

    wake_word: str = Field(..., min_length=1, description="Required. e.g. 'hey jarvis'.")
    run_name: str = Field(default="", description="Defaults to slug of wake_word + timestamp.")

    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    datasets: DatasetConfig = Field(default_factory=DatasetConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @field_validator("wake_word")
    @classmethod
    def _strip_wake_word(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("wake_word is required")
        return v

    def slug(self) -> str:
        return "".join(c if c.isalnum() else "_" for c in self.wake_word.lower()).strip("_")
