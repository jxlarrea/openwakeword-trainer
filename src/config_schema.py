"""Per-run training configuration. Submitted from the Web UI."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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
    use_kokoro: bool = True
    kokoro_voices: list[str] = Field(default_factory=list)
    n_kokoro_positive_per_phrase_per_voice: int = 2
    use_kokoro_for_negatives: bool = False
    n_kokoro_negative_per_phrase_per_voice: int = 1
    kokoro_speed_min: float = 0.9
    kokoro_speed_max: float = 1.1
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

    rir_probability: float = 0.9
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
    use_tablet_far_field_augmentation: bool = True
    tablet_far_field_probability: float = 0.6
    augmentations_per_clip: int = 6  # multiplies dataset size (DGX Spark default)


class DatasetConfig(BaseModel):
    """Which augmentation / negative-speech corpora to use."""

    use_mit_rirs: bool = True
    use_but_reverbdb: bool = True
    use_musan_noise: bool = True
    use_musan_music: bool = True
    use_fsd50k: bool = True
    use_common_voice_negatives: bool = True
    common_voice_subset: int = 20000  # DGX Spark default
    use_openwakeword_negative_features: bool = True
    use_openwakeword_validation_features: bool = True


class TrainingConfig(BaseModel):
    """Classifier head + optimizer."""

    # DGX Spark defaults: large batch, longer schedule, more patience.
    model_type: Literal["dnn", "rnn"] = "dnn"
    layer_dim: int = 128
    n_blocks: int = 3
    learning_rate: float = 1e-4
    batch_size: int = 2048
    positive_sample_fraction: float = 0.35
    negative_loss_weight: float = 3.0
    hard_negative_loss_weight: float = 2.0
    hard_negative_threshold: float = 0.70
    hard_negative_mining_top_k: int = 50_000
    hard_negative_finetune_steps: int = 0
    hard_negative_finetune_positive_fraction: float = 0.50
    max_steps: int = 200_000
    val_every_n_steps: int = 500
    target_false_positives_per_hour: float = 0.5
    min_recall_at_p95_for_export: float = 0.80
    min_recall_at_target_fp_for_export: float = 0.62
    early_stop_min_steps: int = 30_000
    early_stop_patience: int = 30
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

    @model_validator(mode="after")
    def _check_inputs_sane(self) -> "TrainRunConfig":
        """Reject configs that can't possibly produce a usable training dataset."""
        # At least one voice source.
        n_piper = len(self.generation.piper_voices)
        n_kokoro = (
            len(self.generation.kokoro_voices)
            if self.generation.use_kokoro
            else 0
        )
        n_eleven = (
            len(self.generation.elevenlabs_voice_ids)
            if self.generation.use_elevenlabs
            else 0
        )
        if n_piper == 0 and n_kokoro == 0 and n_eleven == 0:
            raise ValueError(
                "Select at least one Piper, Kokoro, or ElevenLabs voice."
            )

        # At least one augmentation corpus.
        any_corpus = (
            self.datasets.use_mit_rirs
            or self.datasets.use_but_reverbdb
            or self.datasets.use_musan_noise
            or self.datasets.use_musan_music
            or self.datasets.use_fsd50k
            or self.datasets.use_common_voice_negatives
            or self.datasets.use_openwakeword_negative_features
            or self.datasets.use_openwakeword_validation_features
        )
        if not any_corpus:
            raise ValueError(
                "Enable at least one augmentation corpus (MIT IR, BUT ReverbDB, MUSAN, FSD50K, or Common Voice)."
            )
        return self

    def slug(self) -> str:
        return "".join(c if c.isalnum() else "_" for c in self.wake_word.lower()).strip("_")
