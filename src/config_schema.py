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
    n_positive_per_phrase_per_voice: int = 8

    # User-supplied hard negatives - phrases the model must NOT trigger on.
    # Typical use: paste false-triggers observed in production
    # (e.g. "hey google", "hey man"). Each is synthesized on every selected
    # voice with the same emphasis as positives so the model strongly learns
    # to reject them.
    negative_phrases: list[str] = Field(
        default_factory=list,
        description="Hard-negative phrases observed to false-trigger. Same emphasis as positives.",
    )
    n_negative_per_phrase_per_voice: int = 5

    n_adversarial_phrases: int = 8000
    n_adversarial_per_phrase_per_voice: int = 1

    piper_voices: list[VoiceSelection] = Field(default_factory=list)
    use_kokoro: bool = True
    kokoro_voices: list[str] = Field(default_factory=list)
    n_kokoro_positive_per_phrase_per_voice: int = 2
    use_kokoro_for_negatives: bool = True
    n_kokoro_negative_per_phrase_per_voice: int = 1
    kokoro_speed_min: float = 0.9
    kokoro_speed_max: float = 1.1

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
    background_noise_probability: float = 0.75
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
    tablet_far_field_probability: float = 0.75
    augmentations_per_clip: int = 6  # multiplies dataset size (DGX Spark default)


class DatasetConfig(BaseModel):
    """Which augmentation / negative-speech corpora to use."""

    use_mit_rirs: bool = True
    use_but_reverbdb: bool = True
    use_musan_noise: bool = True
    use_musan_music: bool = True
    use_fsd50k: bool = True
    use_common_voice_negatives: bool = True
    common_voice_subset: int = 100000  # DGX Spark default
    use_background_corpus_negatives: bool = True
    background_corpus_negative_subset: int = 20000
    use_openwakeword_negative_features: bool = True
    use_openwakeword_validation_features: bool = True


class TrainingConfig(BaseModel):
    """Classifier head + optimizer."""

    # DGX Spark defaults: large batch, longer schedule, more patience.
    model_type: Literal["dnn", "rnn"] = "dnn"
    layer_dim: int = 64
    n_blocks: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05
    mixup_alpha: float = 0.20
    positive_confidence_target: float = 0.90
    positive_confidence_weight: float = 0.0
    negative_confidence_target: float = 0.10
    negative_confidence_weight: float = 0.0
    separation_margin: float = 0.50
    separation_loss_weight: float = 0.0
    separation_top_k: int = 128
    # Negative pressure schedule ramps from 1 -> this value. Keep this high
    # enough that broad negative speech banks still shape the operating point;
    # export gates below decide whether the resulting model is deployable.
    max_negative_loss_weight: float = 1000.0
    lr_warmup_fraction: float = 0.20
    lr_hold_fraction: float = 0.33
    lr_reduce_on_plateau: bool = False
    lr_reduce_patience: int = 4
    lr_reduce_factor: float = 0.5
    min_learning_rate: float = 1e-6
    max_lr_reductions: int = 4
    batch_size: int = 2048
    positive_sample_fraction: float = 0.08
    negative_loss_weight: float = 1.0
    hard_negative_loss_weight: float = 1.0
    hard_negative_threshold: float = 0.90
    hard_negative_mining_top_k: int = 50_000
    hard_negative_finetune_steps: int = 0
    hard_negative_finetune_positive_fraction: float = 0.50
    hard_negative_refresh_on_plateau: bool = False
    hard_negative_refresh_top_k: int = 20_000
    hard_negative_refresh_steps: int = 300
    hard_negative_refresh_positive_fraction: float = 0.50
    max_hard_negative_refreshes: int = 3
    max_steps: int = 50_000
    val_every_n_steps: int = 500
    target_false_positives_per_hour: float = 0.5
    min_recall_at_p95_for_export: float = 0.80
    min_recall_at_target_fp_for_export: float = 0.70
    # Keep the raw operating point sane even though export calibration maps the
    # selected threshold to runtime 0.5. High raw thresholds and high raw FP/hr
    # are strong signs that sensitivity offsets or alternate runners will be
    # noisy in the real world.
    max_calibration_threshold_for_export: float = 0.80
    min_recall_at_0_5_for_export: float = 0.80
    max_fp_per_hour_at_0_5_for_export: float = 10.0
    min_positive_median_score_for_export: float = 0.75
    min_positive_p10_score_for_export: float = 0.35
    use_positive_curve_validation: bool = True
    curve_validation_max_positive_clips: int = 400
    # v13-style gates (loosened from v15/v16 which were unachievable for a
    # sharp model). A precise wake-word head naturally peaks across 3-5 hops
    # at the wake-word completion, not 5-7.
    min_curve_recall_for_export: float = 0.65
    min_curve_median_peak_for_export: float = 0.78
    min_curve_p10_peak_for_export: float = 0.02
    min_curve_median_frames_for_export: int = 2
    min_curve_median_span_ms_for_export: float = 160.0
    min_curve_confirmation_rate_for_export: float = 0.30
    use_tablet_curve_validation: bool = True
    # Generate a few tablet variants per clip so the metric is stable when the
    # tablet aug is now using per-clip training probabilities (no longer the
    # adversarial "100% tablet + 90% RIR + 50% bg" stacking).
    tablet_curve_validation_variants_per_clip: int = 1
    # Tablet validation is intentionally stricter than the generic curve check:
    # a model that only works on clean, front-facing TTS is not good enough for
    # wall-mounted tablets. Failed tablet gates must block export.
    min_tablet_curve_recall_for_export: float = 0.24
    min_tablet_curve_median_peak_for_export: float = 0.27
    min_tablet_curve_p10_peak_for_export: float = 0.04
    min_tablet_curve_median_frames_for_export: int = 0
    min_tablet_curve_median_span_ms_for_export: float = 0.0
    min_tablet_curve_confirmation_rate_for_export: float = 0.08
    curve_confirmation_min_gap_ms: float = 320.0
    positive_temporal_windows: int = 1
    positive_temporal_stride_embeddings: int = 1
    positive_context_seconds: float = 2.0
    early_stop_min_steps: int = 30_000
    exportable_quality_extension_steps: int = 30_000
    early_stop_patience: int = 40
    seed: int = 4044


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
        if n_piper == 0 and n_kokoro == 0:
            raise ValueError(
                "Select at least one Piper or Kokoro voice."
            )

        # At least one augmentation corpus.
        any_corpus = (
            self.datasets.use_mit_rirs
            or self.datasets.use_but_reverbdb
            or self.datasets.use_musan_noise
            or self.datasets.use_musan_music
            or self.datasets.use_fsd50k
            or self.datasets.use_common_voice_negatives
            or self.datasets.use_background_corpus_negatives
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
