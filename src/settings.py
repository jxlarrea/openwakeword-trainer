"""Global settings loaded from environment / .env."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide configuration. Read once at startup."""

    model_config = SettingsConfigDict(
        env_prefix="OWW_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Web UI
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # Storage
    data_dir: Path = Path("/data")

    # Workers
    generation_workers: int = 0  # 0 -> os.cpu_count()
    dataloader_workers: int = 0  # 0 -> min(8, cpu_count)

    # Optional API keys (also accepted via the unprefixed env vars).
    elevenlabs_api_key: str | None = Field(default=None, alias="ELEVENLABS_API_KEY")
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")

    # ----- derived paths -----
    @property
    def voices_dir(self) -> Path:
        return self.data_dir / "piper_voices"

    @property
    def augmentations_dir(self) -> Path:
        return self.data_dir / "augmentations"

    @property
    def rirs_dir(self) -> Path:
        return self.augmentations_dir / "rirs"

    @property
    def musan_dir(self) -> Path:
        return self.augmentations_dir / "musan"

    @property
    def fsd50k_dir(self) -> Path:
        return self.augmentations_dir / "fsd50k"

    @property
    def common_voice_dir(self) -> Path:
        return self.augmentations_dir / "common_voice"

    @property
    def generated_dir(self) -> Path:
        return self.data_dir / "generated"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    def ensure_dirs(self) -> None:
        for p in (
            self.data_dir,
            self.voices_dir,
            self.augmentations_dir,
            self.rirs_dir,
            self.musan_dir,
            self.fsd50k_dir,
            self.common_voice_dir,
            self.generated_dir,
            self.models_dir,
            self.runs_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)

    def resolved_generation_workers(self) -> int:
        # Cap at 8 by default: each Piper worker has its own onnxruntime
        # session that internally spawns more threads. More than 8 processes
        # causes thread thrashing on most hosts (including DGX Spark's
        # 20-core Grace CPU). Set OWW_GENERATION_WORKERS explicitly to override.
        return self.generation_workers or min(8, max(1, (os.cpu_count() or 2)))

    def resolved_dataloader_workers(self) -> int:
        return self.dataloader_workers or min(8, max(1, (os.cpu_count() or 2)))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
