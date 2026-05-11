"""Piper voice manifest + downloads.

The Piper team publishes a manifest at
https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json that lists
every voice. We fetch it once, cache it, then download voice files on demand.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

from src.settings import get_settings

logger = logging.getLogger(__name__)

VOICES_REPO = "rhasspy/piper-voices"
VOICES_MANIFEST_URL = (
    f"https://huggingface.co/{VOICES_REPO}/resolve/main/voices.json"
)

_manifest_lock = threading.Lock()
_cached_manifest: dict | None = None


@dataclass(frozen=True)
class PiperVoiceInfo:
    """Metadata for a single Piper voice (one quality of one voice)."""

    key: str  # e.g. "en_US-libritts-high"
    language_code: str  # e.g. "en_US"
    language_family: str  # e.g. "en"
    voice_name: str  # e.g. "libritts"
    quality: str  # x_low | low | medium | high
    sample_rate: int
    n_speakers: int
    onnx_path_in_repo: str
    config_path_in_repo: str

    @property
    def is_multi_speaker(self) -> bool:
        return self.n_speakers > 1


def fetch_manifest(force: bool = False) -> dict:
    """Download (or read cached) voices.json. Cached on disk + in memory."""
    global _cached_manifest
    settings = get_settings()
    cache_path = settings.voices_dir / "voices.json"

    with _manifest_lock:
        if _cached_manifest is not None and not force:
            return _cached_manifest

        if cache_path.exists() and not force:
            _cached_manifest = json.loads(cache_path.read_text())
            return _cached_manifest

        logger.info("Fetching Piper voices manifest from HF")
        resp = requests.get(VOICES_MANIFEST_URL, timeout=30)
        resp.raise_for_status()
        manifest = resp.json()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(manifest))
        _cached_manifest = manifest
        return manifest


def list_english_voices() -> list[PiperVoiceInfo]:
    """All English voices across qualities, sorted by family + quality."""
    manifest = fetch_manifest()
    voices: list[PiperVoiceInfo] = []
    quality_rank = {"x_low": 0, "low": 1, "medium": 2, "high": 3}

    for key, entry in manifest.items():
        lang = entry.get("language", {})
        family = lang.get("family", "")
        if family != "en":
            continue
        try:
            files = entry["files"]
            onnx_files = [p for p in files if p.endswith(".onnx")]
            cfg_files = [p for p in files if p.endswith(".onnx.json")]
            if not onnx_files or not cfg_files:
                continue
            voices.append(
                PiperVoiceInfo(
                    key=key,
                    language_code=entry.get("language", {}).get("code", "en_US"),
                    language_family=family,
                    voice_name=entry.get("name", ""),
                    quality=entry.get("quality", "medium"),
                    sample_rate=int(entry.get("audio", {}).get("sample_rate", 22050)),
                    n_speakers=int(entry.get("num_speakers", 1)),
                    onnx_path_in_repo=onnx_files[0],
                    config_path_in_repo=cfg_files[0],
                )
            )
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping malformed manifest entry %s: %s", key, exc)

    voices.sort(key=lambda v: (v.language_code, -quality_rank.get(v.quality, 0), v.voice_name))
    return voices


def get_voice_info(key: str) -> PiperVoiceInfo:
    for v in list_english_voices():
        if v.key == key:
            return v
    raise KeyError(f"Unknown Piper voice: {key}")


def ensure_voice_downloaded(info: PiperVoiceInfo) -> tuple[Path, Path]:
    """Download the voice's ONNX and JSON config if not already cached.

    hf_hub_download returns a symlink under the snapshot tree pointing at a
    blob in the cache. We resolve to the blob and copy it to the flat layout
    /data/piper_voices/<voice_key>/<filename> that the rest of the code expects.
    """
    import shutil

    settings = get_settings()
    target_dir = settings.voices_dir / info.key
    target_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = str(settings.voices_dir / ".hf_cache")

    onnx_dst = target_dir / Path(info.onnx_path_in_repo).name
    cfg_dst = target_dir / Path(info.config_path_in_repo).name

    def _stale(p: Path) -> bool:
        # is_symlink returns True for broken symlinks; exists() returns False.
        return p.is_symlink() and not p.exists()

    if _stale(onnx_dst):
        onnx_dst.unlink()
    if _stale(cfg_dst):
        cfg_dst.unlink()

    if not onnx_dst.exists():
        logger.info("Downloading Piper voice %s (onnx)", info.key)
        src = Path(
            hf_hub_download(
                repo_id=VOICES_REPO,
                filename=info.onnx_path_in_repo,
                cache_dir=cache_dir,
            )
        ).resolve()
        shutil.copy(src, onnx_dst)

    if not cfg_dst.exists():
        logger.info("Downloading Piper voice %s (config)", info.key)
        src = Path(
            hf_hub_download(
                repo_id=VOICES_REPO,
                filename=info.config_path_in_repo,
                cache_dir=cache_dir,
            )
        ).resolve()
        shutil.copy(src, cfg_dst)

    return onnx_dst, cfg_dst
