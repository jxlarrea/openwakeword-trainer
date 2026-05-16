"""Persistent wake-word sessions.

A session is the durable working directory for one wake word. It owns the run
directory under /data/runs/<session_id>, so generated WAVs/features/checkpoints
are reused whenever the user comes back to that wake word.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from src.config_schema import AugmentationConfig, DatasetConfig, GenerationConfig, TrainRunConfig, TrainingConfig
from src.settings import get_settings
from src.tts.kokoro_generator import default_kokoro_voice_keys
from src.tts.voices import list_english_voices

_DISK_SUMMARY_CACHE: dict[str, Any] | None = None
_DISK_SUMMARY_CACHE_T = 0.0
_DISK_SUMMARY_TTL_SECONDS = 30.0


def slugify(value: str) -> str:
    slug = "".join(c if c.isalnum() else "_" for c in value.lower()).strip("_")
    return "_".join(part for part in slug.split("_") if part)


def _sessions_root() -> Path:
    root = get_settings().runs_dir
    root.mkdir(parents=True, exist_ok=True)
    return root


def session_dir(session_id: str) -> Path:
    sid = slugify(session_id)
    if not sid:
        raise ValueError("Invalid session id")
    root = _sessions_root().resolve()
    path = (root / sid).resolve()
    if not str(path).startswith(str(root)):
        raise ValueError("Invalid session path")
    return path


def _session_path(session_id: str) -> Path:
    return session_dir(session_id) / "session.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _config_for_session(wake_word: str, session_id: str) -> dict[str, Any]:
    generation = GenerationConfig().model_dump(mode="json")
    generation["kokoro_voices"] = default_kokoro_voice_keys()
    try:
        generation["piper_voices"] = [
            {"voice_key": v.key}
            for v in list_english_voices()
            if v.quality in {"high", "medium"}
        ]
    except Exception:
        generation["piper_voices"] = []
    return {
        "wake_word": wake_word,
        "run_name": session_id,
        "generation": generation,
        "augmentation": AugmentationConfig().model_dump(mode="json"),
        "datasets": DatasetConfig().model_dump(mode="json"),
        "training": TrainingConfig().model_dump(mode="json"),
    }


def create_session(wake_word: str, session_id: str | None = None) -> dict[str, Any]:
    wake_word = wake_word.strip()
    explicit_session_id = session_id is not None and bool(str(session_id).strip())
    sid = slugify(session_id or wake_word)
    if not wake_word:
        raise ValueError("Wake word is required")
    if not sid:
        raise ValueError("Session name is required")
    path = session_dir(sid)
    if explicit_session_id and (
        (path / "session.json").exists() or (path / "config.json").exists()
    ):
        raise ValueError(f"Session '{sid}' already exists. Select it from the session list instead.")
    path.mkdir(parents=True, exist_ok=True)
    now = time.time()
    existing = _read_json(path / "session.json") or {}
    data = {
        "id": sid,
        "wake_word": existing.get("wake_word") or wake_word,
        "created_at": existing.get("created_at") or now,
        "updated_at": now,
        "config": existing.get("config") or _config_for_session(wake_word, sid),
    }
    (path / "session.json").write_text(json.dumps(data, indent=2))
    return session_summary(sid, include_size=False)


def save_session_config(session_id: str, cfg: TrainRunConfig) -> dict[str, Any]:
    sid = slugify(session_id or cfg.run_name or cfg.slug())
    path = session_dir(sid)
    path.mkdir(parents=True, exist_ok=True)
    existing = _read_json(path / "session.json") or {}
    now = time.time()
    cfg = cfg.model_copy(update={"run_name": sid})
    data = {
        "id": sid,
        "wake_word": cfg.wake_word,
        "created_at": existing.get("created_at") or now,
        "updated_at": now,
        "config": cfg.model_dump(mode="json"),
    }
    (path / "session.json").write_text(json.dumps(data, indent=2))
    return session_summary(sid, include_size=False)


def session_summary(session_id: str, include_size: bool = True) -> dict[str, Any]:
    sid = slugify(session_id)
    path = session_dir(sid)
    data = _read_json(path / "session.json")
    cfg_data = _read_json(path / "config.json")
    if data is None and cfg_data is None:
        raise FileNotFoundError(sid)
    config = data.get("config") if data else cfg_data
    wake_word = (data or {}).get("wake_word") or (config or {}).get("wake_word") or sid
    updated = (data or {}).get("updated_at")
    size_bytes = _dir_size(path) if include_size else None
    return {
        "id": sid,
        "wake_word": wake_word,
        "path": str(path),
        "created_at": (data or {}).get("created_at"),
        "updated_at": updated,
        "size_bytes": size_bytes,
        "has_model": (get_settings().models_dir / f"{sid}.onnx").exists(),
        "config": config,
    }


def list_sessions() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(_sessions_root().iterdir()):
        if not path.is_dir():
            continue
        if not (path / "session.json").exists() and not (path / "config.json").exists():
            continue
        try:
            out.append(session_summary(path.name, include_size=False))
        except Exception:
            continue
    out.sort(key=lambda s: (s.get("updated_at") or 0, s["id"]), reverse=True)
    return out


def list_sessions_with_size() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for session in list_sessions():
        try:
            out.append(session_summary(session["id"], include_size=True))
        except Exception:
            continue
    return out


def get_session(session_id: str, include_size: bool = False) -> dict[str, Any]:
    return session_summary(session_id, include_size=include_size)


def delete_session(session_id: str) -> None:
    sid = slugify(session_id)
    path = session_dir(sid)
    if path.exists():
        shutil.rmtree(path)
    models_dir = get_settings().models_dir
    for artifact_path in (models_dir / f"{sid}.onnx", models_dir / f"{sid}.zip"):
        artifact_path.unlink(missing_ok=True)


def delete_session_cache(session_id: str) -> int:
    """Delete generated artifacts for a session while preserving its settings."""

    sid = slugify(session_id)
    path = session_dir(sid)
    before = _dir_size(path) if path.exists() else 0
    if path.exists():
        session_json = _read_json(path / "session.json")
        config_json = _read_json(path / "config.json")
        for child in path.iterdir():
            if child.name == "session.json":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
        if session_json is None and config_json is not None:
            now = time.time()
            data = {
                "id": sid,
                "wake_word": config_json.get("wake_word") or sid,
                "created_at": now,
                "updated_at": now,
                "config": config_json,
            }
            (path / "session.json").write_text(json.dumps(data, indent=2))
    models_dir = get_settings().models_dir
    model_size = 0
    for artifact_path in (models_dir / f"{sid}.onnx", models_dir / f"{sid}.zip"):
        if artifact_path.exists():
            model_size += artifact_path.stat().st_size
            artifact_path.unlink(missing_ok=True)
    after = _dir_size(path) if path.exists() else 0
    return max(0, before + model_size - after)


def delete_all_session_caches() -> int:
    total = 0
    for session in list_sessions():
        total += delete_session_cache(session["id"])
    return total


def delete_all_sessions() -> int:
    total = 0
    for session in list_sessions():
        sid = session["id"]
        path = session_dir(sid)
        if path.exists():
            total += _dir_size(path)
        models_dir = get_settings().models_dir
        for artifact_path in (models_dir / f"{sid}.onnx", models_dir / f"{sid}.zip"):
            if artifact_path.exists():
                total += artifact_path.stat().st_size
        delete_session(sid)
    root = _sessions_root()
    for child in root.iterdir():
        if not child.is_dir():
            continue
        try:
            total += _dir_size(child)
            shutil.rmtree(child)
        except OSError:
            continue
    return total


def disk_cache_summary(*, include_sizes: bool = True, use_cache: bool = True) -> dict[str, Any]:
    global _DISK_SUMMARY_CACHE, _DISK_SUMMARY_CACHE_T
    now = time.time()
    if (
        include_sizes
        and use_cache
        and _DISK_SUMMARY_CACHE is not None
        and now - _DISK_SUMMARY_CACHE_T < _DISK_SUMMARY_TTL_SECONDS
    ):
        return _DISK_SUMMARY_CACHE

    settings = get_settings()
    cache_dirs = [
        settings.voices_dir,
        settings.augmentations_dir,
        settings.openwakeword_features_dir,
        settings.generated_dir,
    ]
    dirs = []
    for path in cache_dirs:
        dirs.append(
            {
                "path": str(path),
                "size_bytes": _dir_size(path) if include_sizes else None,
            }
        )
    sessions = list_sessions_with_size() if include_sizes else list_sessions()
    model_bytes = _dir_size(settings.models_dir) if include_sizes else None
    summary = {
        "sessions": sessions,
        "session_bytes": sum(int(s.get("size_bytes") or 0) for s in sessions),
        "model_bytes": model_bytes or 0,
        "cache_dirs": dirs,
        "cache_bytes": sum(int(d.get("size_bytes") or 0) for d in dirs) + (model_bytes or 0),
        "total_bytes": sum(int(s.get("size_bytes") or 0) for s in sessions)
        + sum(int(d.get("size_bytes") or 0) for d in dirs)
        + (model_bytes or 0),
        "sizes_included": include_sizes,
        "generated_at": now,
    }
    if include_sizes:
        _DISK_SUMMARY_CACHE = summary
        _DISK_SUMMARY_CACHE_T = now
    return summary


def delete_global_disk_cache() -> int:
    settings = get_settings()
    cache_dirs = [
        settings.voices_dir,
        settings.augmentations_dir,
        settings.openwakeword_features_dir,
        settings.generated_dir,
    ]
    total = 0
    for path in cache_dirs:
        if not path.exists():
            continue
        total += _dir_size(path)
        shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.onnx", "*.zip"):
        for model_path in settings.models_dir.glob(pattern):
            try:
                total += model_path.stat().st_size
                model_path.unlink()
            except OSError:
                continue
    settings.ensure_dirs()
    return total


def delete_all_disk_cache_preserving_sessions() -> int:
    return delete_all_session_caches() + delete_global_disk_cache()


def delete_everything() -> int:
    return delete_all_sessions() + delete_global_disk_cache()


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total
