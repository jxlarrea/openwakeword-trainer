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
from src.tts.voices import list_english_voices


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
    try:
        generation["piper_voices"] = [
            {"voice_key": v.key} for v in list_english_voices()
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


def create_session(wake_word: str) -> dict[str, Any]:
    wake_word = wake_word.strip()
    sid = slugify(wake_word)
    if not wake_word or not sid:
        raise ValueError("Wake word is required")
    path = session_dir(sid)
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
    return session_summary(sid)


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
    return session_summary(sid)


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


def get_session(session_id: str) -> dict[str, Any]:
    return session_summary(session_id, include_size=True)


def delete_session(session_id: str) -> None:
    sid = slugify(session_id)
    path = session_dir(sid)
    if path.exists():
        shutil.rmtree(path)
    model_path = get_settings().models_dir / f"{sid}.onnx"
    model_path.unlink(missing_ok=True)


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total
