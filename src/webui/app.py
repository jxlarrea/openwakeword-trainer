"""FastAPI app for the trainer Web UI."""
from __future__ import annotations

import asyncio
import io
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from src.config_schema import (
    AugmentationConfig,
    DatasetConfig,
    GenerationConfig,
    TrainRunConfig,
    TrainingConfig,
    VoiceSelection,
)
from src.inference.tester import ModelTester
from src.pipeline import orchestrator
from src.sessions import (
    create_session,
    delete_all_disk_cache_preserving_sessions,
    delete_everything,
    delete_session,
    delete_session_cache,
    disk_cache_summary,
    get_session,
    list_sessions,
    save_session_config,
    slugify,
)
from src.settings import get_settings
from src.system_monitor import sample_system
from src.train.progress import bus
from src.train.stress_test import run as run_stress_test
from src.tts.kokoro_generator import default_kokoro_voice_keys, list_kokoro_voices
from src.tts.piper_generator import PiperGenerator
from src.tts.voices import list_english_voices
from src.augment.downloader import OPENWAKEWORD_ACAV100M_FILE, OPENWAKEWORD_VALIDATION_FILE

logger = logging.getLogger(__name__)


_BASE_DIR = Path(__file__).parent
_TEMPLATES = Jinja2Templates(directory=str(_BASE_DIR / "templates"))


def _decode_uploaded_audio(data: bytes) -> tuple[np.ndarray, int]:
    """Decode browser-uploaded audio, including MediaRecorder WebM/Opus."""
    try:
        arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        return np.asarray(arr, dtype=np.float32), int(sr)
    except Exception as soundfile_exc:
        try:
            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    "pipe:0",
                    "-f",
                    "f32le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "pipe:1",
                ],
                input=data,
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as ffmpeg_exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not decode audio. The server tried libsndfile and ffmpeg; "
                    f"libsndfile error: {soundfile_exc}; ffmpeg error: {ffmpeg_exc}"
                ),
            ) from ffmpeg_exc
        if not proc.stdout:
            raise HTTPException(status_code=400, detail="Decoded audio was empty")
        return np.frombuffer(proc.stdout, dtype=np.float32).copy(), 16_000


def create_app() -> FastAPI:
    app = FastAPI(title="OpenWakeWord Trainer", version="0.1.0")
    app.mount(
        "/static",
        StaticFiles(directory=str(_BASE_DIR / "static")),
        name="static",
    )
    assets_dir = _BASE_DIR.parent / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    api = APIRouter()
    register_routes(api)
    app.include_router(api)
    return app


# ----------------------------------------------------------------------------- #
# Routes
# ----------------------------------------------------------------------------- #


def register_routes(api: APIRouter) -> None:
    @api.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True, "status": orchestrator.state.status}

    @api.api_route("/favicon.ico", methods=["GET", "HEAD"], include_in_schema=False)
    def favicon() -> FileResponse:
        path = _BASE_DIR.parent / "assets" / "favicon.ico"
        if not path.exists():
            raise HTTPException(status_code=404, detail="favicon not found")
        return FileResponse(path, media_type="image/x-icon")

    @api.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(
            request,
            "index.html",
            {
                **_common_template_context(),
                "active_page": "trainer",
            },
        )

    @api.get("/tester", response_class=HTMLResponse)
    def tester_page(request: Request) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(
            request,
            "tester.html",
            {
                **_common_template_context(include_voices=False),
                "active_page": "tester",
            },
        )

    @api.get("/system", response_class=HTMLResponse)
    def system_page(request: Request) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(
            request,
            "system.html",
            {
                **_common_template_context(include_voices=False),
                "active_page": "system",
                "disk": disk_cache_summary(),
            },
        )

    # ----- Train endpoints -----

    @api.get("/api/sessions")
    def sessions_list():
        return list_sessions()

    @api.get("/api/system/disk")
    def system_disk():
        return disk_cache_summary()

    @api.delete("/api/system/sessions/{session_id}/cache")
    def system_session_cache_delete(session_id: str):
        if orchestrator.state.status == "running":
            raise HTTPException(status_code=409, detail="Cannot delete cache while training is running.")
        sid = slugify(session_id)
        reclaimed = delete_session_cache(sid)
        return {"deleted": True, "id": sid, "reclaimed_bytes": reclaimed, "disk": disk_cache_summary()}

    @api.delete("/api/system/cache")
    def system_cache_delete():
        if orchestrator.state.status == "running":
            raise HTTPException(status_code=409, detail="Cannot delete cache while training is running.")
        reclaimed = delete_all_disk_cache_preserving_sessions()
        return {"deleted": True, "reclaimed_bytes": reclaimed, "disk": disk_cache_summary()}

    @api.delete("/api/system/all")
    def system_all_delete():
        if orchestrator.state.status == "running":
            raise HTTPException(status_code=409, detail="Cannot delete data while training is running.")
        reclaimed = delete_everything()
        bus.reset()
        return {"deleted": True, "reclaimed_bytes": reclaimed, "disk": disk_cache_summary()}

    @api.post("/api/sessions")
    async def sessions_create(request: Request):
        if orchestrator.state.status == "running":
            raise HTTPException(status_code=409, detail="Cannot create a session while training is running.")
        payload = await request.json()
        try:
            payload = payload or {}
            return create_session(
                payload.get("wake_word", ""),
                payload.get("session_id") or payload.get("session_name"),
            )
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @api.get("/api/sessions/{session_id}")
    def sessions_get(session_id: str):
        try:
            return get_session(session_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc

    @api.delete("/api/sessions/{session_id}")
    def sessions_delete(session_id: str):
        sid = slugify(session_id)
        if orchestrator.state.status == "running":
            raise HTTPException(status_code=409, detail="Cannot delete sessions while training is running.")
        delete_session(sid)
        return {"deleted": True, "id": sid}

    @api.post("/api/train/start")
    async def train_start(request: Request):
        # Accept either JSON or url-encoded form data, so the same endpoint
        # serves the JS client and a no-JS fallback.
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                payload = await request.json()
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc
        else:
            form = await request.form()
            payload = _form_to_config_payload(form)

        if not payload:
            raise HTTPException(status_code=400, detail="empty payload")

        if orchestrator.state.status == "running":
            raise HTTPException(status_code=409, detail="A run is already in progress.")

        try:
            session_id = slugify(payload.get("session_id") or payload.get("run_name") or "")
            cfg = _payload_to_run_config(payload)
            if session_id:
                cfg = cfg.model_copy(update={"run_name": session_id})
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid config: {exc}") from exc

        if cfg.run_name:
            save_session_config(cfg.run_name, cfg)
        ok = orchestrator.start_run(cfg)
        if not ok:
            raise HTTPException(status_code=409, detail="A run is already in progress.")
        return JSONResponse(orchestrator.state.to_dict())

    @api.post("/api/train/cancel")
    def train_cancel():
        ok = orchestrator.cancel_run()
        return {"cancelled": ok, **orchestrator.state.to_dict()}

    @api.get("/api/train/status")
    def train_status():
        return {
            **orchestrator.state.to_dict(),
            "system": sample_system(),
            "progress": bus.snapshot(),
        }

    # ----- SSE -----

    @api.get("/api/events")
    async def events(request: Request):
        async def event_stream():
            q = await bus.subscribe()
            last_system_t = 0.0
            try:
                # send a comment frame on connect to flush proxies
                yield {"event": "ping", "data": "open"}
                while True:
                    if await request.is_disconnected():
                        break
                    now = asyncio.get_running_loop().time()
                    if now - last_system_t >= 2.0:
                        yield {
                            "event": "system",
                            "data": json.dumps(
                                {
                                    "kind": "system",
                                    "status": orchestrator.state.status,
                                    **sample_system(),
                                }
                            ),
                        }
                        last_system_t = now
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=2.0)
                        yield {"event": ev.kind, "data": json.dumps(ev.to_dict())}
                    except asyncio.TimeoutError:
                        # heartbeat to keep the connection alive
                        yield {"event": "ping", "data": "keepalive"}
            finally:
                await bus.unsubscribe(q)

        return EventSourceResponse(event_stream())

    # ----- Sample auditioning -----

    @api.post("/api/audition/piper")
    async def audition_piper(payload: dict):
        text = (payload or {}).get("text", "").strip()
        voice_key = (payload or {}).get("voice_key", "").strip()
        speaker_id = (payload or {}).get("speaker_id")
        if not text or not voice_key:
            raise HTTPException(status_code=400, detail="text and voice_key required")
        if speaker_id is not None:
            try:
                speaker_id = int(speaker_id)
            except (TypeError, ValueError):
                speaker_id = None
        gen = PiperGenerator(use_cuda=False)
        sample = gen.synthesize_one(text=text, voice_key=voice_key, speaker_id=speaker_id)
        return Response(content=_audio_to_wav_bytes(sample.audio, sample.sample_rate), media_type="audio/wav")

    @api.get("/api/voices/piper")
    def list_voices_piper():
        try:
            voices = list_english_voices()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Failed to fetch voices: {exc}") from exc
        return [
            {
                "key": v.key,
                "language_code": v.language_code,
                "voice_name": v.voice_name,
                "quality": v.quality,
                "n_speakers": v.n_speakers,
                "sample_rate": v.sample_rate,
            }
            for v in voices
        ]

    # ----- Models -----

    @api.get("/api/models")
    def list_models():
        models_dir = get_settings().models_dir
        models = []
        for p in sorted(models_dir.glob("*.onnx")):
            package = p.with_suffix(".zip")
            models.append(
                {
                    "name": p.name,
                    "path": str(p),
                    "size": p.stat().st_size,
                    "package_name": package.name if package.exists() else None,
                    "package_size": package.stat().st_size if package.exists() else None,
                }
            )
        return models

    @api.get("/api/models/{name}")
    def download_model(name: str):
        path = (get_settings().models_dir / name).resolve()
        models_dir = get_settings().models_dir.resolve()
        if not str(path).startswith(str(models_dir)) or not path.exists() or path.suffix != ".onnx":
            raise HTTPException(status_code=404, detail="model not found")
        return FileResponse(path, filename=path.name, media_type="application/octet-stream")

    @api.get("/api/model-packages/{name}")
    def download_model_package(name: str):
        path = (get_settings().models_dir / name).resolve()
        models_dir = get_settings().models_dir.resolve()
        if not str(path).startswith(str(models_dir)) or not path.exists() or path.suffix != ".zip":
            raise HTTPException(status_code=404, detail="model package not found")
        return FileResponse(path, filename=path.name, media_type="application/zip")

    @api.post("/api/test/file")
    async def test_file(
        model_name: str = Form(...),
        threshold: float = Form(0.5),
        audio: UploadFile = File(...),
    ):
        path = (get_settings().models_dir / model_name).resolve()
        models_dir = get_settings().models_dir.resolve()
        if not str(path).startswith(str(models_dir)) or not path.exists():
            raise HTTPException(status_code=404, detail="model not found")
        data = await audio.read()
        arr, sr = _decode_uploaded_audio(data)
        tester = ModelTester(path)
        result = tester.score_audio(arr, sr, threshold=threshold)
        return {
            "duration_seconds": result.duration_seconds,
            "max_score": result.max_score,
            "mean_score": result.mean_score,
            "triggered": result.triggered,
            "threshold": result.threshold,
            "detections": [
                {"time_seconds": d.time_seconds, "score": d.score} for d in result.detections
            ],
            "score_curve": [{"t": t, "s": s} for t, s in result.score_curve],
        }

    @api.post("/api/test/stress")
    async def stress_test(payload: dict):
        payload = payload or {}
        model_name = str(payload.get("model_name") or "").strip()
        if not model_name:
            raise HTTPException(status_code=400, detail="model_name is required")
        threshold = _bounded_float(payload.get("threshold"), 0.5, 0.0, 1.0)
        batch_size = _bounded_int(payload.get("batch_size"), 8192, 1, 65536)
        max_windows_raw = _bounded_int(payload.get("max_windows"), 1_000_000, 0, 50_000_000)
        max_windows = max_windows_raw or None
        include_session = bool(payload.get("include_session", True))
        include_validation = bool(payload.get("include_validation", True))
        include_acav100m = bool(payload.get("include_acav100m", False))
        use_cuda = bool(payload.get("use_cuda", False))

        models_dir = get_settings().models_dir.resolve()
        model_path = (models_dir / model_name).resolve()
        if (
            not str(model_path).startswith(str(models_dir))
            or not model_path.exists()
            or model_path.suffix != ".onnx"
        ):
            raise HTTPException(status_code=404, detail="model not found")

        settings = get_settings()
        run_dir = None
        if include_session:
            candidate = (settings.runs_dir / model_path.stem).resolve()
            runs_dir = settings.runs_dir.resolve()
            if str(candidate).startswith(str(runs_dir)) and candidate.exists():
                run_dir = candidate

        external_features: list[Path] = []
        skipped_sources: list[dict[str, str]] = []
        if include_validation:
            validation_path = settings.openwakeword_features_dir / OPENWAKEWORD_VALIDATION_FILE
            if validation_path.exists():
                external_features.append(validation_path)
            else:
                skipped_sources.append({"name": "openWakeWord validation", "reason": "not downloaded"})
        if include_acav100m:
            acav_path = settings.openwakeword_features_dir / OPENWAKEWORD_ACAV100M_FILE
            if acav_path.exists():
                external_features.append(acav_path)
            else:
                skipped_sources.append({"name": "ACAV100M", "reason": "not downloaded"})
        if include_session and run_dir is None:
            skipped_sources.append({"name": "session shards", "reason": f"no run cache for {model_path.stem}"})
        if run_dir is None and not external_features:
            raise HTTPException(status_code=400, detail="No stress-test sources are available for this request.")

        try:
            report = await asyncio.to_thread(
                run_stress_test,
                model_path=model_path,
                threshold=threshold,
                run_dir=run_dir,
                external_features=external_features,
                batch_size=batch_size,
                max_windows=max_windows,
                use_cuda=use_cuda,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Stress test failed: {exc}") from exc

        report["skipped_sources"] = skipped_sources
        report["max_windows"] = max_windows
        return report


# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #


def _common_template_context(include_voices: bool = True) -> dict[str, Any]:
    settings = get_settings()
    models = sorted(settings.models_dir.glob("*.onnx"))
    model_items = []
    for model in models:
        package = model.with_suffix(".zip")
        model_items.append(
            {
                "name": model.name,
                "path": str(model),
                "size": model.stat().st_size,
                "package_name": package.name if package.exists() else None,
                "package_size": package.stat().st_size if package.exists() else None,
            }
        )
    context: dict[str, Any] = {
        "state": orchestrator.state.to_dict(),
        "models": model_items,
        "sessions": list_sessions(),
    }
    if include_voices:
        try:
            voices = list_english_voices()
        except Exception as exc:
            logger.warning("Failed to fetch Piper voices manifest: %s", exc)
            voices = []
        context["voices"] = voices
        context["kokoro_voices"] = list_kokoro_voices()
        context["default_kokoro_voice_keys"] = set(default_kokoro_voice_keys())
    return context


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def _bounded_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = default
    return max(minimum, min(maximum, out))


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = default
    return max(minimum, min(maximum, out))


def _form_to_config_payload(form) -> dict:
    """Translate a flat HTML form into the nested run-config payload."""

    def _bool(name: str) -> bool:
        v = form.get(name)
        if v is None:
            return False
        return str(v).lower() in ("1", "true", "on", "yes")

    def _int(name: str, default: int) -> int:
        v = form.get(name)
        if v in (None, ""):
            return default
        try:
            return int(v)
        except ValueError:
            return default

    def _float(name: str, default: float) -> float:
        v = form.get(name)
        if v in (None, ""):
            return default
        try:
            return float(v)
        except ValueError:
            return default

    positive_phrases = [p.strip() for p in (form.get("positive_phrases", "") or "").splitlines() if p.strip()]
    negative_phrases = [p.strip() for p in (form.get("negative_phrases", "") or "").splitlines() if p.strip()]
    voice_keys = form.getlist("piper_voice") if hasattr(form, "getlist") else form.get("piper_voice", [])
    if isinstance(voice_keys, str):
        voice_keys = [voice_keys]
    payload = {
        "wake_word": form.get("wake_word", "").strip(),
        "session_id": form.get("session_id", "").strip(),
        "run_name": form.get("run_name", "").strip(),
        "generation": {
            "positive_phrases": positive_phrases,
            "n_positive_per_phrase_per_voice": _int("n_positive_per_phrase_per_voice", 8),
            "negative_phrases": negative_phrases,
            "n_negative_per_phrase_per_voice": _int("n_negative_per_phrase_per_voice", 5),
            "n_adversarial_phrases": _int("n_adversarial_phrases", 8000),
            "n_adversarial_per_phrase_per_voice": _int("n_adversarial_per_phrase_per_voice", 1),
            "piper_voices": [{"voice_key": k} for k in voice_keys],
            "use_kokoro": _bool("use_kokoro"),
            "kokoro_voices": form.getlist("kokoro_voice") if hasattr(form, "getlist") else [],
            "n_kokoro_positive_per_phrase_per_voice": _int("n_kokoro_positive_per_phrase_per_voice", 2),
            "use_kokoro_for_negatives": _bool("use_kokoro_for_negatives"),
            "n_kokoro_negative_per_phrase_per_voice": _int("n_kokoro_negative_per_phrase_per_voice", 1),
            "kokoro_speed_min": _float("kokoro_speed_min", 0.9),
            "kokoro_speed_max": _float("kokoro_speed_max", 1.1),
        },
        "augmentation": {
            "rir_probability": _float("rir_probability", 0.9),
            "background_noise_probability": _float("background_noise_probability", 0.75),
            "use_tablet_far_field_augmentation": _bool("use_tablet_far_field_augmentation"),
            "tablet_far_field_probability": _float("tablet_far_field_probability", 0.75),
            "augmentations_per_clip": _int("augmentations_per_clip", 6),
        },
        "datasets": {
            "use_mit_rirs": _bool("use_mit_rirs"),
            "use_but_reverbdb": _bool("use_but_reverbdb"),
            "use_musan_noise": _bool("use_musan_noise"),
            "use_musan_music": _bool("use_musan_music"),
            "use_fsd50k": _bool("use_fsd50k"),
            "use_common_voice_negatives": _bool("use_common_voice_negatives"),
            "use_openwakeword_negative_features": _bool("use_openwakeword_negative_features"),
            "use_openwakeword_validation_features": _bool("use_openwakeword_validation_features"),
            "common_voice_subset": _int("common_voice_subset", 100000),
        },
        "training": {
            "model_type": form.get("model_type", "dnn"),
            "layer_dim": _int("layer_dim", 64),
            "n_blocks": _int("n_blocks", 3),
            "learning_rate": _float("learning_rate", 1e-4),
            "weight_decay": _float("weight_decay", 1e-2),
            "use_focal_loss": _bool("use_focal_loss"),
            "focal_gamma": _float("focal_gamma", 2.0),
            "label_smoothing": _float("label_smoothing", 0.05),
            "mixup_alpha": _float("mixup_alpha", 0.2),
            "max_negative_loss_weight": _float("max_negative_loss_weight", 1000.0),
            "lr_warmup_fraction": _float("lr_warmup_fraction", 0.2),
            "lr_hold_fraction": _float("lr_hold_fraction", 0.33),
            "lr_reduce_on_plateau": _bool("lr_reduce_on_plateau"),
            "batch_size": _int("batch_size", 2048),
            "positive_sample_fraction": _float("positive_sample_fraction", 0.08),
            "negative_loss_weight": _float("negative_loss_weight", 1.0),
            "hard_negative_loss_weight": _float("hard_negative_loss_weight", 1.0),
            "hard_negative_threshold": _float("hard_negative_threshold", 0.9),
            "hard_negative_mining_top_k": _int("hard_negative_mining_top_k", 50000),
            "hard_negative_finetune_steps": _int("hard_negative_finetune_steps", 0),
            "hard_negative_finetune_positive_fraction": _float("hard_negative_finetune_positive_fraction", 0.5),
            "max_steps": _int("max_steps", 50000),
            "val_every_n_steps": _int("val_every_n_steps", 500),
            "early_stop_patience": _int("early_stop_patience", 40),
            "early_stop_min_steps": _int("early_stop_min_steps", 30000),
            "target_false_positives_per_hour": _float("target_false_positives_per_hour", 0.5),
            "min_recall_at_target_fp_for_export": _float("min_recall_at_target_fp_for_export", 0.70),
            "max_calibration_threshold_for_export": _float("max_calibration_threshold_for_export", 0.80),
            "min_recall_at_0_5_for_export": _float("min_recall_at_0_5_for_export", 0.80),
            "max_fp_per_hour_at_0_5_for_export": _float("max_fp_per_hour_at_0_5_for_export", 10.0),
            "min_positive_median_score_for_export": _float("min_positive_median_score_for_export", 0.75),
            "min_positive_p10_score_for_export": _float("min_positive_p10_score_for_export", 0.35),
            "use_positive_curve_validation": _bool("use_positive_curve_validation"),
            "curve_validation_max_positive_clips": _int("curve_validation_max_positive_clips", 400),
            "min_curve_recall_for_export": _float("min_curve_recall_for_export", 0.65),
            "min_curve_median_peak_for_export": _float("min_curve_median_peak_for_export", 0.78),
            "min_curve_p10_peak_for_export": _float("min_curve_p10_peak_for_export", 0.02),
            "min_curve_median_frames_for_export": _int("min_curve_median_frames_for_export", 2),
            "min_curve_median_span_ms_for_export": _float("min_curve_median_span_ms_for_export", 160.0),
            "min_curve_confirmation_rate_for_export": _float("min_curve_confirmation_rate_for_export", 0.30),
            "use_tablet_curve_validation": _bool("use_tablet_curve_validation"),
            "tablet_curve_validation_variants_per_clip": _int("tablet_curve_validation_variants_per_clip", 1),
            "min_tablet_curve_recall_for_export": _float("min_tablet_curve_recall_for_export", 0.24),
            "min_tablet_curve_median_peak_for_export": _float("min_tablet_curve_median_peak_for_export", 0.27),
            "min_tablet_curve_p10_peak_for_export": _float("min_tablet_curve_p10_peak_for_export", 0.04),
            "min_tablet_curve_median_frames_for_export": _int("min_tablet_curve_median_frames_for_export", 0),
            "min_tablet_curve_median_span_ms_for_export": _float("min_tablet_curve_median_span_ms_for_export", 0.0),
            "min_tablet_curve_confirmation_rate_for_export": _float("min_tablet_curve_confirmation_rate_for_export", 0.08),
            "curve_confirmation_min_gap_ms": _float("curve_confirmation_min_gap_ms", 320.0),
            "seed": _int("seed", 4044),
        },
    }
    return payload


def _payload_to_run_config(payload: dict) -> TrainRunConfig:
    gen = payload.get("generation", {})
    aug = payload.get("augmentation", {})
    ds = payload.get("datasets", {})
    tr = payload.get("training", {})

    voice_selections = [
        VoiceSelection(**v) if isinstance(v, dict) else VoiceSelection(voice_key=str(v))
        for v in gen.get("piper_voices", [])
    ]

    return TrainRunConfig(
        wake_word=payload.get("wake_word", ""),
        run_name=payload.get("run_name", ""),
        generation=GenerationConfig(
            **{**gen, "piper_voices": voice_selections}
        ),
        augmentation=AugmentationConfig(**aug) if aug else AugmentationConfig(),
        datasets=DatasetConfig(**ds) if ds else DatasetConfig(),
        training=TrainingConfig(**tr) if tr else TrainingConfig(),
    )
