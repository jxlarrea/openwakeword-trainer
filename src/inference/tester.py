"""Run inference on uploaded audio using a trained ONNX model."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.data.features import FeatureExtractor, float32_to_int16

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    time_seconds: float
    score: float


@dataclass
class TestResult:
    duration_seconds: float
    max_score: float
    mean_score: float
    triggered: bool
    threshold: float
    detections: list[Detection]
    score_curve: list[tuple[float, float]]  # (time_s, score) sampled


class ModelTester:
    """Loads a trained classifier ONNX + the shared feature extractor."""

    def __init__(self, model_path: Path) -> None:
        providers = ort.get_available_providers()
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred if p in providers] or ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        self._input = self._session.get_inputs()[0].name
        self._extractor = FeatureExtractor(providers=providers)
        logger.info("ModelTester loaded %s", model_path)

    def score_audio(
        self, audio_float32: np.ndarray, sample_rate: int, threshold: float = 0.5
    ) -> TestResult:
        if audio_float32.ndim > 1:
            audio_float32 = audio_float32.mean(axis=1)
        if sample_rate != 16_000:
            from math import gcd
            from scipy.signal import resample_poly

            g = gcd(sample_rate, 16_000)
            audio_float32 = resample_poly(audio_float32, 16_000 // g, sample_rate // g).astype(np.float32)

        int16 = float32_to_int16(audio_float32)
        windows = self._extractor.classifier_inputs(int16)
        if windows.shape[0] == 0:
            return TestResult(
                duration_seconds=audio_float32.size / 16_000.0,
                max_score=0.0,
                mean_score=0.0,
                triggered=False,
                threshold=threshold,
                detections=[],
                score_curve=[],
            )

        # Run inference per-window. Small enough to do in one batch.
        scores: list[float] = []
        # Some openwakeword-style ONNXes were exported with batch=1 only;
        # handle both shapes.
        try:
            out = self._session.run(None, {self._input: windows})[0]
            scores = out.reshape(-1).astype(np.float32).tolist()
        except Exception:
            for i in range(windows.shape[0]):
                out = self._session.run(None, {self._input: windows[i : i + 1]})[0]
                scores.append(float(out.reshape(-1)[0]))

        # Each window represents ~80 ms of new audio (one embedding hop is 8 mel
        # frames ~= 100 ms; classifier window stride is one embedding).
        hop_seconds = 0.08
        score_curve = [(i * hop_seconds, s) for i, s in enumerate(scores)]
        detections = [
            Detection(time_seconds=t, score=s)
            for t, s in score_curve
            if s >= threshold
        ]
        return TestResult(
            duration_seconds=audio_float32.size / 16_000.0,
            max_score=float(max(scores)),
            mean_score=float(np.mean(scores)),
            triggered=any(s >= threshold for s in scores),
            threshold=threshold,
            detections=detections,
            score_curve=score_curve,
        )
