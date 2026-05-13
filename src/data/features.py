"""Feature extraction using openWakeWord's bundled mel + speech_embedding ONNX models.

The openwakeword library ships these models via a v0.5.1 GitHub release; the
Dockerfile pre-downloads them. At runtime we wrap them with onnxruntime and
expose batch APIs.

Pipeline:
    int16 PCM @ 16kHz  -> mel spectrogram  -> 96-dim embeddings (one per ~80 ms hop)
                          (76 mel-frames    (input shape (N, 76, 32, 1))
                           x 32 mel-bins
                           per window,
                           hop 8 frames)

We sample 16 consecutive embeddings (~1.28 s) as the classifier input.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.settings import get_settings

logger = logging.getLogger(__name__)


# Per openwakeword/utils.py
EMBEDDING_WINDOW_FRAMES = 76  # mel frames consumed per embedding window
EMBEDDING_HOP_FRAMES = 8       # mel frames between consecutive embedding windows
EMBEDDING_DIM = 96
CLASSIFIER_WINDOW_EMBEDDINGS = 16  # 16 consecutive embeddings = ~1.28 s


_FEATURE_MODEL_FALLBACK_URLS = {
    "melspectrogram": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
    "embedding": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx",
}


def _resolve_model(name: str) -> Path:
    """Find the ONNX feature model.

    Strategy:
      1. Try to locate it via openwakeword.FEATURE_MODELS (set by openwakeword.__init__).
         Prefer the .onnx sibling of whatever path is configured.
      2. Fall back to fetching it directly from the v0.5.1 GitHub release into
         openwakeword's resources/models/ directory.
    """
    import openwakeword

    spec = getattr(openwakeword, "FEATURE_MODELS", {}).get(name)
    candidates: list[Path] = []
    if spec:
        mp = Path(spec["model_path"])
        candidates += [mp.with_suffix(".onnx"), mp]
        # Some openwakeword versions store .onnx as a separate filename.
        candidates.append(mp.parent / f"{name}.onnx")
        candidates.append(mp.parent / f"{name}_model.onnx")

    for c in candidates:
        if c.exists() and c.suffix == ".onnx":
            return c

    # Fallback: download directly.
    url = _FEATURE_MODEL_FALLBACK_URLS.get(name)
    if not url:
        raise FileNotFoundError(f"No fallback URL for feature model: {name}")

    # Choose a stable target dir under openwakeword/resources/models if we can.
    base_dir: Path
    if spec:
        base_dir = Path(spec["model_path"]).parent
    else:
        base_dir = Path(openwakeword.__file__).parent / "resources" / "models"
    base_dir.mkdir(parents=True, exist_ok=True)
    target = base_dir / Path(url).name

    if not target.exists():
        import requests

        logger.info("Downloading openwakeword feature model %s -> %s", name, target)
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(target, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
    return target


class FeatureExtractor:
    """Mel + speech_embedding ONNX runner."""

    def __init__(self, providers: list[str] | None = None) -> None:
        if providers is None:
            providers = ort.get_available_providers()
            preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            providers = [p for p in preferred if p in providers] or ["CPUExecutionProvider"]
        logger.info("FeatureExtractor providers: %s", providers)

        mel_path = _resolve_model("melspectrogram")
        emb_path = _resolve_model("embedding")
        self._mel = ort.InferenceSession(str(mel_path), providers=providers)
        self._emb = ort.InferenceSession(str(emb_path), providers=providers)

        self._mel_input = self._mel.get_inputs()[0].name
        self._emb_input = self._emb.get_inputs()[0].name

    def melspec(self, audio_int16: np.ndarray) -> np.ndarray:
        """Return mel spectrogram of shape (T, 32) for a 1-D int16 PCM clip."""
        if audio_int16.ndim != 1:
            audio_int16 = audio_int16.reshape(-1)
        # openwakeword passes shape (1, n_samples) float32 (despite int16 source).
        x = audio_int16.astype(np.float32).reshape(1, -1)
        out = self._mel.run(None, {self._mel_input: x})[0]
        # openwakeword applies x/10 + 2 transform.
        mel = out.squeeze() / 10.0 + 2.0
        return mel.astype(np.float32)

    def embeddings(self, audio_int16: np.ndarray) -> np.ndarray:
        """Return embeddings of shape (N, 96) for a 1-D int16 PCM clip.

        N depends on clip length: (T - 76) // 8 + 1 where T is mel-frame count.
        """
        mel = self.melspec(audio_int16)  # (T, 32)
        T = mel.shape[0]
        if T < EMBEDDING_WINDOW_FRAMES:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        # Build (N, 76, 32, 1) batch of stacked windows.
        n_windows = (T - EMBEDDING_WINDOW_FRAMES) // EMBEDDING_HOP_FRAMES + 1
        windows = np.stack(
            [
                mel[i * EMBEDDING_HOP_FRAMES : i * EMBEDDING_HOP_FRAMES + EMBEDDING_WINDOW_FRAMES]
                for i in range(n_windows)
            ]
        )
        windows = windows[..., None].astype(np.float32)  # (N, 76, 32, 1)
        out = self._emb.run(None, {self._emb_input: windows})[0]
        # Output shape is typically (N, 1, 1, 96); squeeze.
        return out.reshape(out.shape[0], -1).astype(np.float32)

    def fixed_classifier_input(self, audio_int16: np.ndarray) -> np.ndarray:
        """Return one streaming-compatible classifier input for a clip.

        Reference openWakeWord-style training uses one fixed example per
        augmented clip. If the clip produces more than 16 embedding timesteps,
        keep the last 16 because inference asks whether the wake word occurred
        at the trailing edge of the rolling audio buffer. If it produces fewer,
        left-pad so the real audio remains anchored to the end.
        """
        emb = self.embeddings(audio_int16)
        if emb.shape[0] >= CLASSIFIER_WINDOW_EMBEDDINGS:
            return emb[-CLASSIFIER_WINDOW_EMBEDDINGS:].astype(np.float32)
        out = np.zeros((CLASSIFIER_WINDOW_EMBEDDINGS, EMBEDDING_DIM), dtype=np.float32)
        if emb.shape[0] > 0:
            out[-emb.shape[0] :] = emb
        return out

    def classifier_inputs(self, audio_int16: np.ndarray) -> np.ndarray:
        """Slide a 16-embedding window over a clip; return (M, 16, 96).

        Used both to build training samples and for inference.
        """
        emb = self.embeddings(audio_int16)
        if emb.shape[0] < CLASSIFIER_WINDOW_EMBEDDINGS:
            return np.zeros((0, CLASSIFIER_WINDOW_EMBEDDINGS, EMBEDDING_DIM), dtype=np.float32)
        M = emb.shape[0] - CLASSIFIER_WINDOW_EMBEDDINGS + 1
        out = np.stack([emb[i : i + CLASSIFIER_WINDOW_EMBEDDINGS] for i in range(M)])
        return out.astype(np.float32)


def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert float32 [-1, 1] to int16, clipping to range."""
    a = np.clip(audio, -1.0, 1.0)
    return (a * 32767.0).astype(np.int16)
