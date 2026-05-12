"""Export a trained PyTorch classifier to ONNX.

Output shape (1, 16, 96) float32 input, (1, 1) float32 output. Matches the
input openwakeword expects when loading custom models via wakeword_models=[...].
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from src.data.features import CLASSIFIER_WINDOW_EMBEDDINGS, EMBEDDING_DIM

logger = logging.getLogger(__name__)


class ThresholdCalibratedModel(nn.Module):
    """Map a trained probability threshold to ONNX output 0.5.

    openWakeWord runtimes commonly trigger at score 0.5. The classifier is
    trained as a probability head, but the validated operating point may be
    much higher than 0.5. This wrapper preserves ordering while shifting the
    chosen threshold to 0.5:

        calibrated = sigmoid(logit(raw_score) - logit(threshold))
    """

    def __init__(self, model: nn.Module, threshold: float) -> None:
        super().__init__()
        self.model = model
        threshold = float(max(1e-6, min(1.0 - 1e-6, threshold)))
        self.register_buffer(
            "threshold",
            torch.tensor(threshold, dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(self.model(x), 1e-6, 1.0 - 1e-6)
        t = self.threshold
        numerator = p * (1.0 - t)
        denominator = numerator + (1.0 - p) * t
        return numerator / torch.clamp(denominator, min=1e-12)


def export_onnx(
    model: torch.nn.Module,
    out_path: Path,
    opset: int = 17,
    score_threshold: float = 0.5,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    if abs(float(score_threshold) - 0.5) > 1e-6:
        model = ThresholdCalibratedModel(model, score_threshold).eval().cpu()
    example = torch.rand(1, CLASSIFIER_WINDOW_EMBEDDINGS, EMBEDDING_DIM, dtype=torch.float32)
    logger.info(
        "Exporting ONNX -> %s (opset=%d, score_threshold=%.6f)",
        out_path,
        opset,
        score_threshold,
    )
    # Stay on the legacy exporter for opset compatibility with onnxruntime <1.16.
    # `dynamo=True` would force opset 18+ and a different graph layout.
    torch.onnx.export(
        model,
        example,
        str(out_path),
        input_names=["onnx::Flatten_0"],
        output_names=["sigmoid"],
        opset_version=opset,
        dynamic_axes=None,
        do_constant_folding=True,
    )
    return out_path
