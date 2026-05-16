"""Export a trained PyTorch classifier to ONNX.

Output shape (1, 16, 96) float32 input, (1, 1) float32 output. Matches the
input openwakeword expects when loading custom models via wakeword_models=[...].
"""
from __future__ import annotations

import logging
import math
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

    NOTE: Adds Clip/Add/Sub/Mul/Div/Constant ops to the ONNX graph. Some
    JS ONNX runtimes (onnxruntime-web wasm build, certain browser embeds)
    fail to load graphs that include this wrapper. Prefer
    ``apply_bias_shift_calibration`` below which produces an ONNX graph
    with the EXACT SAME ops as the uncalibrated model.
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


def apply_bias_shift_calibration(model: nn.Module, threshold: float) -> nn.Module:
    """Bake the threshold calibration into the model's final Linear bias.

    Mathematically equivalent to wrapping in ThresholdCalibratedModel:
        sigmoid(W x + b)              <- raw model
        sigmoid(W x + b - logit(t))   <- after this fn, with t=threshold
    A raw_score == threshold now produces calibrated output 0.5, exactly
    like the wrapper, but the ONNX graph has zero added ops. Required for
    onnxruntime-web compatibility (the wrapper produces Clip+Div+broadcast
    patterns that some JS builds reject).
    """
    threshold = float(max(1e-6, min(1.0 - 1e-6, threshold)))
    logit_t = math.log(threshold / (1.0 - threshold))
    last_linear: nn.Linear | None = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is None:
        raise RuntimeError(
            "Model has no Linear layer; cannot bake bias-shift calibration"
        )
    with torch.no_grad():
        # Subtract logit(threshold) from the bias. This shifts the pre-sigmoid
        # logit by exactly the same amount the wrapper would have applied.
        if last_linear.bias is None:
            last_linear.bias = nn.Parameter(
                torch.full((last_linear.out_features,), -logit_t, dtype=last_linear.weight.dtype)
            )
        else:
            last_linear.bias.data = last_linear.bias.data - logit_t
    return model


def export_onnx(
    model: torch.nn.Module,
    out_path: Path,
    opset: int = 17,
    score_threshold: float = 0.5,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    if abs(float(score_threshold) - 0.5) > 1e-6:
        # Use bias-shift calibration instead of the wrapper. Same math, but
        # the ONNX graph stays free of Clip/Div/broadcast ops that some
        # JS ONNX runtimes (notably onnxruntime-web's wasm build in some
        # Voice Satellite-style embeddings) refuse to load.
        model = apply_bias_shift_calibration(model, score_threshold).eval().cpu()
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
