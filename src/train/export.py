"""Export a trained PyTorch classifier to ONNX.

Output shape (1, 16, 96) float32 input, (1, 1) float32 output. Matches the
input openwakeword expects when loading custom models via wakeword_models=[...].
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from src.data.features import CLASSIFIER_WINDOW_EMBEDDINGS, EMBEDDING_DIM

logger = logging.getLogger(__name__)


def export_onnx(model: torch.nn.Module, out_path: Path, opset: int = 17) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    example = torch.rand(1, CLASSIFIER_WINDOW_EMBEDDINGS, EMBEDDING_DIM, dtype=torch.float32)
    logger.info("Exporting ONNX -> %s (opset=%d)", out_path, opset)
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
