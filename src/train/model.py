"""Classifier head trained on top of speech_embedding output.

Mirrors openwakeword/train.py:Model architecture so the resulting ONNX is
compatible with `openwakeword.Model(wakeword_models=[path])` runtime.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.data.features import CLASSIFIER_WINDOW_EMBEDDINGS, EMBEDDING_DIM


class WakeWordDNN(nn.Module):
    """Flatten -> Linear(1536, H) -> LN -> ReLU -> [Linear(H,H) -> LN -> ReLU]*n -> Linear(H,1) -> Sigmoid."""

    def __init__(self, layer_dim: int = 128, n_blocks: int = 1) -> None:
        super().__init__()
        in_dim = CLASSIFIER_WINDOW_EMBEDDINGS * EMBEDDING_DIM  # 16 * 96 = 1536

        layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(in_dim, layer_dim),
            nn.LayerNorm(layer_dim),
            nn.ReLU(),
        ]
        for _ in range(n_blocks):
            layers += [
                nn.Linear(layer_dim, layer_dim),
                nn.LayerNorm(layer_dim),
                nn.ReLU(),
            ]
        layers += [nn.Linear(layer_dim, 1), nn.Sigmoid()]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 16, 96)
        return self.net(x)


class WakeWordRNN(nn.Module):
    """Bi-LSTM alternative."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(nn.Linear(hidden * 2, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def build_model(model_type: str, layer_dim: int, n_blocks: int) -> nn.Module:
    if model_type == "rnn":
        return WakeWordRNN(hidden=max(32, layer_dim // 2))
    return WakeWordDNN(layer_dim=layer_dim, n_blocks=n_blocks)
