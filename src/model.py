"""
Podejście 1: MLP na embeddingach z DistilBERT.

Architektura: Linear -> BatchNorm -> ReLU -> Dropout x2 -> Linear (wyjście)
Wejście: 768-wymiarowy embedding CLS z DistilBERT
Wyjście: 3 klasy (0=negative, 1=neutral, 2=positive)
"""

import torch
import torch.nn as nn


class SentimentMLP(nn.Module):

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)