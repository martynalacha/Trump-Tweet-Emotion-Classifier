"""
Podejście 2: MLP z residual connections na embeddingach SBERT.

Identyczna architektura jak model.py (SentimentMLP), ale input_dim=384
(all-MiniLM-L6-v2 zamiast DistilBERT 768-dim).

SBERT (Sentence-BERT) różni się od DistilBERT tym że używa mean pooling
i był fine-tunowany na zadaniach podobieństwa semantycznego (NLI/STS),
przez co jego embeddingi lepiej oddają znaczenie całego zdania.

Dataset dla tego podejścia: SBERTDataset w dataset.py
"""

import torch
import torch.nn as nn
from model_distilbert import ResidualBlock


class SentimentSBERT(nn.Module):
    """
    MLP z residual connections na embeddingach SBERT (384-dim).

    Parametry
    ---------
    input_dim  : wymiar embeddingu (384 dla all-MiniLM-L6-v2)
    hidden_dim : wymiar wewnętrzny bloków residualnych
    num_classes: liczba klas emocji (11)
    num_blocks : liczba bloków residualnych
    dropout    : dropout w każdym bloku
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 512,
        num_classes: int = 11,
        num_blocks: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.residual_blocks(x)
        return self.classifier(x)