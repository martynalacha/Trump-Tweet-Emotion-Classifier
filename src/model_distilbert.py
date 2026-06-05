"""
Podejście 1: MLP z residual connections na embeddingach DistilBERT.

Architektura — ResidualMLP:
    Projekcja wejścia (768 → hidden_dim)
    N bloków residualnych:
        LayerNorm → Linear → GELU → Dropout → Linear → Dropout
        + skip connection (x = x + blok(x))
    Głowa klasyfikacyjna: LayerNorm → Linear(hidden_dim → num_classes)

Zalety residual connections vs zwykłe MLP:
- Gradients płyną bezpośrednio przez skip connection → łatwiejszy trening
- Model może się nauczyć "nic nie robić" w danym bloku (identity mapping)
- Pozwala na więcej warstw bez degradacji

Wejście:  768-dim CLS token z DistilBERT
Wyjście:  11 klas emocji (NRC Lexicon)
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Jeden blok residualny:
        LayerNorm → Linear → GELU → Dropout → Linear → Dropout
        y = x + blok(x)

    Używamy GELU (zamiast ReLU) — standard w architekturach transformerowych,
    łagodniejsze zerowanie ujemnych wartości.
    """

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),   # rozszerzenie do dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),   # powrót do dim
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)       # skip connection


class SentimentMLP(nn.Module):
    """
    Klasyfikator sentymentu — MLP z residual connections.

    Parametry
    ---------
    input_dim   : wymiar embeddingu wejściowego (768 dla DistilBERT)
    hidden_dim  : wymiar wewnętrzny bloków residualnych
    num_classes : liczba klas (11)
    num_blocks  : liczba bloków residualnych
    dropout     : dropout w każdym bloku
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        num_classes: int = 11,
        num_blocks: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Projekcja wejścia do hidden_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Stosy bloków residualnych
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )

        # Głowa klasyfikacyjna
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