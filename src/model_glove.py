"""
Podejście 3: BiLSTM z mechanizmem uwagi (Attention) na embeddingach GloVe.

Architektura:
    Embedding (GloVe twitter-200, opcjonalnie zamrożone)
    → BiLSTM (2 warstwy, bidirectional)
    → Additive Attention (model uczy się które tokeny są ważne)
    → Weighted sum ukrytych stanów
    → LayerNorm → Dropout → Linear → GELU → Dropout → Linear(num_classes)

Dlaczego Attention ma sens przy BiLSTM:
    BiLSTM zwraca ukryty stan dla każdego tokenu. Bez attention bierzemy
    tylko ostatni stan — tracimy informację ze środka zdania. Attention
    liczy kontekst = Σ α_t * h_t gdzie α_t to wyuczone wagi ważności
    każdego tokenu — model skupia się np. na "great", "disaster", "lol".

Zupełnie inna rodzina niż BERT:
    GloVe  — statyczne embeddingi (co-occurrence matrix, brak kontekstu)
    BiLSTM — rekurencyjna sieć (przetwarza sekwencję krok po kroku)

Dataset dla tego podejścia: GloVeDataset w dataset.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD_IDX = 0


class AdditiveAttention(nn.Module):
    """
    Additive Attention (Bahdanau-style).

    Dla każdego tokenu t liczy wagę:
        e_t = v^T * tanh(W * h_t + b)
        α   = softmax(e)
    Następnie zwraca ważoną sumę: kontekst = Σ α_t * h_t
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention      = nn.Linear(hidden_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (batch, seq_len, hidden_dim)
        mask: torch.Tensor = None,    # (batch, seq_len) True = PAD
    ) -> torch.Tensor:
        e = self.context_vector(torch.tanh(self.attention(hidden_states)))
        e = e.squeeze(-1)  # (batch, seq_len)
        if mask is not None:
            e = e.masked_fill(mask, -1e9)
        alpha   = F.softmax(e, dim=1)                                # (batch, seq_len)
        context = (alpha.unsqueeze(-1) * hidden_states).sum(dim=1)  # (batch, hidden_dim)
        return context, alpha


class SentimentBiLSTM(nn.Module):
    """
    BiLSTM + Additive Attention klasyfikator emocji na embeddingach GloVe.

    Parametry
    ---------
    embedding_matrix : macierz GloVe (vocab_size, embedding_dim) z GloVeDataset
    hidden_dim       : wymiar ukryty LSTM (na jeden kierunek)
    num_classes      : liczba klas emocji (11)
    num_layers       : liczba warstw LSTM
    dropout          : dropout
    freeze_emb       : czy zamrozić wagi GloVe podczas treningu
    """

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        hidden_dim: int = 256,
        num_classes: int = 11,
        num_layers: int = 2,
        dropout: float = 0.3,
        freeze_emb: bool = True,
    ):
        super().__init__()

        vocab_size, embedding_dim = embedding_matrix.shape

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.embedding.weight = nn.Parameter(
            embedding_matrix, requires_grad=not freeze_emb
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * 2  # bidirectional → *2

        self.attention = AdditiveAttention(lstm_out_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,              # (batch, seq_len) — indeksy tokenów
        lengths: torch.Tensor = None, # (batch,) — prawdziwe długości
    ) -> torch.Tensor:

        embedded = self.embedding(x)  # (batch, seq_len, emb_dim)

        if lengths is not None:
            packed   = pack_padded_sequence(embedded, lengths.cpu(),
                                            batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(embedded)

        pad_mask          = (x == PAD_IDX)
        context, _        = self.attention(lstm_out, mask=pad_mask)
        return self.classifier(context)