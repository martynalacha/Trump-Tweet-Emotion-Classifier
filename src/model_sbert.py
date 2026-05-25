"""
Podejście 2: SBERT (all-MiniLM-L6-v2) embeddingi + MLP.

Różnica vs Podejście 1:
- DistilBERT generuje embeddingi ogólnego przeznaczenia (768-dim)
- SBERT (sentence-transformers) generuje embeddingi zoptymalizowane
  pod kątem semantycznego podobieństwa zdań (384-dim dla MiniLM)

Ten plik zawiera:
- SBERTEmbedder  : generowanie i cache'owanie embeddingów SBERT
- SBERTDataset   : Dataset PyTorch dla embeddingów SBERT
- SentimentSBERT : MLP klasyfikator (taka sama architektura jak model.py,
                   ale mniejszy input_dim=384)
"""

import os

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from prepare_data import (
    PROCESSED_DIR,
    LABELS_PATH,
    METADATA_PATH,
    prepare_dataset,
)

SBERT_EMBEDDINGS_PATH = os.path.join(PROCESSED_DIR, "sbert_embeddings.pt")
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"


def prepare_sbert_embeddings(
    overwrite: bool = False,
    batch_size: int = 64,
    device=None,
) -> None:
    """
    Generuje embeddingi SBERT dla tweetów i zapisuje je na dysk.
    Etykiety są współdzielone z Podejściem 1 (VADER labels).
    """

    if not overwrite and os.path.exists(SBERT_EMBEDDINGS_PATH):
        print("SBERT embeddingi już istnieją, pomijam generowanie.")
        return


    if not os.path.exists(METADATA_PATH):
        print("Brak metadata — uruchamiam prepare_dataset (DistilBERT)...")
        prepare_dataset(overwrite=False, device=device)

    df = pd.read_csv(METADATA_PATH)
    tweets = df["tweet"].fillna("").astype(str).tolist()

    print(f"Ładuję model SBERT: {SBERT_MODEL_NAME}...")
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device=str(device) if device else "cpu")

    print("Generuję embeddingi SBERT...")
    all_embeddings = []

    for i in tqdm(range(0, len(tweets), batch_size)):
        batch = tweets[i : i + batch_size]
        embeddings = sbert_model.encode(
            batch,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        all_embeddings.append(embeddings.cpu())

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    torch.save(embeddings_tensor, SBERT_EMBEDDINGS_PATH)
    print(f"Zapisano SBERT embeddingi: {embeddings_tensor.shape}")


class SBERTDataset(Dataset):
    """Dataset korzystający z embeddingów SBERT (384-dim)."""

    def __init__(
        self,
        overwrite: bool = False,
        device=None,
    ):
        super().__init__()

        prepare_sbert_embeddings(overwrite=overwrite, device=device)

        self.embeddings = torch.load(SBERT_EMBEDDINGS_PATH)
        self.labels = torch.load(LABELS_PATH)
        self.df = pd.read_csv(METADATA_PATH)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class SentimentSBERT(nn.Module):
    """
    MLP na embeddingach SBERT.
    Identyczna architektura jak SentimentMLP, ale input_dim=384.
    """

    def __init__(
        self,
        input_dim: int = 384,
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