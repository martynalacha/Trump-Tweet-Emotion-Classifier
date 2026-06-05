"""
Dataset dla Podejścia 1b: Fine-tuning DistilBERT end-to-end.

Zamiast wczytywać gotowe embeddingi (jak TrumpTweetDataset),
zwraca surowe tokeny (input_ids + attention_mask) — DistilBERT
jest trenowany razem z klasyfikatorem.

Ograniczenia szybkości:
- max_samples: podzbiór danych (domyślnie 8000)
- max_length: 64 tokeny (zamiast 512)
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from prepare_data import (
    prepare_dataset,
    METADATA_PATH,
    LABELS_PATH,
)

DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 64


class DistilBertFTDataset(Dataset):
    """
    Dataset zwracający tokeny DistilBERT do fine-tuningu.

    Parametry
    ---------
    max_samples : ile próbek wziąć (None = wszystkie)
    max_length  : maksymalna długość sekwencji tokenów
    """

    def __init__(
        self,
        overwrite: bool = False,
        device=None,
        max_samples: int = 8000,
        max_length: int = MAX_LENGTH,
    ):
        super().__init__()
        # Upewniamy się że etykiety i metadane istnieją na dysku
        prepare_dataset(overwrite=overwrite, device=device)

        self.labels = torch.load(LABELS_PATH, weights_only=False)
        self.df = pd.read_csv(METADATA_PATH)

        # Przycinamy do max_samples — stratyfikowane żeby zachować rozkład klas
        if max_samples is not None and max_samples < len(self.labels):
            from sklearn.model_selection import train_test_split
            import numpy as np
            indices = np.arange(len(self.labels))
            labels_np = self.labels.numpy()
            # train_test_split z stratify daje reprezentatywny podzbiór
            subset_idx, _ = train_test_split(
                indices,
                train_size=max_samples,
                stratify=labels_np,
                random_state=42,
            )
            subset_idx = sorted(subset_idx)
            self.labels = self.labels[subset_idx]
            self.df = self.df.iloc[subset_idx].reset_index(drop=True)

        self.tweets = self.df["tweet"].fillna("").astype(str).tolist()
        self.max_length = max_length

        print(f"Ładuję tokenizer DistilBERT: {DISTILBERT_MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_NAME)

        print(f"Tokenizuję {len(self.tweets)} tweetów (max_length={max_length})...")
        encodings = self.tokenizer(
            self.tweets,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids      = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

        print(f"  input_ids shape: {self.input_ids.shape}")
        print(f"  Rozkład klas (podzbiór): {torch.bincount(self.labels).tolist()}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.labels[idx],
        )
