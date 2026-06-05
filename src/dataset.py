"""
Klasy Dataset dla wszystkich trzech podejść opartych na trenowaniu.
Podejście 1: TrumpTweetDataset  - embeddingi DistilBERT (768-dim)
Podejście 2: SBERTDataset       - embeddingi SBERT all-MiniLM-L6-v2 (384-dim)
Podejście 3: GloVeDataset       - tokenizowane sekwencje GloVe (200-dim)

Wszystkie trzy korzystają z tych samych etykiet NRC (11 klas emocji)
zapisanych w data/processed/ i  zapisanych w data/processed/ przez prepare_data.py
"""

import os 
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from prepare_data import (
    prepare_dataset,
    DISTILBERT_EMBEDDINGS_PATH,
    LABELS_PATH,
    METADATA_PATH,
    SBERT_EMBEDDINGS_PATH,
    GLOVE_EMBEDDINGS_PATH,
    GLOVE_SEQUENCES_PATH,
    GLOVE_VOCAB_PATH,
    PAD_IDX,
)

class TrumpTweetDataset(Dataset):

    """
    Podejście 1 - embeddingi CLS token z DistilBERT (768-dim).
    Etykiety: 11 klas emocji NRC.
    """

    def __init__(self, overwrite: bool = False, device=None):
        super().__init__()
        prepare_dataset(overwrite, device=device)

        self.embeddings = torch.load(DISTILBERT_EMBEDDINGS_PATH, weights_only=False)
        self.labels = torch.load(LABELS_PATH, weights_only=False)
        self.df = pd.read_csv(METADATA_PATH)

    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


class SBERTDataset(Dataset):
    """
    Podejście 2 - embeddingi SBERT all-MiniLM -L6-v2 (384-dim).
    Etykiety: te same 11 klas NRC.
    """

    def __init__(self, overwrite: bool = False, device=None):
        super().__init__()
        prepare_dataset(overwrite=overwrite, device=device)

        self.embeddings = torch.load(SBERT_EMBEDDINGS_PATH, weights_only=False)
        self.labels = torch.load(LABELS_PATH, weights_only=False)
        self.df = pd.read_csv(METADATA_PATH)

    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]
    

class GloVeDataset(Dataset):
    """
    Podejście 3 — tokenizowane sekwencje indeksów GloVe (200-dim).
    Etykiety: te same 11 klas NRC.

    Wymaga glove_collate_fn jako collate_fn w DataLoader (padding sekwencji).
    """

    def __init__(self, overwrite: bool = False, device=None):
        super().__init__()
        prepare_dataset(overwrite=overwrite, device=device)
        
        # Ładowanie struktur danych GloVe zapisanych w data/processed/ przez prepare_data.py
        self.sequences        = torch.load(GLOVE_SEQUENCES_PATH, weights_only=False)
        self.labels           = torch.load(LABELS_PATH, weights_only=False)
        self.embedding_matrix = torch.load(GLOVE_EMBEDDINGS_PATH, weights_only=False)
        self.vocab            = torch.load(GLOVE_VOCAB_PATH, weights_only=False)
        self.df               = pd.read_csv(METADATA_PATH)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]   


def glove_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function dla GloVeDataset.
    Wyrównuje sekwencje o zmiennych długościach w danej paczce (barch) za pomocą paddingu.
    Zwraca wyrównany tensor, oryginalne długości oraz etykiety.
    """

    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=PAD_IDX)
    labels = torch.stack(labels)
    return padded, lengths, labels