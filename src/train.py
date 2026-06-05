"""
Pętla treningowa — wspólna dla wszystkich podejść (MLP i BiLSTM).

Funkcje:
- train_epoch   : jedna epoka treningu
- eval_epoch    : ewaluacja na zbiorze val/test
- train_model   : pełny trening z early stopping + zapis najlepszego modelu
- plot_history  : wykres loss i accuracy

Obsługuje:
- MLP (Podejście 1: DistilBERT, Podejście 2: SBERT) — batch = (embeddings, labels)
- BiLSTM (Podejście 3: GloVe) — batch = (sequences, lengths, labels)
"""

import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def _is_lstm_batch(batch) -> bool:
    """Sprawdza czy batch pochodzi z GloVeDataset (3 elementy: seq, lengths, labels)."""
    return len(batch) == 3


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Jedna epoka treningu. Zwraca (avg_loss, accuracy)."""

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        if _is_lstm_batch(batch):
            inputs, lengths, labels = batch
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs, lengths)
        else:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)

        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Ewaluacja na zbiorze val lub test. Zwraca (avg_loss, accuracy)."""

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            if _is_lstm_batch(batch):
                inputs, lengths, labels = batch
                inputs = inputs.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)
                logits = model(inputs, lengths)
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                logits = model(inputs)

            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    return total_loss / total, correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str = "model",
    epochs: int = 30,
    lr: float = 1e-3,
    patience: int = 5,
    device: torch.device = None,
) -> dict:
    """
    Pełny trening z early stopping.

    Parametry
    ---------
    model_name : nazwa używana do zapisu pliku (.pt)
    patience   : ile epok bez poprawy val_loss przed zatrzymaniem

    Zwraca słownik z historią treningową.
    """

    if device is None:
        device = torch.device("cpu")

    model.to(device)

    # Zliczanie klas
    subset_indices = train_loader.dataset.indices
    full_labels = train_loader.dataset.dataset.labels
    train_labels = full_labels[subset_indices]

    class_counts = torch.bincount(train_labels).to(torch.float32)
    total_samples = class_counts.sum()
    class_weights = total_samples / (len(class_counts) * class_counts)

    class_weights = torch.where(torch.isinf(class_weights), torch.zeros_like(class_weights), class_weights)
    class_weights = class_weights.to(device)

    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(MODELS_DIR, f"{model_name}_best.pt")

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    print(f"\n{'='*50}")
    print(f"Trening modelu: {model_name}")
    print(f"{'='*50}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoka {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping po epoce {epoch}.")
                break

    print(f"\nNajlepszy model zapisany: {best_model_path}")
    print(f"Najlepsza val_loss: {best_val_loss:.4f}")

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    return history


def plot_history(
    history: dict,
    model_name: str = "model",
    save_path: str = None,
) -> None:
    """Rysuje wykresy loss i accuracy z historii treningu."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].set_xlabel("Epoka")
    axes[0].set_ylabel("CrossEntropy Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title(f"{model_name} — Accuracy")
    axes[1].set_xlabel("Epoka")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Wykres zapisany: {save_path}")

    plt.show()