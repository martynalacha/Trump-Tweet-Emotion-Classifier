"""
Pętla treningowa — wspólna dla obu podejść (MLP na DistilBERT i MLP na SBERT).

Funkcje:
- train_epoch   : jedna epoka treningu
- eval_epoch    : ewaluacja na zbiorze val/test
- train_model   : pełny trening z early stopping + zapis najlepszego modelu
- plot_history  : wykres loss i accuracy
"""

import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


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

    for embeddings, labels in loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(embeddings)
        loss = criterion(logits, labels)
        loss.backward()
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
        for embeddings, labels in loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            logits = model(embeddings)
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

    criterion = nn.CrossEntropyLoss()
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

    # Załaduj najlepszy model
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    return history


def plot_history(
    history: dict,
    model_name: str = "model",
    save_path: str = None,
) -> None:
    """Rysuje wykresy loss i accuracy z historii treningu."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].set_xlabel("Epoka")
    axes[0].set_ylabel("CrossEntropy Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
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