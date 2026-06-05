"""
Ewaluacja modelu na zbiorze testowym — 10 klas emocji.

Funkcje:
- get_predictions   : zebranie predykcji i prawdziwych etykiet
- evaluate_model    : raport klasyfikacji + confusion matrix
- compare_models    : porównanie podejść na jednym wykresie
"""

import os

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)

from prepare_data import EMOTION_LABELS


def _is_lstm_batch(batch) -> bool:
    return len(batch) == 3


def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    """Zbiera predykcje i prawdziwe etykiety ze zbioru testowego."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            if _is_lstm_batch(batch):
                inputs, lengths, labels = batch
                inputs = inputs.to(device)
                lengths = lengths.to(device)
                logits = model(inputs, lengths)
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                logits = model(inputs)

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    model_name: str = "model",
    device: torch.device = None,
    save_dir: str = "plots",
) -> dict:
    """
    Ewaluacja na zbiorze testowym.
    Drukuje raport klasyfikacji i rysuje confusion matrix (10x10).
    """
    os.makedirs(save_dir, exist_ok=True)

    if device is None:
        device = torch.device("cpu")

    y_true, y_pred = get_predictions(model, test_loader, device)

    print(f"\n{'='*60}")
    print(f"Wyniki: {model_name}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=EMOTION_LABELS, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMOTION_LABELS)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Confusion matrix zapisana: {save_path}")
    plt.show()

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    return results


def compare_models(
    results_dict: dict,
    save_dir: str = "plots",
) -> None:
    """
    Porównanie metryk modeli na słupkowym wykresie.
    results_dict: {"Nazwa modelu": {"accuracy": ..., "f1_macro": ..., "f1_weighted": ...}}
    """
    os.makedirs(save_dir, exist_ok=True)

    metrics = ["accuracy", "f1_macro", "f1_weighted"]
    metric_labels = ["Accuracy", "F1 Macro", "F1 Weighted"]
    model_names = list(results_dict.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, name in enumerate(model_names):
        values = [results_dict[name].get(m, 0) for m in metrics]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width * 0.9, label=name,
                      color=colors[i % len(colors)])
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Wartość metryki")
    ax.set_title("Porównanie modeli — 11 klas emocji")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Porównanie modeli zapisane: {save_path}")
    plt.show()