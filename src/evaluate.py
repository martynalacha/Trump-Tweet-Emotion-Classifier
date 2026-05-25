"""
Ewaluacja modelu na zbiorze testowym.

Funkcje:
- get_predictions   : zebranie predykcji i prawdziwych etykiet
- evaluate_model    : raport klasyfikacji + confusion matrix
- compare_models    : porównanie obu podejść na jednym wykresie
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


CLASS_NAMES = ["Negative", "Neutral", "Positive"]


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
        for embeddings, labels in loader:
            embeddings = embeddings.to(device)
            logits = model(embeddings)
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
    Drukuje raport klasyfikacji i rysuje confusion matrix.
    """

    import os
    os.makedirs(save_dir, exist_ok=True)

    if device is None:
        device = torch.device("cpu")

    y_true, y_pred = get_predictions(model, test_loader, device)

    print(f"\n{'='*50}")
    print(f"Wyniki: {model_name}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Confusion matrix zapisana: {save_path}")
    plt.show()

    from sklearn.metrics import accuracy_score, f1_score
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }

    return results


def compare_models(
    results_dict: dict,
    save_dir: str = "plots",
) -> None:
    """
    Porównanie metryk obu modeli na słupkowym wykresie.

    results_dict: {"MLP (DistilBERT)": {...}, "MLP (SBERT)": {...}}
    """

    import os
    os.makedirs(save_dir, exist_ok=True)

    metrics = ["accuracy", "f1_macro", "f1_weighted"]
    model_names = list(results_dict.keys())
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, name in enumerate(model_names):
        values = [results_dict[name][m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=name)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(["Accuracy", "F1 Macro", "F1 Weighted"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Wartość metryki")
    ax.set_title("Porównanie modeli — Podejście 1 vs Podejście 2")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Porównanie modeli zapisane: {save_path}")
    plt.show()