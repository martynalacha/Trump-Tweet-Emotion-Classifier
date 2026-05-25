"""
Główny skrypt projektu — analiza sentymentu tweetów Trumpa.

Podejście 1: DistilBERT embeddingi (768-dim) + MLP
Podejście 2: SBERT embeddingi (384-dim) + MLP

Uruchomienie:
    python main.py

Wyniki zapisywane są do:
    models/   — wytrenowane modele (.pt)
    plots/    — wykresy (confusion matrix, SHAP, porównanie)
    data/processed/ — embeddingi i etykiety (cache)
"""

import os
import torch
from torch.utils.data import DataLoader, random_split

from dataset import TrumpTweetDataset
from model import SentimentMLP
from model_sbert import SBERTDataset, SentimentSBERT
from train import train_model, plot_history
from evaluate import evaluate_model, compare_models
from shap_analysis import run_shap_deep, plot_shap_summary, plot_shap_bar, run_shap_tfidf


# ──────────────────────────────────────────────
# Konfiguracja
# ──────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
PATIENCE = 5
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
PLOTS_DIR = "plots"

os.makedirs(PLOTS_DIR, exist_ok=True)


def split_dataset(dataset, val_split=VAL_SPLIT, test_split=TEST_SPLIT, seed=42):
    n = len(dataset)
    test_size = int(test_split * n)
    val_size = int(val_split * n)
    train_size = n - val_size - test_size
    return random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )


def make_loaders(dataset):
    train_ds, val_ds, test_ds = split_dataset(dataset)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Używane urządzenie: {device}")

    # ──────────────────────────────────────────
    # Dane
    # ──────────────────────────────────────────
    print("\n[1/6] Ładuję dataset (DistilBERT embeddingi)...")
    dataset_bert = TrumpTweetDataset(threshold=0.25, overwrite=False, device=device)
    print(f"  Rozkład etykiet:\n{dataset_bert.df['label'].value_counts().to_string()}")

    print("\n[2/6] Ładuję dataset (SBERT embeddingi)...")
    dataset_sbert = SBERTDataset(overwrite=False, device=device)

    # ──────────────────────────────────────────
    # Podejście 1: DistilBERT + MLP
    # ──────────────────────────────────────────
    print("\n[3/6] === PODEJŚCIE 1: DistilBERT + MLP ===")
    train_loader_1, val_loader_1, test_loader_1 = make_loaders(dataset_bert)

    model_1 = SentimentMLP(input_dim=768, hidden_dim=256, num_classes=3, dropout=0.3)

    history_1 = train_model(
        model=model_1,
        train_loader=train_loader_1,
        val_loader=val_loader_1,
        model_name="distilbert_mlp",
        epochs=EPOCHS,
        lr=LR,
        patience=PATIENCE,
        device=device,
    )

    plot_history(
        history_1,
        model_name="DistilBERT MLP",
        save_path=os.path.join(PLOTS_DIR, "history_distilbert.png"),
    )

    results_1 = evaluate_model(
        model=model_1,
        test_loader=test_loader_1,
        model_name="DistilBERT_MLP",
        device=device,
        save_dir=PLOTS_DIR,
    )

    # ──────────────────────────────────────────
    # Podejście 2: SBERT + MLP
    # ──────────────────────────────────────────
    print("\n[4/6] === PODEJŚCIE 2: SBERT + MLP ===")
    train_loader_2, val_loader_2, test_loader_2 = make_loaders(dataset_sbert)

    model_2 = SentimentSBERT(input_dim=384, hidden_dim=256, num_classes=3, dropout=0.3)

    history_2 = train_model(
        model=model_2,
        train_loader=train_loader_2,
        val_loader=val_loader_2,
        model_name="sbert_mlp",
        epochs=EPOCHS,
        lr=LR,
        patience=PATIENCE,
        device=device,
    )

    plot_history(
        history_2,
        model_name="SBERT MLP",
        save_path=os.path.join(PLOTS_DIR, "history_sbert.png"),
    )

    results_2 = evaluate_model(
        model=model_2,
        test_loader=test_loader_2,
        model_name="SBERT_MLP",
        device=device,
        save_dir=PLOTS_DIR,
    )

    # ──────────────────────────────────────────
    # Porównanie modeli
    # ──────────────────────────────────────────
    print("\n[5/6] Porównanie modeli...")
    compare_models(
        {
            "MLP (DistilBERT)": results_1,
            "MLP (SBERT)": results_2,
        },
        save_dir=PLOTS_DIR,
    )

    # ──────────────────────────────────────────
    # Analiza SHAP
    # ──────────────────────────────────────────
    print("\n[6/6] Analiza SHAP...")

    # SHAP dla Podejście 1
    shap_vals_1, X_test_1, y_test_1 = run_shap_deep(
        model=model_1,
        train_loader=train_loader_1,
        test_loader=test_loader_1,
        model_name="DistilBERT_MLP",
        device=device,
    )
    for class_idx in [0, 1, 2]:
        plot_shap_summary(shap_vals_1, X_test_1, model_name="DistilBERT_MLP", class_idx=class_idx)
    plot_shap_bar(shap_vals_1, model_name="DistilBERT_MLP")

    # SHAP dla Podejście 2
    shap_vals_2, X_test_2, y_test_2 = run_shap_deep(
        model=model_2,
        train_loader=train_loader_2,
        test_loader=test_loader_2,
        model_name="SBERT_MLP",
        device=device,
    )
    for class_idx in [0, 1, 2]:
        plot_shap_summary(shap_vals_2, X_test_2, model_name="SBERT_MLP", class_idx=class_idx)
    plot_shap_bar(shap_vals_2, model_name="SBERT_MLP")

    # SHAP TF-IDF (baseline — interpretacja na poziomie słów)
    tweets = dataset_bert.df["tweet"].fillna("").astype(str).tolist()
    labels = dataset_bert.df["label"].tolist()
    run_shap_tfidf(tweets, labels, model_name="baseline")

    print("\n✓ Gotowe! Wszystkie wyniki zapisane w folderze 'plots/'.")