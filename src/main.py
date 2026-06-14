"""
Główny skrypt projektu — analiza sentymentu tweetów Trumpa.

WSZYSTKIE modele pracują na tych samych 11 etykietach emocji (NRC Lexicon):
    0  angry        4  disgusted    8  sarcastic
    1  fearful      5  surprised    9  patriotic
    2  joyful       6  enthusiastic
    3  sad          7  trusting

Podejście 1: DistilBERT embeddingi (768-dim) + MLP
Podejście 2: SBERT embeddingi (384-dim) + MLP
Podejście 3: GloVe (200-dim) + BiLSTM  (inna rodzina niż BERT)
Podejście 4: Zero-Shot z BART          (bez trenowania, te same 11 etykiet)

"""

import os
import torch
from torch.utils.data import DataLoader, random_split

from dataset import TrumpTweetDataset, SBERTDataset, GloVeDataset, glove_collate_fn
from model_distilbert import SentimentMLP
from model_sbert import SentimentSBERT
from model_glove import SentimentBiLSTM
from model_zero_shot import run_zero_shot, evaluate_zero_shot, plot_zero_shot_examples
from prepare_data import EMOTION_LABELS
from train import train_model, plot_history
from evaluate import evaluate_model, compare_models
from shap_analysis import run_shap_deep, plot_shap_summary, plot_shap_bar, run_shap_tfidf


NUM_CLASSES = 11          # wszystkie modele używają 10 klas
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


def make_loaders(dataset, collate_fn=None):
    train_ds, val_ds, test_ds = split_dataset(dataset)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Używane urządzenie: {device}")
    print(f"Liczba klas: {NUM_CLASSES} ({', '.join(EMOTION_LABELS)})\n")

    # Dane — wszystkie korzystają z tych samych etykiet NRC
    print("[1/8] Ładuję dataset (DistilBERT embeddingi, etykiety NRC)...")
    dataset_bert = TrumpTweetDataset(overwrite=True, device=device)
    label_counts = dataset_bert.df["label"].value_counts().sort_index()
    print("  Rozkład etykiet:")
    for idx, name in enumerate(EMOTION_LABELS):
        count = label_counts.get(idx, 0)
        print(f"    {name:<15} {count}")

    print("\n[2/8] Ładuję dataset (SBERT embeddingi)...")
    dataset_sbert = SBERTDataset(overwrite=True, device=device)

    print("\n[3/8] Ładuję dataset (GloVe + tokenizacja)...")
    dataset_glove = GloVeDataset(overwrite=True, device=device)

    # Podejście 1: DistilBERT + MLP (10 klas)
    print("\n[4/8] === PODEJŚCIE 1: DistilBERT + MLP ===")
    train_loader_1, val_loader_1, test_loader_1 = make_loaders(dataset_bert)
    model_1 = SentimentMLP(input_dim=768, hidden_dim=256, num_classes=NUM_CLASSES, num_blocks=2, dropout=0.5)
    history_1 = train_model(
        model=model_1, train_loader=train_loader_1, val_loader=val_loader_1,
        model_name="distilbert_mlp", epochs=EPOCHS, lr=LR, patience=PATIENCE, device=device,
    )
    plot_history(history_1, model_name="DistilBERT MLP",
                 save_path=os.path.join(PLOTS_DIR, "history_distilbert.png"))
    results_1 = evaluate_model(model=model_1, test_loader=test_loader_1,
                               model_name="DistilBERT_MLP", device=device, save_dir=PLOTS_DIR)

    # Podejście 2: SBERT + MLP (10 klas)
    print("\n[5/8] === PODEJŚCIE 2: SBERT + MLP ===")
    train_loader_2, val_loader_2, test_loader_2 = make_loaders(dataset_sbert)
    model_2 = SentimentSBERT(input_dim=384, hidden_dim=256, num_classes=NUM_CLASSES, num_blocks=2, dropout=0.5)
    history_2 = train_model(
        model=model_2, train_loader=train_loader_2, val_loader=val_loader_2,
        model_name="sbert_mlp", epochs=EPOCHS, lr=LR, patience=PATIENCE, device=device,
    )
    plot_history(history_2, model_name="SBERT MLP",
                 save_path=os.path.join(PLOTS_DIR, "history_sbert.png"))
    results_2 = evaluate_model(model=model_2, test_loader=test_loader_2,
                               model_name="SBERT_MLP", device=device, save_dir=PLOTS_DIR)

    # Podejście 3: GloVe + BiLSTM (10 klas)
    print("\n[6/8] === PODEJŚCIE 3: GloVe + BiLSTM ===")
    train_loader_3, val_loader_3, test_loader_3 = make_loaders(
        dataset_glove, collate_fn=glove_collate_fn
    )
    model_3 = SentimentBiLSTM(
        embedding_matrix=dataset_glove.embedding_matrix,
        hidden_dim=256,
        num_classes=NUM_CLASSES,
        num_layers=2,
        dropout=0.3,
        freeze_emb=True,
    )
    history_3 = train_model(
        model=model_3, train_loader=train_loader_3, val_loader=val_loader_3,
        model_name="glove_bilstm", epochs=EPOCHS, lr=LR, patience=PATIENCE, device=device,
    )
    plot_history(history_3, model_name="GloVe BiLSTM",
                 save_path=os.path.join(PLOTS_DIR, "history_glove.png"))
    results_3 = evaluate_model(model=model_3, test_loader=test_loader_3,
                               model_name="GloVe_BiLSTM", device=device, save_dir=PLOTS_DIR)

    # Podejście 4: Zero-Shot BART (te same 11 etykiet, bez trenowania)
    print("\n[7/8] === PODEJŚCIE 4: Zero-Shot BART (11 etykiet) ===")
    tweets_all = dataset_bert.df["tweet"].fillna("").astype(str).tolist()
    labels_all = dataset_bert.df["label"].tolist()

    ZERO_SHOT_SUBSET = 2000  # usuń [:2000] dla pełnego datasetu
    tweets_subset = tweets_all[:ZERO_SHOT_SUBSET]
    labels_subset = labels_all[:ZERO_SHOT_SUBSET]

    zero_shot_results = run_zero_shot(
        tweets=tweets_subset,
        labels=EMOTION_LABELS,
        model_name="facebook/bart-large-mnli",
        batch_size=16,
        overwrite=True,
        device=device,
    )

    results_4 = evaluate_zero_shot(
        results=zero_shot_results,
        true_labels=labels_subset,
        label_names=EMOTION_LABELS,
        save_dir=PLOTS_DIR,
    )

    plot_zero_shot_examples(
        tweets=tweets_subset,
        results=zero_shot_results,
        label_names=EMOTION_LABELS,
        n=10,
        save_dir=PLOTS_DIR,
    )

    # Porównanie wszystkich 4 podejść
    print("\n[8/8] Porównanie wszystkich modeli...")
    compare_models(
        {
            "MLP (DistilBERT)": results_1,
            "MLP (SBERT)":      results_2,
            "BiLSTM (GloVe)":   results_3,
            "Zero-Shot (BART)": {k: v for k, v in results_4.items()
                                 if k != "mean_confidence"},
        },
        save_dir=PLOTS_DIR,
    )

    # Analiza SHAP
    print("\nAnaliza SHAP (Podejście 1 i 2)...")
    shap_vals_1, X_test_1, _ = run_shap_deep(
        model=model_1, train_loader=train_loader_1, test_loader=test_loader_1,
        model_name="DistilBERT_MLP", device=device,
    )
    for class_idx in range(NUM_CLASSES):
        plot_shap_summary(shap_vals_1, X_test_1, model_name="DistilBERT_MLP", class_idx=class_idx)
    plot_shap_bar(shap_vals_1, model_name="DistilBERT_MLP")

    shap_vals_2, X_test_2, _ = run_shap_deep(
        model=model_2, train_loader=train_loader_2, test_loader=test_loader_2,
        model_name="SBERT_MLP", device=device,
    )
    for class_idx in range(NUM_CLASSES):
        plot_shap_summary(shap_vals_2, X_test_2, model_name="SBERT_MLP", class_idx=class_idx)
    plot_shap_bar(shap_vals_2, model_name="SBERT_MLP")

    run_shap_tfidf(tweets_all, labels_all, model_name="baseline")

    print("\n" + "="*60)
    print("PODSUMOWANIE WYNIKÓW (11 klas emocji)")
    print("="*60)
    print(f"  Podejście 1 (DistilBERT+MLP): acc={results_1['accuracy']:.3f}  f1_macro={results_1['f1_macro']:.3f}")
    print(f"  Podejście 2 (SBERT+MLP):      acc={results_2['accuracy']:.3f}  f1_macro={results_2['f1_macro']:.3f}")
    print(f"  Podejście 3 (GloVe+BiLSTM):   acc={results_3['accuracy']:.3f}  f1_macro={results_3['f1_macro']:.3f}")
    print(f"  Podejście 4 (Zero-Shot BART):  acc={results_4['accuracy']:.3f}  f1_macro={results_4['f1_macro']:.3f}")
    print(f"\n✓ Wszystkie wyniki zapisane w '{PLOTS_DIR}/'.")