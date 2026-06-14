"""
Podejście 1b: Fine-tuning DistilBERT end-to-end — osobny skrypt.

Uruchomienie (zakłada że main.py był już raz uruchomiony i dane istnieją):
    python train_finetune.py

Co robi:
- Ładuje tokeny z DistilBertFTDataset (podzbiór 8k, stratyfikowany)
- Trenuje DistilBERT + klasyfikator end-to-end 
- Zapisuje wykresy treningowe i confusion matrix do plots/
- Zapisuje najlepszy model do models/distilbert_ft_best.pt
- Drukuje porównanie z zamrożonym DistilBERT (Podejście 1) jeśli dostępne

Dlaczego osobny skrypt a nie main.py:
- Fine-tuning BERT jest wolny — nie chcemy przetaczać całego potoku
- Można uruchomić wielokrotnie zmieniając hiperparametry
- Wyniki i tak trafiają do tych samych plots/ i models/
"""


import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AutoModel
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)

from dataset_ft import DistilBertFTDataset
from prepare_data import EMOTION_LABELS

# Hiperparametry 
MAX_SAMPLES  = 12000    # podzbiór 
MAX_LENGTH   = 64      # długość sekwencji tokenów
BATCH_SIZE   = 32       # mniejszy niż MLP bo BERT jest cięższy
EPOCHS       = 8        # szybki trening;
LR_BERT      = 3e-5     # standardowy LR dla fine-tuningu BERT
LR_HEAD      = 3e-4     # wyższy LR dla głowy klasyfikacyjnej
PATIENCE     = 2        # early stopping
VAL_SPLIT    = 0.15
TEST_SPLIT   = 0.15
NUM_CLASSES  = 11
NUM_WORKERS  = 2 
PLOTS_DIR    = "plots"
MODELS_DIR   = "models"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "distilbert_ft_best.pt")


# Model 

class DistilBertClassifier(nn.Module):
    """
    DistilBERT z głową klasyfikacyjną trenowany end-to-end.

    Różnica vs Podejście 1 (zamrożony):
    - Tutaj wagi DistilBERT SĄ aktualizowane podczas treningu
    - Model uczy się reprezentacji specyficznych dla emocji w tweetach
    - Wymaga tokenów (input_ids, attention_mask) zamiast gotowych embeddingów

    Architektura:
        DistilBERT → CLS token (768-dim)
        → Dropout → Linear(768, 256) → GELU → Dropout → Linear(256, num_classes)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(DISTILBERT_MODEL_NAME)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]   # CLS token
        return self.classifier(cls_token)


# Pętle treningowe 

def train_epoch_ft(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for input_ids, attention_mask, labels in loader:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels         = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch_ft(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for input_ids, attention_mask, labels in loader:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels         = labels.to(device)

        logits      = model(input_ids, attention_mask)
        loss        = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def get_predictions_ft(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for input_ids, attention_mask, labels in loader:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        preds = model(input_ids, attention_mask).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


# Wykresy 

def plot_history_ft(history: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="val")
    axes[0].set_title("DistilBERT FT — Loss")
    axes[0].set_xlabel("Epoka")
    axes[0].set_ylabel("CrossEntropy Loss")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"],   label="val")
    axes[1].set_title("DistilBERT FT — Accuracy")
    axes[1].set_xlabel("Epoka")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "history_distilbert_ft.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Wykres treningowy zapisany: {save_path}")
    plt.show()


def plot_confusion_matrix_ft(y_true, y_pred) -> None:
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMOTION_LABELS)
    fig, ax = plt.subplots(figsize=(11, 8))
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    ax.set_title("Confusion Matrix — DistilBERT Fine-tuned")
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "confusion_matrix_DistilBERT_FT.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Confusion matrix zapisana: {save_path}")
    plt.show()


# Main 

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")
    print(f"Hiperparametry: samples={MAX_SAMPLES}, epochs={EPOCHS}, "
          f"lr_bert={LR_BERT}, lr_head={LR_HEAD}, batch={BATCH_SIZE}\n")

    # Dataset i split 
    print("[1/4] Ładuję dataset (tokeny DistilBERT, podzbiór stratyfikowany)...")
    dataset = DistilBertFTDataset(
        overwrite=False,
        device=device,
        max_samples=MAX_SAMPLES,
        max_length=MAX_LENGTH,
    )

    n         = len(dataset)
    test_size = int(TEST_SPLIT * n)
    val_size  = int(VAL_SPLIT * n)
    train_size = n - val_size - test_size

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Model i optymizator 
    print("\n[2/4] Inicjalizuję model DistilBERT + klasyfikator...")
    model = DistilBertClassifier(num_classes=NUM_CLASSES).to(device)
    model.bert.enable_input_require_grads()
    model.bert.gradient_checkpointing_enable()

    # Dwie grupy parametrów z różnymi LR:
    # - wagi BERT: mały LR żeby nie zniszczyć pre-treningu
    # - głowa klasyfikacyjna: większy LR bo inicjalizowana losowo
    bert_params = list(model.bert.parameters())
    head_params = list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": bert_params, "lr": LR_BERT,  "weight_decay": 0.01},
        {"params": head_params, "lr": LR_HEAD,  "weight_decay": 1e-4},
    ])

    # Wagi klas — jak w train.py
    train_labels = dataset.labels[train_ds.indices]
    class_counts = torch.bincount(train_labels).float()
    class_weights = (class_counts.sum() / (len(class_counts) * class_counts)).to(device)
    class_weights = torch.where(torch.isinf(class_weights),
                                torch.zeros_like(class_weights), class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=1, factor=0.5
)

    # Trening z early stopping 
    print(f"\n[3/4] Trening ({EPOCHS} epoki, early stopping patience={PATIENCE})...")
    print("=" * 60)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss     = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch_ft(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = eval_epoch_ft(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoka {epoch:2d}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping po epoce {epoch}.")
                break

    print(f"\nNajlepszy model zapisany: {MODEL_SAVE_PATH}")
    print(f"Najlepsza val_loss: {best_val_loss:.4f}")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))

    #  Ewaluacja i wykresy 
    print("\n[4/4] Ewaluacja na zbiorze testowym...")
    plot_history_ft(history)

    y_true, y_pred = get_predictions_ft(model, test_loader, device)

    print("\n" + "=" * 60)
    print("Wyniki: DistilBERT Fine-tuned (end-to-end)")
    print("=" * 60)
    print(classification_report(y_true, y_pred,
                                 target_names=EMOTION_LABELS, zero_division=0))

    plot_confusion_matrix_ft(y_true, y_pred)

    acc       = accuracy_score(y_true, y_pred)
    f1_macro  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_weight = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 60)
    print("PODSUMOWANIE — DistilBERT Fine-tuned vs Frozen")
    print("=" * 60)
    print(f"  DistilBERT FT  (end-to-end): acc={acc:.3f}  f1_macro={f1_macro:.3f}")
    print(f"  DistilBERT MLP (zamrożony):  acc=0.297  f1_macro=0.215  [z main.py]")
    print(f"  GloVe+BiLSTM   (referencja): acc=0.481  f1_macro=0.412  [z main.py]")
    print(f"\n✓ Wykresy zapisane w '{PLOTS_DIR}/'.")
    print(f"✓ Model zapisany: {MODEL_SAVE_PATH}")
