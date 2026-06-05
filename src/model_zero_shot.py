"""
Podejście 4: Zero-Shot Classification z facebook/bart-large-mnli.

Zero-shot = model NIE jest trenowany na danych Trumpa. Podajemy mu listę
etykiet (tych samych 11 co w modelach 1-3) i tweet, a on oblicza
prawdopodobieństwo każdej etykiety przez NLI:
    "Czy tweet X wyraża sentyment: angry / joyful / sarcastic ...?"

Dzięki temu wszystkie 4 podejścia porównujemy na tych samych etykietach.

Model: facebook/bart-large-mnli (~1.6GB)
"""

import os
import json
from typing import List, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from transformers import pipeline

PLOTS_DIR = "plots"
PROCESSED_DIR = "data/processed"
ZERO_SHOT_RESULTS_PATH = os.path.join(PROCESSED_DIR, "zero_shot_results.json")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def run_zero_shot(
    tweets: List[str],
    labels: List[str],
    model_name: str = "facebook/bart-large-mnli",
    batch_size: int = 16,
    overwrite: bool = False,
    device=None,
) -> List[Dict]:
    """
    Klasyfikacja zero-shot dla listy tweetów.

    Parametry
    ---------
    tweets     : lista tweetów
    labels     : lista etykiet kandydatów (np. EMOTION_LABELS z prepare_data.py)
    model_name : model HuggingFace NLI
    overwrite  : czy nadpisać cache

    Zwraca listę słowników:
        {"labels": [...], "scores": [...], "top_label": "...", "top_score": ...}
    """
    if not overwrite and os.path.exists(ZERO_SHOT_RESULTS_PATH):
        print("Loading zero-shot results from cache...")
        with open(ZERO_SHOT_RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"Loading zero-shot model: {model_name}...")
    device_id = 0 if (device is not None and str(device) == "cuda") else -1

    # 1. Define batch_size at pipeline initialization to enable internal GPU batching
    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device_id,
        batch_size=batch_size
    )

    active_labels = [l for l in labels if l != "neutral"]
    results = []
    threshold = 0.4 

    # 2. Use a generator to stream data into the pipeline efficiently
    def tweet_generator():
        for tweet in tweets:
            yield tweet

    print(f"Classifying {len(tweets)} tweets using native GPU batching...")

    # 3. Pass the generator directly; the pipeline will internally manage DataLoader batches
    outputs = classifier(
        tweet_generator(),
        candidate_labels=active_labels,
        multi_label=True,
        hypothesis_template="This tweet expresses the emotion of {}."
    )

    # 4. Iterate over the output generator with tqdm for progress tracking
    for out in tqdm(outputs, total=len(tweets)):
        label_score_map = dict(zip(out["labels"], out["scores"]))
        
        top_active_label = out["labels"][0]
        top_active_score = out["scores"][0]
        
        full_scores = {l: label_score_map[l] for l in active_labels}
        
        if top_active_score < threshold:
            full_scores["neutral"] = 1.0 - top_active_score
            final_label = "neutral"
            final_score = full_scores["neutral"]
        else:
            full_scores["neutral"] = 0.0
            final_label = top_active_label
            final_score = top_active_score
            
        sorted_items = sorted(full_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_labels = [item[0] for item in sorted_items]
        sorted_scores = [item[1] for item in sorted_items]

        results.append({
            "labels": sorted_labels,
            "scores": sorted_scores,
            "top_label": final_label,
            "top_score": final_score,
        })

    with open(ZERO_SHOT_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {ZERO_SHOT_RESULTS_PATH}")

    return results


def evaluate_zero_shot(
    results: List[Dict],
    true_labels: List[int],
    label_names: List[str],
    save_dir: str = PLOTS_DIR,
) -> Dict:
    """
    Ewaluacja zero-shot na tych samych etykietach co modele 1-3.

    true_labels  : indeksy etykiet NRC 
    label_names  : EMOTION_LABELS 
    """
    from sklearn.metrics import (
        classification_report, confusion_matrix, ConfusionMatrixDisplay,
        accuracy_score, f1_score,
    )

    os.makedirs(save_dir, exist_ok=True)

    # Mapuj top_label (string) -> indeks (int)
    label_to_idx = {name: idx for idx, name in enumerate(label_names)}
    top_labels_str = [r["top_label"] for r in results]
    y_pred = np.array([label_to_idx.get(l, 0) for l in top_labels_str])
    y_true = np.array(true_labels)

    # ── 1. Raport klasyfikacji ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Wyniki Zero-Shot (BART) — 11 klas emocji")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))

    # ── 2. Confusion matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(11, 8))
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    ax.set_title("Confusion Matrix — Zero-Shot (BART)")
    plt.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrix_ZeroShot.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    print(f"Confusion matrix zapisana: {cm_path}")
    plt.show()

    # ── 3. Rozkład etykiet zero-shot ──────────────────────────────────────────
    label_counts = {name: top_labels_str.count(name) for name in label_names}
    sorted_names = sorted(label_counts, key=label_counts.get, reverse=True)
    counts = [label_counts[n] for n in sorted_names]

    colors = list(mcolors.TABLEAU_COLORS.values())[:len(label_names)]
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(sorted_names, counts, color=colors)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("Etykieta emocji")
    ax.set_ylabel("Liczba tweetów")
    ax.set_title("Zero-Shot: Rozkład etykiet emocji (Trump Tweets)")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    dist_path = os.path.join(save_dir, "zero_shot_label_distribution.png")
    plt.savefig(dist_path, dpi=150, bbox_inches="tight")
    print(f"Rozkład etykiet zapisany: {dist_path}")
    plt.show()

    # ── 4. Rozkład pewności predykcji ─────────────────────────────────────────
    top_scores = [r["top_score"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(top_scores, bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(top_scores), color="red", linestyle="--",
               label=f"Średnia: {np.mean(top_scores):.3f}")
    ax.set_xlabel("Pewność predykcji (top score)")
    ax.set_ylabel("Liczba tweetów")
    ax.set_title("Zero-Shot: Rozkład pewności predykcji")
    ax.legend()
    plt.tight_layout()
    score_path = os.path.join(save_dir, "zero_shot_confidence.png")
    plt.savefig(score_path, dpi=150, bbox_inches="tight")
    print(f"Rozkład pewności zapisany: {score_path}")
    plt.show()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mean_confidence": float(np.mean(top_scores)),
    }
    return metrics


def plot_zero_shot_examples(
    tweets: List[str],
    results: List[Dict],
    label_names: List[str],
    n: int = 5,
    save_dir: str = PLOTS_DIR,
) -> None:
    """
    Wyświetla N przykładowych tweetów z rozkładem prawdopodobieństw
    dla wszystkich etykiet.
    """
    os.makedirs(save_dir, exist_ok=True)
    indices = np.random.choice(len(tweets), size=min(n, len(tweets)), replace=False)
    colors = list(mcolors.TABLEAU_COLORS.values())

    fig, axes = plt.subplots(n, 1, figsize=(12, n * 2.5))
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        result = results[idx]
        tweet_text = tweets[idx][:100] + "..." if len(tweets[idx]) > 100 else tweets[idx]

        label_score_map = dict(zip(result["labels"], result["scores"]))
        scores = [label_score_map.get(l, 0.0) for l in label_names]
        bar_colors = [colors[i % len(colors)] for i in range(len(label_names))]

        ax.barh(label_names, scores, color=bar_colors, alpha=0.8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Prawdopodobieństwo")
        ax.set_title(f'"{tweet_text}"', fontsize=9, pad=3)
        ax.axvline(0.1, color="gray", linestyle=":", alpha=0.5)

    plt.suptitle("Zero-Shot: Przykładowe predykcje emocji", fontsize=13, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "zero_shot_examples.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Przykłady zapisane: {save_path}")
    plt.show()