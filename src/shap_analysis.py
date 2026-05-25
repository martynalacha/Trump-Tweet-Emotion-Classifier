"""
Analiza SHAP dla obu podejść.

Dla MLP na embeddingach (DistilBERT i SBERT) SHAP działa na wymiarach
przestrzeni embeddingów — pokazuje, które wymiary (cechy ukryte) najbardziej
wpływają na predykcję danej klasy.

Dodatkowa analiza dla SBERT: można zbadać, które tweety (przykłady) mają
największy wpływ — SHAP DeepExplainer lub KernelExplainer.

Używamy shap.DeepExplainer (PyTorch) dla szybkości.
Dla porównania z "interpretowalnymi cechami" (TF-IDF) używamy
shap.LinearExplainer na prostym baseline modelu LogReg+TF-IDF.

Funkcje:
- run_shap_deep         : SHAP dla modeli MLP (DistilBERT / SBERT embeddingi)
- run_shap_tfidf        : SHAP dla baseline LogReg + TF-IDF (słowa jako cechy)
- plot_shap_summary     : summary plot
- plot_shap_bar         : bar plot ważności cech
"""

import os

import numpy as np
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# SHAP dla modeli MLP (embeddingi)

def _loader_to_numpy(loader: DataLoader, max_samples: int = 2000):
    """Konwertuje DataLoader -> numpy array (max_samples próbek)."""
    all_emb = []
    all_lab = []
    for emb, lab in loader:
        all_emb.append(emb.numpy())
        all_lab.append(lab.numpy())
        if sum(len(x) for x in all_emb) >= max_samples:
            break
    X = np.concatenate(all_emb, axis=0)[:max_samples]
    y = np.concatenate(all_lab, axis=0)[:max_samples]
    return X, y


def run_shap_deep(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    model_name: str = "model",
    device: torch.device = None,
    n_background: int = 200,
    n_explain: int = 100,
) -> shap.Explanation:
    """
    Oblicza wartości SHAP dla MLP za pomocą DeepExplainer.

    Parametry
    ---------
    n_background : liczba próbek tła (referencyjne)
    n_explain    : liczba próbek do wyjaśnienia

    Zwraca obiekt shap.Explanation (shap_values).
    """

    if device is None:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    X_train, _ = _loader_to_numpy(train_loader, max_samples=n_background)
    X_test, y_test = _loader_to_numpy(test_loader, max_samples=n_explain)

    background = torch.tensor(X_train[:n_background], dtype=torch.float32).to(device)
    explain_data = torch.tensor(X_test[:n_explain], dtype=torch.float32).to(device)

    print(f"Obliczam SHAP (DeepExplainer) dla: {model_name}...")
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(explain_data)
    
    if torch.is_tensor(shap_values):
        shap_values = shap_values.cpu().numpy()
        
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Konwersja (n_samples, n_features, n_classes) na liste [ (n_samples, n_features) dla kazdej klasy ]
        shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

    print(f"SHAP obliczone. Kształt (klasa 0): {shap_values[0].shape}")

    return shap_values, X_test, y_test


def plot_shap_summary(
    shap_values,
    X_test: np.ndarray,
    model_name: str = "model",
    class_idx: int = 2,
    max_display: int = 20,
) -> None:
    """
    Summary plot SHAP — top N najbardziej wpływowych wymiarów embeddingu
    dla wybranej klasy.

    class_idx: 0=negative, 1=neutral, 2=positive
    """

    class_names = {0: "Negative", 1: "Neutral", 2: "Positive"}
    class_label = class_names.get(class_idx, str(class_idx))

    plt.figure()
    shap.summary_plot(
        shap_values[class_idx],
        X_test,
        max_display=max_display,
        show=False,
        plot_type="dot",
    )
    plt.title(f"SHAP Summary — {model_name} | Klasa: {class_label}")
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, f"shap_summary_{model_name}_class{class_idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"SHAP summary zapisany: {save_path}")
    plt.show()


def plot_shap_bar(
    shap_values,
    model_name: str = "model",
    max_display: int = 20,
) -> None:
    """
    Bar plot — średnia |SHAP| dla każdej klasy (mean absolute importance).
    """

    n_classes = len(shap_values)
    class_names = ["Negative", "Neutral", "Positive"]
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5))

    for i, ax in enumerate(axes):
        mean_abs = np.abs(shap_values[i]).mean(axis=0)
        top_idx = np.argsort(mean_abs)[-max_display:]
        ax.barh(range(len(top_idx)), mean_abs[top_idx])
        ax.set_yticks(range(len(top_idx)))
        ax.set_yticklabels([f"dim_{j}" for j in top_idx], fontsize=7)
        ax.set_title(f"Klasa: {class_names[i]}")
        ax.set_xlabel("mean |SHAP|")

    plt.suptitle(f"SHAP Feature Importance — {model_name}", fontsize=13)
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, f"shap_bar_{model_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"SHAP bar plot zapisany: {save_path}")
    plt.show()

# SHAP dla TF-IDF + LogisticRegression (baseline — słowa jako cechy)

def run_shap_tfidf(
    tweets: list,
    labels: list,
    model_name: str = "tfidf_logreg",
    max_features: int = 5000,
    n_explain: int = 200,
) -> None:
    """
    Trenuje prosty baseline LogReg + TF-IDF i oblicza SHAP LinearExplainer.
    
    Jest to DODATKOWA analiza dająca interpretowalność na poziomie słów
    (w przeciwieństwie do wymiarów embeddingów).
    
    Przydatne do sekcji dyskusji w projekcie.
    """

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    print("Trening baseline TF-IDF + LogReg...")

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X = vectorizer.fit_transform(tweets)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nWyniki baseline TF-IDF + LogReg:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Neutral", "Positive"]))

    # SHAP LinearExplainer
    print("Obliczam SHAP (LinearExplainer)...")
    explainer = shap.LinearExplainer(clf, X_train, feature_perturbation="interventional")

    X_explain = X_test[:n_explain]
    shap_values = explainer.shap_values(X_explain)
    
    # Obsługa nowszego formatu shap.Explanation / macierzy 3D dla LinearExplainer
    if not isinstance(shap_values, list):
        if hasattr(shap_values, 'values'):
            shap_values_array = shap_values.values
        else:
            shap_values_array = shap_values
            
        if shap_values_array.ndim == 3:
            shap_values = [shap_values_array[:, :, i] for i in range(shap_values_array.shape[2])]

    feature_names = vectorizer.get_feature_names_out()

    for class_idx, class_label in enumerate(["Negative", "Neutral", "Positive"]):
        plt.figure()
        shap.summary_plot(
            shap_values[class_idx],
            X_explain.toarray(),
            feature_names=feature_names,
            max_display=20,
            show=False,
            plot_type="dot",
        )
        plt.title(f"SHAP TF-IDF — {class_label}")
        plt.tight_layout()

        save_path = os.path.join(PLOTS_DIR, f"shap_tfidf_{model_name}_class{class_idx}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"SHAP TF-IDF zapisany: {save_path}")
        plt.show()

    return clf, vectorizer