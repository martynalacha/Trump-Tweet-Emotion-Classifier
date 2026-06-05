"""
Przygotowanie danych - generowanie embeddingów i etykiet emocji.

Etykiety (11 klas) generowane przez NRC Emotion Lexicon oraz reguły:
0 angry     (NRC: anger)
1 fearful   (NRC: fear)
2 joyful    (NRC: joy)
3 sad       (NRC: sadness)
4 disgusted (NRC: disgust)
5 surprised (NRC: surprise)
6 enthusiastic (NRC: anticipation)
7 trusting  (NRC: trust)
8 sarcastic (reguły: negacja + pozytywny sentyment VADER)
9 patriotic (słownik słów kluczowych)
10 neutral

Jeśli tweet nie pasuje do żadnej klasy -> klasa z najwyższym score NRC,
Jeśli żadna emocja NRC nie przekracza progu -> klasa 10 (neutral).

Embeddingi: CLS token DistilBERT (768-dim), SBERT (384-dim), GloVe (200-dim).
"""

import os
import re 

import kagglehub
import nltk
import pandas as pd
import torch
import numpy as np

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import gensim.downloader as api
from tqdm import tqdm
from nrclex import NRCLex

# Ścieżki i stałe
DATASET_NAME = "austinreese/trump-tweets"
CSV_NAME = "realdonaldtrump.csv"

PROCESSED_DIR = "data/processed"
METADATA_PATH = os.path.join(PROCESSED_DIR, "metadata.csv")
LABELS_PATH = os.path.join(PROCESSED_DIR, "labels.pt")

# Ścieżki reprezentacji wektorowych (ujednolicone nazwy)
DISTILBERT_EMBEDDINGS_PATH = os.path.join(PROCESSED_DIR, "distilbert_embeddings.pt")
SBERT_EMBEDDINGS_PATH      = os.path.join(PROCESSED_DIR, "sbert_embeddings.pt")
GLOVE_EMBEDDINGS_PATH      = os.path.join(PROCESSED_DIR, "glove_embeddings.pt")
GLOVE_VOCAB_PATH           = os.path.join(PROCESSED_DIR, "glove_vocab.pt")
GLOVE_SEQUENCES_PATH       = os.path.join(PROCESSED_DIR, "glove_sequences.pt")

# Konfiguracja modeli
SBERT_MODEL_NAME      = "all-MiniLM-L6-v2"
GLOVE_MODEL_NAME      = "glove-twitter-200"
GLOVE_DIM             = 200
MAX_SEQ_LEN           = 64
PAD_IDX               = 0
UNK_IDX               = 1

EMOTION_LABELS = [
    "angry", "fearful", "joyful", "sad", "disgusted", 
    "surprised", "enthusiastic", "trusting", "sarcastic", "patriotic", "neutral"
]

NRC_TO_IDX = {
    "anger": 0, "fear": 1, "joy": 2, "sadness": 3, 
    "disgust": 4, "surprise": 5, "anticipation": 6, "trust": 7
}

PATRIOTIC_KEYWORDS = {"america", "american", "usa", "us", "nation", "country", "patriot", 
                      "patriotic", "flag", "freedom", "liberty", "constitution", 
                      "military", "veterans", "troops", "maga", "greatagain", 
                      "proud", "border", "homeland"}

# ──────────────────────────────────────────────────────────────────────────────
# Funkcje pomocnicze logiki biznesowej
# ──────────────────────────────────────────────────────────────────────────────

def _get_nrc_scores(text: str) -> dict:
    """Zwraca słownik {emotion: count} dla danego tekstu używając NRCLex."""
    emotion = NRCLex(text)
    scores = emotion.raw_emotion_scores
    for e in NRC_TO_IDX:
        if e not in scores:
            scores[e] = 0
    return scores

def _is_sarcastic(text: str, vader_sia) -> bool:
    """Prosta heurystyka sarkastyczna."""
    lower = text.lower()
    negations = {"no", "not", "never", "nobody", "none", "nothing", "nor", "neither"}
    has_negation = any(neg in lower.split() for neg in negations)
    vader_score = vader_sia.polarity_scores(text)["compound"]
    if has_negation and vader_score > 0.3:
        return True
    if "!" in text and vader_score < -0.3:
        return True
    return False

def _is_patriotic(text: str) -> bool:
    """Sprawdza, czy tweet zawiera słowa patriotyczne."""
    tokens = set(re.sub(r"[^a-z\s]", "", text.lower()).split())
    return bool(tokens & PATRIOTIC_KEYWORDS)

def generate_emotion_label(text: str, vader_sia, threshold: float = 1.0) -> int:
    """Generuje etykietę emocji (0-10) dla danego tekstu na podstawie reguł i NRC."""
    
    try:
        scores = _get_nrc_scores(text)
        nrc_emotions = {e: scores.get(e, 0) for e in NRC_TO_IDX}
        max_score = max(nrc_emotions.values())
        if max_score > threshold:
            dominant = max(nrc_emotions, key=nrc_emotions.get)
            return NRC_TO_IDX[dominant]
    except Exception as e:
        print(f"Error during NRC scoring: {e}")

    if _is_sarcastic(text, vader_sia):
        return 8
    if _is_patriotic(text):
        return 9
    
    return 10

def clean_text(text: str) -> str:
    """Usuwa URL-e i wzmianki z tekstu."""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", " ", text)
    return text.strip()

# ──────────────────────────────────────────────────────────────────────────────
# Generatory poszczególnych embeddingów
# ──────────────────────────────────────────────────────────────────────────────

def _prepare_distilbert_embeddings(cleaned_tweets: list, batch_size: int, device) -> None:
    """Generuje i zapisuje embeddingi DistilBERT (CLS token)."""
    print("\nŁaduję model DistilBERT i tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    bert_model = AutoModel.from_pretrained("distilbert-base-uncased")
    bert_model.to(device)
    bert_model.eval()

    all_embeddings = []
    print("Generuję embeddingi DistilBERT (CLS token)...")
    for i in tqdm(range(0, len(cleaned_tweets), batch_size), desc="DistilBERT"):
        batch_tweets = cleaned_tweets[i: i + batch_size]
        inputs = tokenizer(batch_tweets, return_tensors="pt", padding=True, truncation=True, max_length=64)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = bert_model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu())

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    torch.save(embeddings_tensor, DISTILBERT_EMBEDDINGS_PATH)
    print(f"Zapisano DistilBERT embeddingi: {embeddings_tensor.shape}")


def _prepare_sbert_embeddings(cleaned_tweets: list, batch_size: int, device) -> None:
    """Generuje i zapisuje embeddingi SBERT (all-MiniLM-L6-v2)."""
    print(f"\nŁaduję model SBERT: {SBERT_MODEL_NAME}...")
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device=str(device) if device else "cpu")

    all_embeddings = []
    for i in tqdm(range(0, len(cleaned_tweets), batch_size), desc="SBERT"):
        batch = cleaned_tweets[i: i + batch_size]
        emb = sbert_model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
        all_embeddings.append(emb.cpu())

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    torch.save(embeddings_tensor, SBERT_EMBEDDINGS_PATH)
    print(f"Zapisano SBERT embeddingi: {embeddings_tensor.shape}")


def _prepare_glove_data(cleaned_tweets: list) -> None:
    """Generuje macierz embeddingów oraz tokenizuje sekwencje indeksów GloVe."""
    print(f"\nPobieram / Ładuję model GloVe: {GLOVE_MODEL_NAME}...")
    glove = api.load(GLOVE_MODEL_NAME)

    vocab = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
    embedding_matrix = [np.zeros(GLOVE_DIM), np.random.randn(GLOVE_DIM) * 0.01]
    
    for word in glove.key_to_index:
        if word not in vocab:
            vocab[word] = len(vocab)
            embedding_matrix.append(glove[word])

    embeddings_tensor = torch.tensor(np.array(embedding_matrix, dtype=np.float32))
    print(f"Macierz GloVe: {embeddings_tensor.shape}")

    sequences = []
    for tweet in cleaned_tweets:
        tokens = tweet.lower().split()[:MAX_SEQ_LEN]
        idxs = [vocab.get(tok, UNK_IDX) for tok in tokens] or [PAD_IDX]
        sequences.append(torch.tensor(idxs, dtype=torch.long))

    torch.save(embeddings_tensor, GLOVE_EMBEDDINGS_PATH)
    torch.save(vocab, GLOVE_VOCAB_PATH)
    torch.save(sequences, GLOVE_SEQUENCES_PATH)
    print(f"Zapisano GloVe dane. Vocab: {len(vocab)}, Sekwencje: {len(sequences)}")

# ──────────────────────────────────────────────────────────────────────────────
# Główny koordynator procesu
# ──────────────────────────────────────────────────────────────────────────────

def prepare_dataset(
        overwrite: bool = False,
        batch_size: int = 64,
        device=None,
        threshold: float = 0.0
):
    """Odpowiada za kompletne pobranie, etykietowanie i wektoryzację danych offline."""
    if (
        not overwrite
        and os.path.exists(DISTILBERT_EMBEDDINGS_PATH)
        and os.path.exists(SBERT_EMBEDDINGS_PATH)
        and os.path.exists(GLOVE_EMBEDDINGS_PATH)
        and os.path.exists(LABELS_PATH)
        and os.path.exists(METADATA_PATH)
    ):
        print("Komplet wszystkich danych (etykiety + 3 rodzaje embeddingów) już istnieje na dysku.")
        return
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    nltk.download("vader_lexicon", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    vader_sia = SentimentIntensityAnalyzer()

    print("Pobieram dataset z Kaggle...")
    path = kagglehub.dataset_download(DATASET_NAME)
    raw_df = pd.read_csv(os.path.join(path, CSV_NAME))

    tweets = raw_df["content"].dropna().tolist()
    cleaned_tweets = [clean_text(t) for t in tweets]
    print(f"Liczba tweetów: {len(cleaned_tweets)}")

    # Generowanie Etykiet i Metadanych
    print("\nGeneruję etykiety emocji (NRC + reguły)...")
    all_labels = []
    metadata_rows = []

    for tweet in tqdm(cleaned_tweets, desc="Etykietowanie"):
        label = generate_emotion_label(tweet, vader_sia, threshold=threshold)
        all_labels.append(label)
        metadata_rows.append({"tweet": tweet, "label": label})

    # Statystyki dystrybucji klas
    label_series = pd.Series(all_labels)
    print("\nRozkład etykiet:")
    for idx, name in enumerate(EMOTION_LABELS):
        count = (label_series == idx).sum()
        pct = count / len(all_labels) * 100 if len(all_labels) > 0 else 0
        print(f"{idx:2d}. {name:<15} {count:6d} ({pct:.2f}%)")

    # Zapis podstawowych etykiet i metadanych tekstowych
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    torch.save(labels_tensor, LABELS_PATH)
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(METADATA_PATH, index=False)

    # Ustalenie urządzenia sprzętowego
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Generowanie trzech reprezentacji embeddingowych na dysk
    _prepare_distilbert_embeddings(cleaned_tweets, batch_size, device)
    _prepare_sbert_embeddings(cleaned_tweets, batch_size, device)
    _prepare_glove_data(cleaned_tweets)

    print("\nWszystkie potoki danych wejściowych zostały przygotowane pomyślnie!")


if __name__ == "__main__":
    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Wykryte urządzenie obliczeniowe: {target_device.upper()}")
    
    prepare_dataset(
        overwrite=True,       
        batch_size=64,       
        device=target_device,
        threshold=1.0        
    )