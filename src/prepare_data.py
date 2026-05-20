import os
import re

import kagglehub
import nltk
import pandas as pd
import torch

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer,AutoModel

from tqdm import tqdm


DATASET_NAME = "austinreese/trump-tweets"
CSV_NAME = "realdonaldtrump.csv"

PROCESSED_DIR = "data/processed"

EMBEDDINGS_PATH = os.path.join(
    PROCESSED_DIR,
    "embeddings.pt",
)

LABELS_PATH = os.path.join(
    PROCESSED_DIR,
    "labels.pt",
)

METADATA_PATH = os.path.join(
    PROCESSED_DIR,
    "metadata.csv",
)


def clean_text(text: str):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", " ", text)

    return text.strip()


def generate_sentiment_label(
    sia,
    text,
    threshold,
):

    score = sia.polarity_scores(text)

    compound = score["compound"]

    if compound >= threshold:
        label = 2  # positive

    elif compound <= -threshold:
        label = 0  # negative

    else:
        label = 1  # neutral

    return label


def prepare_dataset(
    threshold: float = 0.25,
    overwrite: bool = False,
    batch_size: int = 32,
    device = None,
):

    if (
        not overwrite
        and os.path.exists(EMBEDDINGS_PATH)
        and os.path.exists(LABELS_PATH)
    ):
        print("Loading from files.")
        return

    os.makedirs(
        PROCESSED_DIR,
        exist_ok=True,
    )

    nltk.download(
        "vader_lexicon",
        quiet=True,
    )




    print("Downloading dataset...")

    path = kagglehub.dataset_download(
        DATASET_NAME
    )

    raw_df = pd.read_csv(
        os.path.join(path, CSV_NAME)
    )

    tweets = (
        raw_df["content"].dropna().tolist()
    )

    cleaned_tweets = [
        clean_text(tweet) for tweet in tweets
    ]

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )

    print("Loading BERT model...")

    bert_model = AutoModel.from_pretrained(
        "distilbert-base-uncased"
    )

    bert_model.to(device)

    bert_model.eval()

    sia = SentimentIntensityAnalyzer()

    all_embeddings = []
    all_labels = []

    metadata_rows = []

    print(
        "Generating embeddings..."
    )

    for i in tqdm(
        range(
            0,
            len(cleaned_tweets),
            batch_size,
        )
    ):

        batch_tweets = cleaned_tweets[
            i : i + batch_size
        ]

        batch_labels = [
            generate_sentiment_label(
                sia,
                tweet,
                threshold,
            )
            for tweet in batch_tweets
        ]

        inputs = tokenizer(
            batch_tweets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )

        inputs = {
            key: value.to(device)
            for key, value
            in inputs.items()
        }

        with torch.no_grad():

            outputs = bert_model(
                **inputs
            )

            cls_embeddings = (
                outputs.last_hidden_state[
                    :, 0, :
                ]
            )

        cls_embeddings = (
            cls_embeddings
            .cpu()
        )

        all_embeddings.append(
            cls_embeddings
        )

        all_labels.extend(
            batch_labels
        )

        for tweet, label in zip(
            batch_tweets,
            batch_labels,
        ):
            metadata_rows.append(
                {
                    "tweet": tweet,
                    "label": label,
                }
            )

    print("Saving tensors...")

    embeddings_tensor = torch.cat(
        all_embeddings,
        dim=0,
    )

    labels_tensor = torch.tensor(
        all_labels,
        dtype=torch.long,
    )

    torch.save(
        embeddings_tensor,
        EMBEDDINGS_PATH,
    )

    torch.save(
        labels_tensor,
        LABELS_PATH,
    )

    metadata_df = pd.DataFrame(
        metadata_rows
    )

    metadata_df.to_csv(
        METADATA_PATH,
        index=False,
    )

    print("Done.")