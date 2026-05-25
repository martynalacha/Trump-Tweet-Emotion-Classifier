import torch
import pandas as pd
from torch.utils.data import Dataset

from prepare_data import prepare_dataset,EMBEDDINGS_PATH,LABELS_PATH,METADATA_PATH


class TrumpTweetDataset(Dataset):
    """
    Dataset tweetów Trumpa z embeddingami DistilBERT i etykietami VADER.
 
    Etykiety: 0 = negative, 1 = neutral, 2 = positive
    Embeddingi: 768-dim CLS token z DistilBERT
    """

    def __init__(
        self,
        threshold: float = 0.25,
        overwrite: bool = False,
        device = None,
    ):
        super().__init__()

        prepare_dataset(
            threshold=threshold,
            overwrite=overwrite,
            batch_size=32,
            device=device
        )

        self.embeddings = torch.load(
            EMBEDDINGS_PATH
        )

        self.labels = torch.load(
            LABELS_PATH
        )

        self.df = pd.read_csv(
            METADATA_PATH
        )

    def __len__(self):

        return len(self.labels)

    def __getitem__(
        self,
        idx,
    ):

        return (
            self.embeddings[idx],
            self.labels[idx],
        )