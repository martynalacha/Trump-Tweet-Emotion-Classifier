import torch
import pandas as pd
from torch.utils.data import Dataset

from prepare_data import prepare_dataset,EMBEDDINGS_PATH,LABELS_PATH,METADATA_PATH


class TrumpTweetDataset(Dataset):

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