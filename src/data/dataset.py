# Note - you must have torchvision installed for this example
import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class DesasterTweets(Dataset):
    def __init__(self, path: str, type: str = "train") -> None:
        def load_csv(path, filename, label=False):
            n = np.genfromtxt(os.path.join(path, filename), delimiter=",")
            t = torch.from_numpy(n)
            if label:
                return t.to(torch.int64)
            else:
                return t

        if type == "train":
            self.input_ids = load_csv(path, "train_input_ids.csv").int()
            self.input_mask = load_csv(path, "train_input_mask.csv")
            self.segment_ids = load_csv(path, "train_segment_ids.csv")
            self.labels = load_csv(path, "train_labels.csv", label=True)
        elif type == "test":
            self.input_ids = load_csv(path, "test_input_ids.csv").int()
            self.input_mask = load_csv(path, "test_input_mask.csv")
            self.segment_ids = load_csv(path, "test_segment_ids.csv")
            self.labels = load_csv(path, "train_labels.csv", label=True)
        elif type == "eval":
            self.input_ids = load_csv(path, "val_input_ids.csv").int()
            self.input_mask = load_csv(path, "val_input_mask.csv")
            self.segment_ids = load_csv(path, "val_segment_ids.csv")
            self.labels = load_csv(path, "val_labels.csv", label=True)
        else:
            raise Exception(f"Unknown Dataset type: {type}")

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return (
            self.input_ids[idx],
            self.input_mask[idx],
            self.segment_ids[idx],
        ), self.labels[idx]


class DesasterTweetDataModule(pl.LightningDataModule):
    def __init__(
        self, data_path: str, batch_size: int = 8
    ):  # todo: Batch size from config
        super().__init__()
        self.data_path = os.path.join(data_path, "processed")
        self.batch_size = batch_size
        self.cpu_cnt = os.cpu_count() or 2

    def prepare_data(self) -> None:
        if not os.path.isdir(self.data_path):
            raise Exception("data is not prepared")

    def setup(self, stage: Optional[str] = None) -> None:
        self.trainset = DesasterTweets(self.data_path, "train")
        self.testset = DesasterTweets(self.data_path, "test")
        self.valset = DesasterTweets(self.data_path, "eval")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.trainset, batch_size=self.batch_size, num_workers=self.cpu_cnt
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.testset, batch_size=self.batch_size, num_workers=self.cpu_cnt
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valset, batch_size=self.batch_size, num_workers=self.cpu_cnt
        )
