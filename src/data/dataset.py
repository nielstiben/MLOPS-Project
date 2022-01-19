# Note - you must have torchvision installed for this example
import os
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class DesasterTweets(Dataset):
    def __init__(self, path: str, type: str = "train") -> None:
        if type == "train":
            file_tweets = os.path.join(path, "train_inputs.pkl")
            file_labels = os.path.join(path, "train_labels.pkl")
            file_masks = os.path.join(path, "train_masks.pkl")
        elif type == "test":
            file_tweets = os.path.join(path, "validation_inputs.pkl")
            file_labels = os.path.join(path, "validation_labels.pkl")
            file_masks = os.path.join(path, "validation_masks.pkl")
        # elif type == "eval":
        #     file_tweets = os.path.join(path, "eval_inputs.pkl")
        #     # file_labels = os.path.join(path, "eval_labels.pkl")
        #     file_masks = os.path.join(path, "eval_masks.pkl")
        else:
            raise Exception(f"Unknown Dataset type: {type}")

        self.tweets = torch.load(file_tweets)
        self.labels = torch.load(file_labels)
        self.masks = torch.load(file_masks)

        assert len(self.tweets) == len(
            self.labels
        ), "Number of tweets does not match the number of labels"

    def __len__(self) -> int:
        return len(self.tweets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.tweets[idx],
            self.masks[idx],
            self.labels[idx],
        )


class DesasterTweetDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int = 32):
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
        # self.valset = DesasterTweets(self.data_path, "eval")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.trainset, batch_size=self.batch_size, num_workers=self.cpu_cnt
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.testset, batch_size=self.batch_size, num_workers=self.cpu_cnt
        )

    # def val_dataloader(self) -> DataLoader:
    #     return DataLoader(
    #         self.valset, batch_size=self.batch_size, num_workers=self.cpu_cnt
    #     )
