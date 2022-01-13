from importlib import import_module
from pydoc import importfile
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytest
from src.data.dataset import DesasterTweetDataModule
import hydra
from hydra import initialize, compose 
from omegaconf import DictConfig
import os
from tests import _PROJECT_ROOT, _PATH_DATA


def test_loaders_len_split():
    with initialize('../config/'):
        cfg = compose(config_name='default_config.yaml')
        data_module = DesasterTweetDataModule(
            _PATH_DATA,
            batch_size=cfg.train.batch_size,
        )
        data_module.setup()
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()
        
        train_set_len = len(train_loader.dataset)
        val_set_len = len(val_loader.dataset)
        test_set_len = len(test_loader.dataset)
        assert train_set_len + val_set_len + test_set_len == 7613

        assert round(train_set_len * 100 / 7613) == cfg.build_features.split_train
        assert round(val_set_len * 100 / 7613) == cfg.build_features.split_eval
        assert round(test_set_len * 100 / 7613) == cfg.build_features.split_test




# def test_datapoints_shape():
#     trainloader, testloader = mnist()
#     for img, _ in trainloader:
#         assert img.shape[1:] == torch.Size([1, 28, 28])

#     for img, _ in testloader:
#         assert img.shape[1:] == torch.Size([1, 28, 28])


# def test_wrong_data_mode():
#     with pytest.raises(AttributeError, match='wrong option'):
#         trainset = CorruptMNISTset(type="haha")


# def test_all_labels_present():
#     trainloader, testloader = mnist()
#     train_labels = torch.empty(0)
#     for _, labels in trainloader:
#         train_labels = torch.cat((train_labels, labels), 0)
