# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig
from transformers import AutoTokenizer


# See kaggle notebook:
# https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert


@hydra.main(config_path="./../../config", config_name="default_config.yaml")
def main(cfg: DictConfig) -> None:
    """Converts .CSV files into tokenized PyTorch tensors"""
    logger = logging.getLogger(__name__)
    logger.info("Tokenize tweets")
    c = cfg.build_features

    # %% Fetch Data
    data_path = os.path.join(hydra.utils.get_original_cwd(), c.path, "interim")
    train = pd.read_csv(
        os.path.join(data_path, "train.csv"), dtype={"id": np.int16, "target": np.int8}
    )
    test = pd.read_csv(os.path.join(data_path, "test.csv"), dtype={"id": np.int16})
    X_train, y_train = list(train.text), list(train.target)
    X_test = list(test.text)

    # %% Encode
    tokenizer = AutoTokenizer.from_pretrained(c.tokenizer)

    def encode(text: str) -> list[int]:
        tokens = tokenizer.encode(text)
        tokens = tokens[: c.max_sequence_length - 2]
        pad_len = c.max_sequence_length - len(tokens)
        tokens += [0] * pad_len
        return tokens

    X_train = list(map(encode, list(X_train)))
    X_test = list(map(encode, list(X_test)))

    #%% Convert to tensor
    X_train = torch.IntTensor(X_train)
    y_train = torch.BoolTensor(y_train)
    X_test = torch.IntTensor(X_test)

    #%% Save to file
    data_path = os.path.join(hydra.utils.get_original_cwd(), c.path, "processed")
    torch.save(X_train, os.path.join(data_path, "X_train.pkl"))
    torch.save(y_train, os.path.join(data_path, "y_train.pkl"))
    torch.save(X_test, os.path.join(data_path, "X_test.pkl"))
    logger.info("Finished! Output saved to '{}'".format(data_path))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
