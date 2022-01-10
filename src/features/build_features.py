# -*- coding: utf-8 -*-
import logging
import math
import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd  # type: ignore
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
    assert (
        c.split_train + c.split_test + c.split_eval == 100
    ), "The split train:{c.split_train} test:{c.split_test} is not possible"

    # %% Fetch Data
    data_path = os.path.join(hydra.utils.get_original_cwd(), c.path, "interim")
    data = pd.read_csv(
        os.path.join(data_path, "train.csv"), dtype={"id": np.int16, "target": np.int8}
    )
    split_train = math.floor(len(data) * (c.split_train / 100))
    split_eval = math.floor(len(data) * ((c.split_train + c.split_eval) / 100))
    tweet_train, label_train = list(data.text[:split_train]), list(
        data.target[:split_train]
    )
    tweet_eval, label_eval = list(data.text[split_train:split_eval]), list(
        data.target[split_train:split_eval]
    )
    tweet_test, lable_test = list(data.text[split_eval:]), list(
        data.target[split_eval:]
    )

    # %% Encode
    tokenizer = AutoTokenizer.from_pretrained(cfg.model["pretrained-model"])

    def encode(text: str) -> list[int]:
        tokens = tokenizer.encode(text)
        tokens = tokens[: c.max_sequence_length - 2]
        pad_len = c.max_sequence_length - len(tokens)
        tokens += [0] * pad_len
        return tokens

    X_train = list(map(encode, list(tweet_train)))
    X_eval = list(map(encode, list(tweet_eval)))
    X_test = list(map(encode, list(tweet_test)))

    # %% Convert to tensor
    X_train_t = torch.IntTensor(X_train)
    y_train_t = torch.IntTensor(label_train).long()
    X_eval_t = torch.IntTensor(X_eval)
    y_eval_t = torch.IntTensor(label_eval).long()
    X_test_t = torch.IntTensor(X_test)
    y_test_t = torch.IntTensor(lable_test).long()

    # %% Save to file
    data_path = os.path.join(hydra.utils.get_original_cwd(), c.path, "processed")
    torch.save(X_train_t, os.path.join(data_path, "tweets_train.pkl"))
    torch.save(y_train_t, os.path.join(data_path, "label_train.pkl"))
    torch.save(X_eval_t, os.path.join(data_path, "tweets_eval.pkl"))
    torch.save(y_eval_t, os.path.join(data_path, "label_eval.pkl"))
    torch.save(X_test_t, os.path.join(data_path, "tweets_test.pkl"))
    torch.save(y_test_t, os.path.join(data_path, "label_test.pkl"))
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
