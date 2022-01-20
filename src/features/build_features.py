# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import hydra
import nltk
import numpy as np
import pandas as pd  # type: ignore
import torch
from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from tweet_cleaner import clean_tweet_list

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

    train = pd.read_csv(
        os.path.join(data_path, "train.csv"), dtype={"id": np.int16, "target": np.int8}
    )
    test = pd.read_csv(
        os.path.join(data_path, "test.csv"), dtype={"id": np.int16, "target": np.int8}
    )

    df_train = train
    df_test = test

    # %% Clean
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    df_train["text"] = clean_tweet_list(list(df_train.text))
    df_test["text"] = clean_tweet_list(list(df_test.text))
    df_train = df_train[df_train["text"] != ""]
    df_test = df_test[df_test["text"] != ""]
    df_train = df_train[["text", "target"]]

    texts_eval = df_test.text.values

    texts = df_train.text.values
    labels = df_train.target.values

    # %% Encode
    tokenizer = AutoTokenizer.from_pretrained(cfg.model["pretrained-model"])

    indices = tokenizer.batch_encode_plus(
        list(texts),
        max_length=c["max_sequence_length"],
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        truncation=True,
    )
    indices_eval = tokenizer.batch_encode_plus(
        list(texts_eval),
        max_length=c["max_sequence_length"],
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        truncation=True,
    )

    input_ids_eval = indices_eval["input_ids"]
    attention_masks_eval = indices_eval["attention_mask"]

    input_ids = indices["input_ids"]
    attention_masks = indices["attention_mask"]

    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
        input_ids, labels, random_state=42, test_size=0.1
    )

    train_masks, validation_masks, _, _ = train_test_split(
        attention_masks, labels, random_state=42, test_size=0.1
    )

    # %% Convert to tensor
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    validation_labels = torch.tensor(validation_labels, dtype=torch.long)
    train_masks = torch.tensor(train_masks, dtype=torch.long)
    validation_masks = torch.tensor(validation_masks, dtype=torch.long)
    eval_inputs = torch.tensor(input_ids_eval)
    eval_masks = torch.tensor(attention_masks_eval, dtype=torch.long)

    # %% Save to file
    data_path = os.path.join(hydra.utils.get_original_cwd(), c.path, "processed")
    torch.save(train_inputs, os.path.join(data_path, "train_inputs.pkl"))
    torch.save(train_labels, os.path.join(data_path, "train_labels.pkl"))
    torch.save(validation_inputs, os.path.join(data_path, "validation_inputs.pkl"))
    torch.save(validation_labels, os.path.join(data_path, "validation_labels.pkl"))
    torch.save(train_masks, os.path.join(data_path, "train_masks.pkl"))
    torch.save(validation_masks, os.path.join(data_path, "validation_masks.pkl"))
    torch.save(eval_inputs, os.path.join(data_path, "eval_inputs.pkl"))
    torch.save(eval_masks, os.path.join(data_path, "eval_masks.pkl"))
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
