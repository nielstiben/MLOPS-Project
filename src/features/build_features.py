# -*- coding: utf-8 -*-
import logging
import math
import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd  # type: ignore
from dotenv import find_dotenv, load_dotenv
from HelperFunctions import create_features, read_samples, select_field
from omegaconf import DictConfig
from transformers import BertTokenizer


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
    logger.info("fetching data...")
    data_path = os.path.join(hydra.utils.get_original_cwd(), c.path, "interim")
    train = pd.read_csv(os.path.join(data_path, "train.csv"))
    # test = pd.read_csv(os.path.join(data_path, "test.csv"))

    # %% Encode
    logger.info("encoding...")
    tokenizer = BertTokenizer.from_pretrained(
        cfg.model["pretrained-model"], do_lower_case=True
    )
    train_examples, train_df = read_samples(train)
    all_label = train_df["target"].astype(int).values
    train_features = create_features(
        train_examples, tokenizer, cfg.model["max_sequence_length"], True
    )
    all_input_ids = np.array(select_field(train_features, "input_ids"))
    all_input_mask = np.array(select_field(train_features, "input_mask"))
    all_segment_ids = np.array(select_field(train_features, "segment_ids"))
    # all_label = np.array([f.label for f in train_features])

    # %% Train test val split
    logger.info("train test val splitting...")
    split_train = math.floor(len(all_label) * (c.split_train / 100))
    split_eval = math.floor(len(all_label) * ((c.split_train + c.split_eval) / 100))

    train_input_ids, train_input_mask, train_segment_ids, train_label = (
        list(all_input_ids[:split_train]),
        list(all_input_mask[:split_train]),
        list(all_segment_ids[:split_train]),
        list(all_label[:split_train]),
    )
    test_input_ids, test_input_mask, test_segment_ids, test_label = (
        list(all_input_ids[split_train:split_eval]),
        list(all_input_mask[split_train:split_eval]),
        list(all_segment_ids[split_train:split_eval]),
        list(all_label[split_train:split_eval]),
    )
    val_input_ids, val_input_mask, val_segment_ids, val_label = (
        list(all_input_ids[split_eval:]),
        list(all_input_mask[split_eval:]),
        list(all_segment_ids[split_eval:]),
        list(all_label[split_eval:]),
    )

    # %% Save to file
    logger.info("saving outputs...")
    data_path = os.path.join(hydra.utils.get_original_cwd(), c.path, "processed")
    np.savetxt(
        os.path.join(data_path, "train_input_ids.csv"), train_input_ids, delimiter=","
    )
    np.savetxt(
        os.path.join(data_path, "train_input_mask.csv"), train_input_mask, delimiter=","
    )
    np.savetxt(
        os.path.join(data_path, "train_segment_ids.csv"),
        train_segment_ids,
        delimiter=",",
    )
    np.savetxt(os.path.join(data_path, "train_labels.csv"), train_label, delimiter=",")

    np.savetxt(
        os.path.join(data_path, "test_input_ids.csv"), test_input_ids, delimiter=","
    )
    np.savetxt(
        os.path.join(data_path, "test_input_mask.csv"), test_input_mask, delimiter=","
    )
    np.savetxt(
        os.path.join(data_path, "test_segment_ids.csv"), test_segment_ids, delimiter=","
    )
    np.savetxt(os.path.join(data_path, "test_labels.csv"), test_label, delimiter=",")
    #
    np.savetxt(
        os.path.join(data_path, "val_input_ids.csv"), val_input_ids, delimiter=","
    )
    np.savetxt(
        os.path.join(data_path, "val_input_mask.csv"), val_input_mask, delimiter=","
    )
    np.savetxt(
        os.path.join(data_path, "val_segment_ids.csv"), val_segment_ids, delimiter=","
    )
    np.savetxt(os.path.join(data_path, "val_labels.csv"), val_label, delimiter=",")

    oof_train = np.zeros((len(train_label), 2), dtype=np.float32)
    oof_test = np.zeros((len(test_label), 2), dtype=np.float32)
    oof_val = np.zeros((len(val_label), 2), dtype=np.float32)
    np.savetxt(os.path.join(data_path, "oof_train.csv"), oof_train, delimiter=",")
    np.savetxt(os.path.join(data_path, "oof_test.csv"), oof_test, delimiter=",")
    np.savetxt(os.path.join(data_path, "oof_val.csv"), oof_val, delimiter=",")
    logger.info("done!")


if __name__ == "__main__":
    print(logging.__file__)
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
