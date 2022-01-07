# -*- coding: utf-8 -*-
import logging
import os
import zipfile
import hydra

from pathlib import Path
from dotenv import find_dotenv, load_dotenv

@hydra.main(config_path="../../config/data/", config_name='data_config.yaml')
def main(conf) -> None:
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info("Downloading dataset from kaggle")
    zip_folder = os.path.join(conf.dataset_path, "raw")

    try:
        import kaggle
    except Exception:
        logger.warning(
            "Must athenticate the kaggle api according to https://www.kaggle.com/docs/api"  # noqa: E501
        )
        exit(1)

    try:
        kaggle.api.competition_download_files("nlp-getting-started",
                                              path=zip_folder)
    except Exception:
        logger.warning(
            "Must join the challange at: https://www.kaggle.com/c/nlp-getting-started/data"  # noqa: E501
        )
        exit(1)

    out_folder_raw = os.path.join(conf.dataset_path, "interim")
    os.makedirs(out_folder_raw, exist_ok=True)
    with zipfile.ZipFile(
        os.path.join(zip_folder, "nlp-getting-started.zip"), "r"
    ) as zip_ref:
        zip_ref.extractall(out_folder_raw)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
