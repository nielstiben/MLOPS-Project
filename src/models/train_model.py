import logging
import os
from pathlib import Path

import hydra
import torch
from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from src.data.dataset import DesasterTweetDataModule
from src.models.model import MegaCoolTransformer


@hydra.main(config_path="../../config", config_name="default_config.yaml")
def main(config: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Start Training...")
    gpus = 0
    if torch.cuda.is_available():
        # selects all available gpus
        print(f"Using {torch.cuda.device_count()} GPU(s) for training")
        gpus = -1
    else:
        print("Using CPU for training")

    data_module = DesasterTweetDataModule(
        os.path.join(hydra.utils.get_original_cwd(), config.data.path),
        batch_size=config.train.batch_size,
    )
    model = MegaCoolTransformer(config)

    trainer = Trainer(max_epochs=2, gpus=gpus)
    trainer.fit(model, data_module)
    trainer.test(model, data_module)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
