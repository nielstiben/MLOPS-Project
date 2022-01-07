import logging
import hydra
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from src.models.model import MegaCoolTransformer


@hydra.main(config_path="../../config", config_name='default_config.yaml')
def train():
    logger = logging.getLogger(__name__)
    logger.info("Strat Training..")

    train_dataset = TwitterDataset()
    val_dataset = TwitterDataset()
    train_loader =DataLoader()
    val_loader = DataLoader()

    model = MegaCoolTransformer()

    trainer = Trainer(max_epochs=10, logger=pl.loggers.WandbLogger(project="mlops-mnist", config=config))


