from hydra import compose, initialize

from src.data.dataset import DesasterTweetDataModule
from tests import _PATH_DATA


def test_loaders_len_split():
    with initialize("../config/"):
        cfg = compose(config_name="default_config.yaml")
        data_module = DesasterTweetDataModule(
            _PATH_DATA, batch_size=cfg.train.batch_size,
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
