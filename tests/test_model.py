import os

import pytest
import torch
from hydra import compose, initialize

from src.models.model import MegaCoolTransformer
from tests import _PROJECT_ROOT


@pytest.mark.skipif(
    not os.path.exists(_PROJECT_ROOT + "/config"), reason="Config files not found"
)
def test_distil_model_output_shape():
    with initialize("../config/"):
        cfg = compose(config_name="default_config.yaml")

        cfg.model["model"] = "distilbert"
        model = MegaCoolTransformer(cfg)
        x = torch.randint(0, 1000, (5, 140))
        output = model(x).logits

        assert output.shape == torch.Size([5, 2])


@pytest.mark.skipif(
    not os.path.exists(_PROJECT_ROOT + "/config"), reason="Config files not found"
)
def test_distil_model_is_default():
    with initialize("../config/"):
        cfg = compose(config_name="default_config.yaml")

        cfg.model["model"] = "non-existing-model"
        model = MegaCoolTransformer(cfg)

        assert "DistilBertForSequenceClassification" in str(type(model.model))
