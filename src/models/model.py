from pytorch_lightning import LightningModule
from transformers import RobertaModel, RobertaConfig
import torch

class MegaCoolTransformer(LightningModule):

    def __init__(self, config):
        super().__init__()
        self.hparams = config.model

        self.model = RobertaModel.from_pretrained()

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        pass

    def setup(self, stage=None) -> None:
        pass

    def configure_optimizers(self):
        pass