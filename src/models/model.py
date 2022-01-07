from pytorch_lightning import LightningModule
from transformers import BertForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score
import torch
import numpy as np


class MegaCoolTransformer(LightningModule):

    def __init__(self, config):
        super().__init__()
        self.hparams = config.model

        self.model = BertForSequenceClassification.from_pretrained(self.hparams["pretrained_model_name"],
                                                                   num_labels=self.hparams["num_labels"],
                                                                   output_attentions=False,
                                                                   output_hidden_states=False)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        loss, logits = self(**batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)
        labels = batch["labels"]

        accuracy = self.flat_accuracy(preds, labels)
        f1 = self.flat_f1(preds, batch["labels"])

        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_accuracy', accuracy, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def setup(self, stage=None) -> None:
        pass

    def configure_optimizers(self):
        pass

    @staticmethod
    def flat_accuracy(preds, labels):
        """A function for calculating accuracy scores"""

        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()

        return accuracy_score(labels_flat, pred_flat)

    @staticmethod
    def flat_f1(preds, labels):
        """A function for calculating f1 scores"""

        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()

        return f1_score(labels_flat, pred_flat)
