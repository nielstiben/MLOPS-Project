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
        if self.hparams["optimizer"] == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])
        elif self.hparams["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
        elif self.hparams["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams["lr"])
        elif self.hparams["optimizer"] == "RMSprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.hparams["lr"])
        elif self.hparams["optimizer"] == "Adagrad":
            optimizer = torch.optim.Adagrad(self.parameters(), lr=self.hparams["lr"])
        elif self.hparams["optimizer"] == "Adadelta":
            optimizer = torch.optim.Adadelta(self.parameters(), lr=self.hparams["lr"])
        elif self.hparams["optimizer"] == "Adamax":
            optimizer = torch.optim.Adamax(self.parameters(), lr=self.hparams["lr"])
        elif self.hparams["optimizer"] == "ASGD":
            optimizer = torch.optim.ASGD(self.parameters(), lr=self.hparams["lr"])
        elif self.hparams["optimizer"] == "LBFGS":
            optimizer = torch.optim.LBFGS(self.parameters(), lr=self.hparams["lr"])
        elif self.hparams["optimizer"] == "SparseAdam":
            optimizer = torch.optim.SparseAdam(self.parameters(), lr=self.hparams["lr"])
        else:
            raise ValueError("Unknown optimizer")

        if self.hparams["scheduler"] == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=self.hparams['scheduler']['mode'],
                                                                   factor=self.hparams['scheduler']['factor'],
                                                                   patience=self.hparams['scheduler']['patience'],
                                                                   verbose=True)
        elif self.hparams["scheduler"] == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams['scheduler']['T_max'],
                                                                   eta_min=self.hparams['scheduler']['eta_min'])
        elif self.hparams["scheduler"] == "ExponentialLR":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams['scheduler']['gamma'])
        elif self.hparams["scheduler"] == "CosineAnnealingWarmRestarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.hparams['scheduler']['T_0'],
                                                                             T_mult=self.hparams['scheduler']['T_mult'])
        elif self.hparams["scheduler"] == "MultiStepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams['scheduler']['milestones'],
                                                             gamma=self.hparams['scheduler']['gamma'])
        else:
            raise ValueError("Unknown scheduler")
        return [optimizer], [scheduler]

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
