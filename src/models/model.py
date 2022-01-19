import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

# fmt: off # isort:skip
from transformers import (  # isort:skip
    BertForSequenceClassification,  # isort:skip
    DistilBertForSequenceClassification,  # isort:skip
)  # isort:skip


class MegaCoolTransformer(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        if self.config.model["model"] == "bert":
            self.model = BertForSequenceClassification.from_pretrained(
                self.config.model["pretrained-model"],
                num_labels=self.config.model["num_labels"],
                output_attentions=False,
                output_hidden_states=False,
            )
        elif (
            self.config.model["model"] == "distilbert-base-uncased"
        ):  # default model is distilbert
            print("Using DistilBert")
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.config.model["pretrained-model"],
                num_labels=self.config.model["num_labels"],
            )

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        tweet, labels = batch
        rtn = self.model(tweet, labels=labels)
        loss = rtn["loss"]
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        tweet, labels = batch
        out = self(tweet)
        logits = out["logits"]
        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum()
        accuracy = correct / len(labels)
        self.log("test_accuracy", accuracy, prog_bar=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        tweets, labels = batch
        outputs = self.model(tweets, labels=labels)
        val_loss = outputs["loss"]
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum()
        accuracy = correct / len(labels)
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def setup(self, stage=None) -> None:
        pass

    def configure_optimizers(  # noqa: C901
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[object]]:
        if self.config.train["optimizer"] == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=float(self.config.train["lr"]),
                eps=float(self.config.train["eps"]),
                betas=(0.9, 0.999),
            )  # type: torch.optim.Optimizer
        elif self.config.train["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.train["lr"])
        elif self.config.train["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config.train["lr"])
        elif self.config.train["optimizer"] == "RMSprop":
            optimizer = torch.optim.RMSprop(
                self.parameters(), lr=self.config.train["lr"]
            )
        elif self.config.train["optimizer"] == "Adagrad":
            optimizer = torch.optim.Adagrad(
                self.parameters(), lr=self.config.train["lr"]
            )
        elif self.config.train["optimizer"] == "Adadelta":
            optimizer = torch.optim.Adadelta(
                self.parameters(), lr=self.config.train["lr"]
            )
        elif self.config.train["optimizer"] == "Adamax":
            optimizer = torch.optim.Adamax(
                self.parameters(), lr=self.config.train["lr"]
            )
        elif self.config.train["optimizer"] == "ASGD":
            optimizer = torch.optim.ASGD(self.parameters(), lr=self.config.train["lr"])
        elif self.config.train["optimizer"] == "LBFGS":
            optimizer = torch.optim.LBFGS(self.parameters(), lr=self.config.train["lr"])
        elif self.config.train["optimizer"] == "SparseAdam":
            optimizer = torch.optim.SparseAdam(
                self.parameters(), lr=self.config.train["lr"]
            )
        else:
            raise ValueError("Unknown optimizer")

        if self.config.train["scheduler"]["name"] == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.config.train["scheduler"]["mode"],
                factor=self.config.train["scheduler"]["factor"],
                patience=self.config.train["scheduler"]["patience"],
                verbose=True,
            )  # type: object
        elif self.config.train["scheduler"]["name"] == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.train["scheduler"]["T_max"],
                eta_min=self.config.train["scheduler"]["eta_min"],
            )
        elif self.config.train["scheduler"]["name"] == "ExponentialLR":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.config.train["scheduler"]["gamma"]
            )
        elif self.config.train["scheduler"]["name"] == "CosineAnnealingWarmRestarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.config.train["scheduler"]["T_0"],
                T_mult=self.config.train["scheduler"]["T_mult"],
            )
        elif self.config.train["scheduler"]["name"] == "MultiStepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.config.train["scheduler"]["milestones"],
                gamma=self.config.train["scheduler"]["gamma"],
            )
        else:
            raise ValueError("Unknown scheduler")

        return [optimizer], [scheduler]
