import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from transformers import AutoModel


class EmotionClassifier(pl.LightningModule):
    def __init__(self, model_name: str, num_labels: int, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(hidden_state)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        f1 = f1_score(labels.cpu(), preds.cpu(), average="weighted")

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)

        # сохраняем в in-memory списки
        if not hasattr(self, "val_losses"):
            self.val_losses = []
            self.val_accs = []
            self.val_f1s = []

        self.val_losses.append(loss.detach())
        self.val_accs.append(acc.detach())
        self.val_f1s.append(torch.tensor(f1))

    def on_validation_epoch_end(self):
        # Получаем сохранённые значения из валидации
        avg_loss = torch.tensor(self.val_losses).mean()
        avg_acc = torch.tensor(self.val_accs).mean()
        avg_f1 = torch.tensor(self.val_f1s).mean()

        self.log("val_loss_epoch", avg_loss, prog_bar=True)
        self.log("val_acc_epoch", avg_acc, prog_bar=True)
        self.log("val_f1_epoch", avg_f1, prog_bar=True)

        if not hasattr(self, "history"):
            self.history = {"loss": [], "acc": [], "f1": []}

        self.history["loss"].append(avg_loss.item())
        self.history["acc"].append(avg_acc.item())
        self.history["f1"].append(avg_f1.item())

        # очищаем
        self.val_losses.clear()
        self.val_accs.clear()
        self.val_f1s.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
