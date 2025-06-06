import json
from pathlib import Path

import git
import hydra
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from emotion_recognition.data.datamodule import EmotionDataModule
from emotion_recognition.models.model import EmotionClassifier
from emotion_recognition.utils.save_model import save_model


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
    pl.seed_everything(cfg.train.seed)

    datamodule = EmotionDataModule(
        train_path=cfg.data.train_path,
        val_path=cfg.data.val_path,
        model_name=cfg.model.model_name,
        batch_size=cfg.data.batch_size,
    )

    model = EmotionClassifier(
        model_name=cfg.model.model_name,
        num_labels=cfg.model.num_labels,
        lr=cfg.train.lr,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="cpu" if not torch.cuda.is_available() else "gpu",
        devices=1,
    )

    trainer.fit(model, datamodule=datamodule)
    save_model(model)

    # Create plots folder
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Plot validation metrics
    plt.plot(model.history["loss"], label="val_loss")
    plt.plot(model.history["acc"], label="val_acc")
    plt.title("Validation metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(plots_dir / "val_metrics.png")

    # Save run metadata
    repo = git.Repo(search_parent_directories=True)
    commit_hash = repo.head.object.hexsha

    run_info = {
        "model": cfg.model.model_name,
        "batch_size": cfg.data.batch_size,
        "lr": cfg.train.lr,
        "epochs": cfg.train.epochs,
        "seed": cfg.train.seed,
        "commit": commit_hash,
    }

    with open(plots_dir / "run_metadata.json", "w") as f:
        json.dump(run_info, f, indent=4)


if __name__ == "__main__":
    train()
