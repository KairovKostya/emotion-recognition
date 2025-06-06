from pathlib import Path

import torch


def save_model(model, path="checkpoints/emotion_model.pt"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # создаёт папку, если её нет
    torch.save(model.state_dict(), path)
