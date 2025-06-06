import os

from emotion_recognition.models.model import EmotionClassifier
from emotion_recognition.utils.save_model import save_model


def test_model_saves_to_disk(tmp_path):
    model = EmotionClassifier(
        model_name="distilbert-base-uncased", num_labels=6, lr=1e-5
    )
    path = tmp_path / "test_model.pt"
    save_model(model, path=str(path))
    assert os.path.exists(path)
