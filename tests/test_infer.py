import torch
from transformers import AutoTokenizer

from emotion_recognition.models.model import EmotionClassifier


def test_infer_prediction():
    model = EmotionClassifier(
        model_name="distilbert-base-uncased", num_labels=6, lr=1e-5
    )
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    model.eval()
    inputs = tokenizer(
        "I am very happy today!", return_tensors="pt", padding=True, truncation=True
    )
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        pred = torch.argmax(logits, dim=1).item()

    assert isinstance(pred, int)
    assert 0 <= pred <= 5
