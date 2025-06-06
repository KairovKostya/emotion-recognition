import torch
from transformers import AutoTokenizer

from emotion_recognition.models.model import EmotionClassifier


def load_model(model_path, model_name="distilbert-base-uncased", num_labels=6):
    model = EmotionClassifier(model_name=model_name, num_labels=num_labels, lr=1e-5)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def predict(text, model, tokenizer, max_length=128):
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(encoded["input_ids"], encoded["attention_mask"])
        predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class


if __name__ == "__main__":
    model_path = "checkpoints/emotion_model.pt"
    model_name = "distilbert-base-uncased"

    # Загружаем модель и токенизатор
    model = load_model(model_path, model_name=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Пример текста
    text = "I feel amazing and full of joy!"
    prediction = predict(text, model, tokenizer)

    print(f"Predicted class: {prediction}")
