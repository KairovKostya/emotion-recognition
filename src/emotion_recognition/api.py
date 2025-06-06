import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

from emotion_recognition.models.model import EmotionClassifier

# Инициализация FastAPI
app = FastAPI(title="Emotion Recognition API")

# Загрузка модели и токенизатора один раз при старте
MODEL_PATH = "checkpoints/emotion_model.pt"
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 6
MAX_LENGTH = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = EmotionClassifier(model_name=MODEL_NAME, num_labels=NUM_LABELS, lr=1e-5)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


# Схема входных данных
class TextInput(BaseModel):
    text: str


# Эндпоинт предсказания
@app.post("/predict")
def predict(input_data: TextInput):
    encoded = tokenizer(
        input_data.text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    with torch.no_grad():
        logits = model(encoded["input_ids"], encoded["attention_mask"])
        predicted_class = torch.argmax(logits, dim=1).item()
    return {"emotion_class": predicted_class}
