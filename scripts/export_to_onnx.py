import torch
from transformers import AutoTokenizer
from emotion_recognition.models.model import EmotionClassifier

def export_to_onnx(model_path="checkpoints/emotion_model.pt",
                   model_name="distilbert-base-uncased",
                   num_labels=6,
                   export_path="exports/emotion_model.onnx"):
    # Загрузка модели
    model = EmotionClassifier(model_name=model_name, num_labels=num_labels, lr=1e-5)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Токенизатор и фиктивный вход
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dummy_input = tokenizer("I feel great!", return_tensors="pt", padding="max_length", truncation=True, max_length=128)

    # Экспорт в ONNX
    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        export_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"}
        },
        opset_version=14
    )
    print(f"Model exported to: {export_path}")

if __name__ == "__main__":
    export_to_onnx()
