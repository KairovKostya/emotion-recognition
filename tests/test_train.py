import torch

from emotion_recognition.models.model import EmotionClassifier


def test_model_output_shape():
    model = EmotionClassifier(
        model_name="distilbert-base-uncased", num_labels=6, lr=1e-5
    )
    model.eval()
    input_ids = torch.randint(0, 1000, (1, 32))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = model(input_ids, attention_mask)

    assert output.shape == (1, 6)
