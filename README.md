# Emotion Recognition from Text

## Project Description

This project addresses the task of emotion classification from English-language text using a fine-tuned DistilBERT model.  
The classification is performed across six emotion categories: sadness, joy, love, anger, fear, and surprise.  
The model is intended for use in applications such as conversational agents, sentiment analysis, and social media monitoring.

---

## Environment Setup

Clone the repository and set up the virtual environment:

```bash
git clone <your-repo-url>
cd emotion-recognition
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Additionally, make sure the following packages are installed:

```bash
pip install torch torchvision torchaudio
pip install onnx fastapi uvicorn
```

---

## Model Training

To launch training, use the command below:

```bash
PYTHONPATH=src python src/emotion_recognition/train.py
```

This will:
- Load and preprocess the `dair-ai/emotion` dataset
- Train a DistilBERT-based classifier
- Save validation metrics to the `plots/` directory
- Save model weights to `checkpoints/`

Training parameters and paths are configured via Hydra (`configs/` directory).

---

## Inference (Script Mode)

To test the trained model with a predefined input:

```bash
PYTHONPATH=src python src/emotion_recognition/infer.py
```

Expected output:

```json
Predicted class: 1
```

---

## Inference via API (FastAPI)

To run a local REST API server:

```bash
uvicorn emotion_recognition.api:app --reload --port 8000 --host 127.0.0.1 --app-dir src
```

After the server starts, open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
To test the endpoint:

```json
POST /predict
{
  "text": "I feel amazing and full of joy!"
}
```

Response:

```json
{
  "emotion_class": 1
}
```

---

## Model Export

To export the model to ONNX format:

```bash
PYTHONPATH=src python scripts/export_to_onnx.py
```

The resulting file will be saved as `exports/emotion_model.onnx`.

Files required for deployment:
- `checkpoints/emotion_model.pt` or `exports/emotion_model.onnx`
- HuggingFace tokenizer: `distilbert-base-uncased`

---

## Directory Structure

- `src/emotion_recognition/`: source code
- `scripts/`: utility scripts (e.g., export)
- `configs/`: configuration files (Hydra)
- `plots/`: training metrics visualizations
- `checkpoints/`: saved model weights
- `exports/`: exported ONNX model

---

## Author

Konstantin Kairov, MIPT, MLOps Spring 2025