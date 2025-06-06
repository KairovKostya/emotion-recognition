# scripts/download_data.py
from datasets import load_dataset
import pandas as pd
from pathlib import Path

def download_and_save():
    dataset = load_dataset("dair-ai/emotion")
    save_dir = Path("data/raw")
    save_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "validation", "test"]:
        df = pd.DataFrame(dataset[split])
        df.to_parquet(save_dir / f"{split}.parquet")

if __name__ == "__main__":
    download_and_save()