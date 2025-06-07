import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class EmotionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]["text"]
        label = int(self.df.iloc[idx]["label"])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label),
        }


class EmotionDataModule(pl.LightningDataModule):
    def __init__(self, train_path, val_path, model_name, batch_size):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.model_name = model_name
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Подвыборка для быстрого train
        train_df = pd.read_parquet(self.train_path).sample(n=500, random_state=42)
        val_df = pd.read_parquet(self.val_path).sample(n=100, random_state=42)
        self.train_dataset = EmotionDataset(train_df, self.tokenizer)
        self.val_dataset = EmotionDataset(val_df, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
