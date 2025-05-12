import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    HubertModel,
    HubertForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoFeatureExtractor,
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Tuple, Union
import logging
import random
from dataclasses import dataclass

from shared import accent_mapping
from main import AccentDataset, cfg, DataCollatorForAccentClassification, shared_cfg


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


def predict_accent(audio_path, model_path="./hubert-accent-classifier-final"):
    # Load model and feature extractor
    model = HubertForSequenceClassification.from_pretrained(model_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)

    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    # Create input features
    inputs = feature_extractor(
        waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
    )

    # Move inputs to the same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get prediction
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get predicted class
    predicted_class = torch.argmax(logits, dim=1).item()

    # Map index back to accent name (reverse the accent_mapping)
    rev_accent_mapping = {v: k for k, v in accent_mapping.items()}

    probs = logits.softmax(dim=1)[0].tolist()
    accent_probs = {
        rev_accent_mapping[i]: round(prob * 100, 2) for i, prob in enumerate(probs)
    }
    accent_probs = dict(
        sorted(accent_probs.items(), key=lambda item: item[1], reverse=True)
    )

    predicted_accent = max(accent_probs, key=accent_probs.get)

    return predicted_accent, accent_probs


# print(f'MANDARIN4: {predict_accent("./out-min-n-20-max-t-60-augmented/processed_audio/dataset/mandarin4.wav")}')
# print(f'SWEDISH6: {predict_accent("./out-min-n-20-max-t-60-augmented/processed_audio/dataset/swedish6.wav")}')
# print(f'FARSI5: {predict_accent("./out-min-n-20-max-t-60-augmented/processed_audio/dataset/farsi5.wav")}')
# print(f'RUSSIAN3: {predict_accent("./out-min-n-20-max-t-60-augmented/processed_audio/dataset/russian3.wav")}')
# print(f'GERMAN3: {predict_accent("./out-min-n-20-max-t-60-augmented/processed_audio/dataset/german3.wav")}')
# print(f'JAPANESE6: {predict_accent("./out-min-n-20-max-t-60-augmented/processed_audio/dataset/japanese6.wav")}')


dataset_path = "./out-min-n-20-max-t-60-augmented"
df = pd.read_csv(os.path.join(dataset_path, "accent_data.csv"))
# accent_mapping = {k:v for (v, k) in enumerate(df["accent"].unique())}

model = HubertForSequenceClassification.from_pretrained(
    "./hubert-accent-classifier-final"
)
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "./hubert-accent-classifier-final"
)


# Move model to device
model.to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./hubert-accent-classifier",
    eval_strategy="no",
    eval_steps=100,
    per_device_train_batch_size=cfg.batch_size,
    per_device_eval_batch_size=cfg.batch_size,
    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    learning_rate=cfg.learning_rate,
    num_train_epochs=cfg.num_epochs,
    warmup_steps=cfg.warmup_steps,
    weight_decay=cfg.weight_decay,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="no",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),  # Use mixed precision training if available
    report_to="tensorboard",
)


# Define metrics computation function for Trainer
def compute_metrics(pred):
    predictions = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(pred.label_ids, predictions),
        "f1": f1_score(pred.label_ids, predictions, average="weighted"),
    }


data_collator = DataCollatorForAccentClassification(feature_extractor=feature_extractor)
# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    # train_dataset=train_dataset,
    # eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

test_dataset = AccentDataset(
    data_path=os.path.join(
        dataset_path, "accent_data.csv"
    ),  # Replace with your data path
    accent_mapping=accent_mapping,
    feature_extractor=feature_extractor,
    max_length=shared_cfg.max_length,
    split="test",
)

test_results = trainer.evaluate(test_dataset)
logger.info(f"Test results: {test_results}")
