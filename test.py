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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Tuple, Union
import logging
import random
from tqdm import tqdm
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

from shared import accent_mapping, dataset_path
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


def predict_accent(audio_path, model, feature_extractor):
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
    reverse_accent_mapping = {v: k for k, v in accent_mapping.items()}

    probs = logits.softmax(dim=1)[0].tolist()
    accent_probs = {
        reverse_accent_mapping[i]: round(prob * 100, 2) for i, prob in enumerate(probs)
    }
    accent_probs = dict(
        sorted(accent_probs.items(), key=lambda item: item[1], reverse=True)
    )

    predicted_accent = max(accent_probs, key=accent_probs.get)

    return predicted_accent, accent_probs


model_name = "./hubert-accent-classifier-final4"
df = pd.read_csv(os.path.join(dataset_path, "accent_data.csv"))
# accent_mapping = {k:v for (v, k) in enumerate(df["accent"].unique())}

model = HubertForSequenceClassification.from_pretrained(
    model_name
)
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_name
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



reverse_accent_mapping = {v: k for k, v in accent_mapping.items()}

preds = []
labels = []

for item in tqdm(test_dataset, "Testing"):
    inputs = {"input_values": item["input_values"].unsqueeze(0)}  # Add batch dim
    label = item["labels"].item()

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class = torch.argmax(logits, dim=1).item()
    labels.append(label)
    preds.append(predicted_class)
    # probs = logits.softmax(dim=1)[0].tolist()

    # accent_probs = {
    #     reverse_accent_mapping[i]: round(prob * 100, 2) for i, prob in enumerate(probs)
    # }
    # sorted_accent_probs = dict(sorted(accent_probs.items(), key=lambda item: item[1], reverse=True))
    # predicted = max(accent_probs, key=accent_probs.get)

    # print(
    #     f"PREDICTED: {predicted:10} | ACTUAL: {reverse_accent_mapping[label]:10} | {'CORRECT' if predicted_class == label else 'INCORRECT'}"
    # )
    # print("Accent probabilities (sorted):")
    # for accent, prob in sorted_accent_probs.items():
    #     print(f"  {accent:12}: {prob}%")

cm = confusion_matrix(labels, preds)
class_names = [reverse_accent_mapping[i] for i in range(len(accent_mapping))]
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("hubert_confusion_matrix.png")
    
test_results = trainer.evaluate(test_dataset)
logger.info(f"Test results: {test_results}")
