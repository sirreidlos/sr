"""
HuBERT Fine-tuning for Accent Classification

This script demonstrates how to fine-tune a pre-trained HuBERT model for accent classification.
It includes data preparation, model configuration, training, and evaluation.
"""

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
    AutoFeatureExtractor
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Tuple, Union
import logging
import random
from dataclasses import dataclass
from shared import SharedConfig, accent_mapping, dataset_path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define configuration
class Config:
    model_name = "facebook/hubert-base-ls960"  # Pre-trained HuBERT model
    batch_size = 8
    learning_rate = 5e-5
    num_epochs = 5
    gradient_accumulation_steps = 2
    warmup_steps = 500
    weight_decay = 0.01

cfg = Config()

shared_cfg = SharedConfig()

# Custom dataset for accent classification

class AccentDataset(Dataset):
    def __init__(self, data_path, accent_mapping, feature_extractor, max_length=160000, split="train"):
        """
        Args:
            data_path: Path to the CSV file with audio file paths and accent labels
            accent_mapping: Dictionary mapping accent names to numerical indices
            feature_extractor: HuBERT feature extractor
            max_length: Maximum sequence length
            split: 'train', 'val', or 'test'
        """
        self.data = pd.read_csv(data_path)
        self.accent_mapping = accent_mapping
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.split = split
        
        # Filter data for the specific split if specified in the CSV
        if 'split' in self.data.columns:
            self.data = self.data[self.data['split'] == split].reset_index(drop=True)
            
        logger.info(f"Loaded {len(self.data)} samples for {split} split")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        audio_path = self.data.iloc[idx]['file_path']
        accent = self.data.iloc[idx]['accent']
        label = self.accent_mapping[accent]
        
        # Load and preprocess audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sample_rate != shared_cfg.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, shared_cfg.sample_rate)
            waveform = resampler(waveform)
        
        # Ensure waveform is 1D
        waveform = waveform.squeeze()
        
        # Pad or truncate to max_length
        if waveform.shape[0] < self.max_length:
            # Pad with zeros
            padding = self.max_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            # Truncate
            waveform = waveform[:self.max_length]
        
        # Create input features
        inputs = self.feature_extractor(
            waveform, 
            sampling_rate=shared_cfg.sample_rate, 
            padding="max_length", 
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Prepare output dictionary
        return {
            "input_values": inputs.input_values.squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Collator for batching samples
@dataclass
class DataCollatorForAccentClassification:
    feature_extractor: AutoFeatureExtractor
    padding: Union[bool, str] = True
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_values = [{"input_values": feature["input_values"]} for feature in features]
        batch = self.feature_extractor.pad(
            input_values,
            padding=self.padding,
            return_tensors="pt",
        )
        
        # Add labels
        batch["labels"] = torch.tensor([feature["labels"] for feature in features])
        
        return batch

# Define metrics computation function for Trainer
def compute_metrics(pred):
    predictions = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(pred.label_ids, predictions),
        "f1": f1_score(pred.label_ids, predictions, average='weighted')
    }

def main():
    
    # Define accent mapping (example - modify according to your dataset)
    num_labels = len(accent_mapping)
    
    # Load feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.model_name)
    
    # Create data collator
    data_collator = DataCollatorForAccentClassification(feature_extractor=feature_extractor)
    
    # Load datasets
    train_dataset = AccentDataset(
        data_path=os.path.join(dataset_path, "accent_data.csv"),
        accent_mapping=accent_mapping,
        feature_extractor=feature_extractor,
        max_length=shared_cfg.max_length,
        split="train"
    )
    
    eval_dataset = AccentDataset(
        data_path=os.path.join(dataset_path, "accent_data.csv"),
        accent_mapping=accent_mapping,
        feature_extractor=feature_extractor,
        max_length=shared_cfg.max_length,
        split="val"
    )
    
    # Load HuBERT model for sequence classification
    model = HubertForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=num_labels,
        gradient_checkpointing=True  # Enable gradient checkpointing to save memory
    )
    
    # Move model to device
    model.to(device)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./hubert-accent-classifier",
        eval_strategy="steps",
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
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available(),  # Use mixed precision training if available
        report_to="tensorboard"
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the model
    model.save_pretrained("./hubert-accent-classifier-final")
    feature_extractor.save_pretrained("./hubert-accent-classifier-final")
    logger.info("Model saved to ./hubert-accent-classifier-final")
    
    # Evaluate model on test set
    logger.info("Evaluating model on test set...")
    test_dataset = AccentDataset(
        data_path=os.path.join(dataset_path, "accent_data.csv"),  # Replace with your data path
        accent_mapping=accent_mapping,
        feature_extractor=feature_extractor,
        max_length=shared_cfg.max_length,
        split="test"
    )
    
    test_results = trainer.evaluate(test_dataset)
    logger.info(f"Test results: {test_results}")

# Example of prediction function
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
        waveform.squeeze().numpy(), 
        sampling_rate=16000, 
        return_tensors="pt"
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
    predicted_accent = reverse_accent_mapping[predicted_class]
    return predicted_accent, logits.softmax(dim=1)[0].tolist()

if __name__ == "__main__":
    main()
