from sklearn.metrics import f1_score
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from shared import SharedConfig, accent_mapping, dataset_path

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

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

shared_cfg = SharedConfig()


class Config:
    # Audio processing parameters
    n_fft = 512  # Increased from 400
    n_mels = 80  # Increased from 40
    n_mfcc = 40
    hop_length = int(0.010 * shared_cfg.sample_rate)
    win_length = int(0.025 * shared_cfg.sample_rate)
    
    # Training parameters
    epochs = 15  # Increased from 10
    batch_size = 16  # Increased from 16
    learning_rate = 0.0005  # More specific learning rate
    weight_decay = 1e-5  # Added weight decay for regularization
    
    # Early stopping parameters
    patience = 5
    
    # Data augmentation parameters
    time_shift_pct = 0.1
    spec_aug = True
    freq_mask_param = 10
    time_mask_param = 20
    
    # Model params
    dropout_rate = 0.5  # Increased dropout


cfg = Config()


class SpecAugment(nn.Module):
    """SpecAugment augmentation as described in the paper 
    'SpecAugment: A Simple Data Augmentation Method for ASR'"""
    
    def __init__(self, freq_mask_param, time_mask_param):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        
    def forward(self, spec):
        """
        spec: [batch_size, channels, n_mels, time]
        """
        if not self.training:
            return spec
            
        batch_size, channels, n_mels, time_steps = spec.shape
        
        # Apply frequency masking
        for i in range(batch_size):
            for _ in range(2):  # Apply twice
                f = np.random.randint(0, self.freq_mask_param)
                f0 = np.random.randint(0, n_mels - f)
                spec[i, :, f0:f0+f, :] = 0
        
        # Apply time masking
        for i in range(batch_size):
            for _ in range(2):  # Apply twice
                t = np.random.randint(0, self.time_mask_param)
                t0 = np.random.randint(0, time_steps - t)
                spec[i, :, :, t0:t0+t] = 0
                
        return spec


@dataclass
class DataCollatorForAccentClassification:
    def __call__(
        self, features: List[Tuple[torch.Tensor, str]]
    ) -> Dict[str, torch.Tensor]:
        # Extract MFCCs and labels
        mfccs = [f[0] for f in features]  # Each item is (mfcc, label)
        labels = torch.tensor([f[1] for f in features], dtype=torch.long)

        # Find max time length in batch
        max_len = max(mfcc.shape[-1] for mfcc in mfccs)

        # Pad all MFCCs to max_len along time dimension
        padded_mfccs = []
        for mfcc in mfccs:
            if mfcc.shape[-1] < max_len:
                pad_width = max_len - mfcc.shape[-1]
                mfcc = F.pad(mfcc, (0, pad_width))  # pad last dim only
            padded_mfccs.append(mfcc)

        batch_mfcc = torch.stack(padded_mfccs)  # Shape: [batch, 1, 40, max_len]
        return {"input_values": batch_mfcc, "labels": labels}


class AccentDataset(Dataset):
    def __init__(self, csv_file, split="train", transform=None, time_shift_pct=0.0):
        """
        Args:
            csv_file (str): Path to the full CSV file with file_path and accent columns.
            split (str): One of 'train', 'val', or 'test'.
            transform (callable, optional): MFCC transform to apply to audio.
            time_shift_pct (float): Percentage of audio length to randomly shift (only for training)
        """
        self.transform = transform
        self.data = pd.read_csv(csv_file)
        self.time_shift_pct = time_shift_pct if split == "train" else 0.0

        # Filter rows by split
        self.data = self.data[self.data["split"] == split].reset_index(drop=True)

        # Store file paths and labels
        self.file_paths = [Path(fp) for fp in self.data["file_path"]]
        self.labels = [accent_mapping[label] for label in self.data["accent"]]
        
        # Store class weights for weighted loss
        if split == "train":
            # Count occurrences of each accent
            accent_counts = self.data['accent'].value_counts()
            # Calculate weights (inverse of frequency)
            self.class_weights = {
                accent_mapping[accent]: len(self.data) / (len(accent_counts) * count) 
                for accent, count in accent_counts.items()
            }
            logger.info(f"Class weights: {self.class_weights}")

        logger.info(f"Loaded {len(self.data)} samples for {split} split")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.file_paths[idx])

        # Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
        if sample_rate != shared_cfg.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, shared_cfg.sample_rate)
            waveform = resampler(waveform)

        # Apply time shifting augmentation (for training only)
        if self.time_shift_pct > 0:
            shift_amount = int(waveform.shape[1] * self.time_shift_pct)
            if shift_amount > 0:
                shift = random.randint(-shift_amount, shift_amount)
                if shift > 0:
                    waveform = torch.cat([torch.zeros_like(waveform[:, :shift]), waveform[:, :-shift]], dim=1)
                elif shift < 0:
                    waveform = torch.cat([waveform[:, -shift:], torch.zeros_like(waveform[:, :shift])], dim=1)

        # Ensure waveform is 1D
        waveform = waveform.squeeze()

        mfcc = self.transform(waveform)  # shape: [1, 40, time]
        if mfcc.dim() == 2:  # shape: [40, T]
            mfcc = mfcc.unsqueeze(0)  # make it [1, 40, T]

        label = self.labels[idx]
        return mfcc, label


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ImprovedAccentCNN(nn.Module):
    def __init__(self, num_mfcc, num_classes, dropout_rate=0.5):
        super().__init__()
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        
        # Residual blocks
        self.res1 = ResidualBlock(32, 64, stride=1)
        self.pool2 = nn.MaxPool2d(2)
        
        self.res2 = ResidualBlock(64, 128, stride=1)
        self.pool3 = nn.MaxPool2d(2)
        
        self.res3 = ResidualBlock(128, 256, stride=1)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))  # Global pooling
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
        
        # SpecAugment for training
        self.spec_augment = SpecAugment(
            freq_mask_param=cfg.freq_mask_param,
            time_mask_param=cfg.time_mask_param
        )
        
    def forward(self, x):
        # Apply SpecAugment if in training mode
        if self.training and cfg.spec_aug:
            x = self.spec_augment(x)
            
        # Initial convolution
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Residual blocks
        x = self.pool2(self.res1(x))
        x = self.pool3(self.res2(x))
        x = self.pool4(self.res3(x))
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x


def evaluate_model(model, data_loader, device, criterion=None):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    
    # For confusion matrix
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(inputs)
            
            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / total if criterion is not None else None
    
    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "predictions": all_preds,
        "true_labels": all_labels
    }


def train_model(
    model,
    train_loader,
    val_loader=None,
    num_epochs=10,
    learning_rate=0.001,
    weight_decay=1e-5,
    patience=5,
    device=None,
    class_weights=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    
    # Create class weights tensor if provided
    if class_weights:
        weight_tensor = torch.tensor([class_weights.get(i, 1.0) for i in range(len(accent_mapping))],
                                    device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        logger.info(f"Using weighted CrossEntropyLoss with weights: {weight_tensor}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=3, verbose=True
    )
    
    # For early stopping
    best_val_acc = 0
    no_improvement = 0
    best_model_state = None

    
    total_steps = num_epochs * len(train_loader)
    global_progress = tqdm(total=total_steps, desc="Training Progress", position=0)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False, position=1)

        for batch in progress_bar:
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update tqdm description with running stats
            # avg_loss = running_loss / total
            # accuracy = 100.0 * correct / total

            progress_bar_loss = running_loss / total
            progress_bar_acc = 100.0 * correct / total
            progress_bar_f1 = f1_score(all_labels, all_preds, average='weighted')
            progress_bar.set_postfix({"loss": f"{progress_bar_loss:.4f}", "acc": f"{progress_bar_acc:.2f}%", "f1": f"{progress_bar_f1:.2f}%"})

            global_progress.update(1)
            # progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{accuracy:.2f}%"})

        # accuracy = 100.0 * correct / total

        global_progress.set_postfix({"epoch": epoch + 1, "train_acc": f"{progress_bar_acc:.2f}%"})
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {progress_bar_loss:.4f}, Accuracy: {progress_bar_acc:.2f}%, F1 Score: {progress_bar_f1:.4f}"
        )
        
        # logger.info(
        #     f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
        # )

        # Validate after each epoch
        if val_loader:
            val_results = evaluate_model(model, val_loader, device, criterion)
            val_loss = val_results["loss"]
            val_accuracy = val_results["accuracy"]
            
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
            )
            
            # Update learning rate based on validation loss
            scheduler.step(val_loss)
            
            # Check for early stopping
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                no_improvement = 0
                # Save best model state
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                logger.info(f"New best validation accuracy: {best_val_acc:.2f}%")
            else:
                no_improvement += 1
                logger.info(f"No improvement for {no_improvement} epochs")
                
                if no_improvement >= patience:
                    logger.info(f"Early stopping after {epoch + 1} epochs")
                    # Restore best model
                    model.load_state_dict(best_model_state)
                    break
    
    global_progress.close()
    
    # If we didn't early stop, but we have a best state, restore it
    if best_model_state is not None and no_improvement < patience:
        model.load_state_dict(best_model_state)
        logger.info(f"Training completed. Restoring best model with validation accuracy: {best_val_acc:.2f}%")
    
    return model


def analyze_results(model, test_loader, device):
    """Perform detailed analysis of model performance"""
    results = evaluate_model(model, test_loader, device)
    
    logger.info(f"Test accuracy: {results['accuracy']:.2f}%")
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get class names (reverse the accent_mapping)
    reverse_mapping = {v: k for k, v in accent_mapping.items()}
    class_names = [reverse_mapping[i] for i in range(len(accent_mapping))]
    
    # Confusion matrix
    cm = confusion_matrix(results['true_labels'], results['predictions'])
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    logger.info("Confusion matrix saved to confusion_matrix.png")
    
    # Classification report
    report = classification_report(results['true_labels'], results['predictions'], target_names=class_names)
    logger.info(f"Classification report:\n{report}")


def main():
    # Create MFCC transform
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=shared_cfg.sample_rate,
        n_mfcc=cfg.n_mfcc,
        melkwargs={
            "n_fft": cfg.n_fft,
            "hop_length": cfg.hop_length,
            "n_mels": cfg.n_mels,
        },
    )

    # Data collator
    collator = DataCollatorForAccentClassification()
    
    # Load datasets with time shifting augmentation for training
    train_dataset = AccentDataset(
        csv_file=os.path.join(dataset_path, "accent_data.csv"),
        split="train",
        transform=mfcc_transform,
        time_shift_pct=cfg.time_shift_pct
    )
    
    val_dataset = AccentDataset(
        csv_file=os.path.join(dataset_path, "accent_data.csv"),
        split="val",
        transform=mfcc_transform
    )
    
    test_dataset = AccentDataset(
        csv_file=os.path.join(dataset_path, "accent_data.csv"),
        split="test",
        transform=mfcc_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collator
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collator
    )

    # Create improved model
    model = ImprovedAccentCNN(
        num_mfcc=cfg.n_mfcc, 
        num_classes=len(accent_mapping),
        dropout_rate=cfg.dropout_rate
    )
    
    # Print model summary
    logger.info(f"Model architecture:\n{model}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Train model with class weights for imbalanced data
    logger.info("Starting training...")
    model = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
        device=device,
        class_weights=train_dataset.class_weights if hasattr(train_dataset, 'class_weights') else None
    )

    # Save model
    model_path = "./improved_accent_cnn_model2.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    # Evaluate and analyze model on test set
    logger.info("Evaluating model on test set...")
    try:
        analyze_results(model, test_loader, device)
    except ImportError:
        # If sklearn is not available, fall back to basic evaluation
        results = evaluate_model(model, test_loader, device)
        logger.info(f"Test accuracy: {results['accuracy']:.2f}%")


if __name__ == "__main__":
    main()
