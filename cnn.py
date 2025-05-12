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
    n_fft = 400
    n_mels = 40
    n_mfcc = 40
    hop_length = int(0.010 * shared_cfg.sample_rate)
    win_length = int(0.025 * shared_cfg.sample_rate)
    num_frames = 1 + (shared_cfg.max_length - win_length) // hop_length
    epochs = 10
    batch_size = 16


cfg = Config()


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
    def __init__(self, csv_file, split="train", transform=None):
        """
        Args:
            csv_file (str): Path to the full CSV file with file_path and accent columns.
            split (str): One of 'train', 'val', or 'test'.
            transform (callable, optional): MFCC transform to apply to audio.
        """
        self.transform = transform
        self.data = pd.read_csv(csv_file)

        # Filter rows by split
        self.data = self.data[self.data["split"] == split].reset_index(drop=True)

        # Store file paths and labels
        self.file_paths = [Path(fp) for fp in self.data["file_path"]]
        self.labels = [accent_mapping[label] for label in self.data["accent"]]

        logger.info(f"Loaded {len(self.data)} samples for {split} split")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.file_paths[idx])

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
        if sample_rate != shared_cfg.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, shared_cfg.sample_rate)
            waveform = resampler(waveform)

        # Ensure waveform is 1D
        waveform = waveform.squeeze()

        mfcc = self.transform(waveform)  # shape: [1, 40, time]
        if mfcc.dim() == 2:  # shape: [40, T]
            mfcc = mfcc.unsqueeze(0)  # make it [1, 40, T]

        label = self.labels[idx]
        return mfcc, label


class AccentCNN(nn.Module):
    def __init__(self, num_mfcc, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AdaptiveMaxPool2d((1, 1))  # Global pooling

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.fc(x)
        return x


def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    logger.info(f"Test results: {accuracy}")


def train_model(
    model,
    train_loader,
    val_loader=None,
    num_epochs=10,
    learning_rate=0.001,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in progress_bar:
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            assert inputs.ndim == 4, (
                f"Expected input shape [B, 1, 40, T], got {inputs.shape}"
            )
            assert labels.ndim == 1, f"Expected label shape [B], got {labels.shape}"

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update tqdm description with running stats
            avg_loss = running_loss / total
            accuracy = 100.0 * correct / total
            progress_bar.set_postfix({"loss": avg_loss, "acc": f"{accuracy:.2f}%"})

        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} - Final Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

        if val_loader:
            evaluate_model(model, val_loader, device)


def main():
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=shared_cfg.sample_rate,
        n_mfcc=cfg.n_mfcc,
        melkwargs={
            "n_fft": cfg.n_fft,
            "hop_length": cfg.hop_length,
            "n_mels": cfg.n_mels,
        },
    )

    collator = DataCollatorForAccentClassification()
    train_dataset = AccentDataset(
        csv_file=os.path.join(dataset_path, "accent_data.csv"),
        split="train",
        transform=mfcc_transform,
    )
    val_dataset = AccentDataset(
        csv_file=os.path.join(dataset_path, "accent_data.csv"),
        split="val",
        transform=mfcc_transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collator
    )

    model = AccentCNN(num_mfcc=cfg.n_mfcc, num_classes=len(accent_mapping))
    logger.info("Starting training...")
    train_model(model, train_loader, val_loader, num_epochs=cfg.epochs)

    torch.save(model.state_dict(), "./accent_cnn_model2.pth")
    logger.info("Model saved to ./accent_cnn_model2.pth")

    logger.info("Evaluating model on test set...")
    test_dataset = AccentDataset(
        csv_file=os.path.join(dataset_path, "accent_data.csv"),
        split="test",
        transform=mfcc_transform,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collator
    )

    evaluate_model(model, test_loader, device)
    # logger.info(f"Test results: {test_results}")


if __name__ == "__main__":
    main()
