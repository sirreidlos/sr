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

from shared import shared_cfg, accent_mapping
from cnn import (
    AccentCNN,
    AccentDataset,
    cfg,
    DataCollatorForAccentClassification,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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


def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(inputs)
            logger.info(labels, outputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    logger.info(f"Test results: {accuracy}")


def main():
    reverse_accent_mapping = {v: k for k, v in accent_mapping.items()}

    logger.info("Loading the model from ./accent_cnn_model.pth")
    model = AccentCNN(cfg.n_mfcc, num_classes=len(accent_mapping))
    model.load_state_dict(torch.load("./accent_cnn_model.pth"))
    model.to(device)

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

    logger.info("Evaluating model on test set...")
    test_dataset = AccentDataset(
        csv_file="./out-min-n-20-max-t-60-augmented/accent_data.csv",
        split="test",
        transform=mfcc_transform,
        max_length=shared_cfg.max_length,
    )

    for mfcc, label in test_dataset:
        if mfcc.shape[-1] < shared_cfg.max_length:
            pad_width = shared_cfg.max_length - mfcc.shape[-1]
            mfcc = F.pad(mfcc, (0, pad_width))  # pad last dim only

        model.eval()
        outputs = model(mfcc.unsqueeze(0))

        probs = F.softmax(outputs, dim=1).squeeze()  # shape: [num_classes]

        accent_probs = {
            reverse_accent_mapping[i]: round(prob.item() * 100, 2)
            for i, prob in enumerate(probs)
        }
        sorted_accent_probs = dict(sorted(accent_probs.items(), key=lambda item: item[1], reverse=True))

        # accent_probs = {reverse_accent_mapping[i]: round(prob.item() * 100, 2) for i, prob in enumerate(probs)}

        _, predicted = outputs.max(1)
        print(
            f"PREDICTED: {reverse_accent_mapping.get(predicted.item()):10} | ACTUAL: {reverse_accent_mapping.get(label):10} | {'CORRECT' if predicted.item() == label else 'INCORRECT'}"
        )
        print("Accent probabilities (sorted):")
        for accent, prob in sorted_accent_probs.items():
            print(f"  {accent:12}: {prob}%")

    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, collate_fn=collator
    )

    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()
