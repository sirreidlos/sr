import argparse
import os
from shared import accent_mapping
from cnn3 import SimplerAccentCNN, EnhancedMFCC
import logging
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import (
    HubertModel,
    HubertForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoFeatureExtractor,
)

SAMPLE_RATE = 16000
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = int(0.010 * SAMPLE_RATE)
N_MELS = 40
DROPOUT_RATE = 0.2

reverse_accent_mapping = {v: k for k, v in accent_mapping.items()}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


def pred_cnn(model_path, input_path):
    logger.info(f"Loading the CNN model from {model_path}")
    model = SimplerAccentCNN(
        input_channels=3,
        num_mfcc=N_MFCC,
        num_classes=len(accent_mapping),
        dropout_rate=DROPOUT_RATE,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    mfcc_transform = EnhancedMFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )

    waveform, sample_rate = torchaudio.load(input_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
        waveform = resampler(waveform)

    waveform = waveform.squeeze()
    mfcc = mfcc_transform(waveform)
    if mfcc.dim() == 2:
        mfcc = mfcc.unsqueeze(0)

    model.eval()
    outputs = model(mfcc.unsqueeze(0))
    probs = F.softmax(outputs, dim=1).squeeze()  # shape: [num_classes]
    accent_probs = {
        reverse_accent_mapping[i]: round(prob.item() * 100, 2)
        for i, prob in enumerate(probs)
    }
    sorted_accent_probs = dict(
        sorted(accent_probs.items(), key=lambda item: item[1], reverse=True)
    )
    print("Accent probabilities (sorted):")
    for accent, prob in sorted_accent_probs.items():
        print(f"  {accent:12}: {prob}%")


def pred_hubert(model_path, input_path):
    logger.info(f"Loading the HuBERT model from {model_path}")
    model = HubertForSequenceClassification.from_pretrained(model_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)

    waveform, sample_rate = torchaudio.load(input_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
        waveform = resampler(waveform)

    inputs = feature_extractor(
        waveform.squeeze().numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = logits.softmax(dim=1)[0].tolist()
    accent_probs = {
        reverse_accent_mapping[i]: round(prob * 100, 2) for i, prob in enumerate(probs)
    }
    accent_probs = dict(
        sorted(accent_probs.items(), key=lambda item: item[1], reverse=True)
    )

    sorted_accent_probs = dict(
        sorted(accent_probs.items(), key=lambda item: item[1], reverse=True)
    )
    print("Accent probabilities (sorted):")
    for accent, prob in sorted_accent_probs.items():
        print(f"  {accent:12}: {prob}%")


def main(args):
    logger.info(f"Running prediction for {args.input}")
    if args.cnn:
        pred_cnn(args.cnn, args.input)

    if args.hubert:
        pred_hubert(args.hubert, args.input)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test for single audio")

    parser.add_argument("--hubert", type=str, default=None, help="Path to HuBERT model")
    parser.add_argument("--cnn", type=str, default=None, help="Path to CNN model")
    parser.add_argument("--input", type=str, default=None, help="Path to audio input")

    args = parser.parse_args()

    if not args.input:
        parser.error("No input given.")

    main(args)
