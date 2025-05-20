import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import (
    HubertModel,
    HubertForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoFeatureExtractor,
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Optional, Tuple, Union
import logging
import random
from tqdm import tqdm
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

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


model_name = "./hubert-accent-classifier-final5"
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


# # Define metrics computation function for Trainer
# def compute_metrics(pred):
#     # predictions = np.argmax(pred.predictions, axis=1)
#     logits = pred.predictions
#     labels = pred.label_ids

#     probs = softmax(logits, axis=1)
#     preds = np.argmax(probs, axis=1)

#     return {
#         "accuracy": accuracy_score(pred.label_ids, preds),
#         "precision": precision_score(pred.label_ids, preds, average='weighted'),
#         "recall": recall_score(pred.label_ids, preds, average='weighted'),
#         "f1": f1_score(pred.label_ids, preds, average='weighted'),
#         "roc_auc": roc_auc_score(labels, probs, multi_class='ovr')
#     }

# Define metrics computation function for Trainer
def compute_metrics(pred):
    logits = pred.predictions
    labels = pred.label_ids

    logits_tensor = torch.tensor(logits)
    probs = F.softmax(logits_tensor, dim=1).numpy()

    # probs = F.softmax(logits, dim=1)
    preds = np.argmax(probs, axis=1)
    predictions = np.argmax(pred.predictions, axis=1)

    return {
        "accuracy": accuracy_score(pred.label_ids, predictions),
        "precision": precision_score(pred.label_ids,predictions, average="weighted"),
        "recall": recall_score(pred.label_ids,predictions, average="weighted"),
        "f1": f1_score(pred.label_ids, predictions, average="weighted"),
        "predictions": preds,
        "true_labels": labels,
        "probabilities": probs,
    }


def plot_roc_curve(y_true, y_scores, class_names):
    """
    Plot ROC curve for multi-class classification
    
    Args:
        y_true: True labels (one-hot encoded)
        y_scores: Predicted probabilities
        class_names: List of class names
    """
    n_classes = len(class_names)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot all ROC curves
    plt.figure(figsize=(12, 9))
    
    # Plot micro-average ROC curve
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.2f})',
        color='deeppink',
        linestyle=':',
        linewidth=4,
    )
    
    # Plot ROC curves for all classes
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 
                    'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})',
        )
    
    # Plot diagonal line (represents random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curves - One-vs-Rest')
    plt.legend(loc="lower right")
    
    # Save the plot
    plt.savefig('hubert_roc_curve.png', dpi=300, bbox_inches='tight')
    logger.info("ROC curve saved to roc_curve.png")
    
    # Return AUC scores for logging
    return roc_auc


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
probs = []

class_names = [reverse_accent_mapping[i] for i in range(len(accent_mapping))]
# class_names = [reverse_accent_mapping[i] for i in range(len(accent_mapping))]
n_classes = len(class_names)

for item in tqdm(test_dataset, "Testing"):
    inputs = {"input_values": item["input_values"].unsqueeze(0)}  # Add batch dim
    label = item["labels"].item()

    with torch.no_grad():
      logits = model(**inputs).logits

    predicted_class = torch.argmax(logits, dim=1).item()
    labels.append(label)
    preds.append(predicted_class)
    probs.append(F.softmax(logits, dim=1)[0].cpu().numpy())

    # accent_probs = {
    #   reverse_accent_mapping[i]: round(prob * 100, 2) for i, prob in enumerate(probs)
    # }
    # sorted_accent_probs = dict(sorted(accent_probs.items(), key=lambda item: item[1], reverse=True))
    # predicted = max(accent_probs, key=accent_probs.get)

    # print(
    #   f"PREDICTED: {predicted:10} | ACTUAL: {reverse_accent_mapping[label]:10} | {'CORRECT' if predicted_class == label else 'INCORRECT'}"
    # )
    # print("Accent probabilities (sorted):")
    # for accent, prob in sorted_accent_probs.items():
    #   print(f"  {accent:12}: {prob}%")

cm = confusion_matrix(labels, preds)
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


y_test = np.array(labels)
y_score = np.array(probs)
y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
 
# Plot ROC curves and get AUC scores
roc_auc = plot_roc_curve(y_test_bin, y_score, class_names)

# Log AUC scores
logger.info("ROC AUC scores:")
for i, class_name in enumerate(class_names):
    logger.info(f"{class_name}: {roc_auc[i]:.4f}")
logger.info(f"Micro-average: {roc_auc['micro']:.4f}")
    
test_results = trainer.evaluate(test_dataset)
logger.info(f"Test results: {test_results}")
