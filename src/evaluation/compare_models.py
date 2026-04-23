"""
Model Comparison Script for Deepfake Detection.

Loads all 4 trained model checkpoints, evaluates them on the SAME test set,
and produces a side-by-side comparison table and grouped bar chart.

Models compared:
  1. CNN-Only     (ResNet-50, frame averaging)
  2. CNN+LSTM V2  (ResNet-50 + Unidirectional LSTM)
  3. ViT-Only     (ViT-B/16, frame averaging)
  4. ST-ViT       (ResNet-50 + BiLSTM + ViT Encoder) — Our novel model
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.training.train_cnn_lstm import VideoSequenceDataset

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = "dataset_sequences"
IMG_SIZE = 224
SEQ_LEN = 20
BATCH_SIZE = 2
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()


# ─── Evaluate a single model ────────────────────────────────────────────────
def evaluate_model(model, loader, num_samples):
    """Run inference and return all metrics."""
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for sequences, labels in tqdm(loader, desc="  Evaluating", leave=False):
            sequences = sequences.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with autocast("cuda", enabled=USE_AMP):
                logits = model(sequences)

            probs = torch.sigmoid(logits.float()).squeeze()
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)

            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs.detach().cpu().numpy().flatten())

    all_preds = [1 if p > 0.5 else 0 for p in all_probs]

    metrics = {}
    metrics["accuracy"] = accuracy_score(all_labels, all_preds)
    try: metrics["f1"] = f1_score(all_labels, all_preds)
    except: metrics["f1"] = 0.0
    try: metrics["precision"] = precision_score(all_labels, all_preds)
    except: metrics["precision"] = 0.0
    try: metrics["recall"] = recall_score(all_labels, all_preds)
    except: metrics["recall"] = 0.0
    try: metrics["auc"] = roc_auc_score(all_labels, all_probs)
    except: metrics["auc"] = 0.0

    total_params = sum(p.numel() for p in model.parameters())
    metrics["params_m"] = total_params / 1e6

    return metrics


# ─── Load all models ────────────────────────────────────────────────────────
def load_all_models():
    """Load all 4 model checkpoints. Returns dict of {name: model}."""
    models_dict = {}

    # 1. CNN-Only
    checkpoint = "best_cnn_only_model.pth"
    if os.path.exists(checkpoint):
        from train_cnn_only import CNN_Only
        model = CNN_Only(freeze_cnn=False).to(DEVICE)
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        models_dict["CNN-Only"] = model
        print(f"  Loaded: {checkpoint}")
    else:
        print(f"  MISSING: {checkpoint} — run train_cnn_only.py first")

    # 2. CNN+LSTM V2
    checkpoint = "best_cnn_lstm_v2_model.pth"
    if os.path.exists(checkpoint):
        from train_cnn_lstm_v2 import CNN_LSTM_V2
        model = CNN_LSTM_V2(freeze_cnn=False).to(DEVICE)
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        models_dict["CNN+LSTM"] = model
        print(f"  Loaded: {checkpoint}")
    else:
        print(f"  MISSING: {checkpoint} — run train_cnn_lstm_v2.py first")

    # 3. ViT-Only
    checkpoint = "best_vit_only_model.pth"
    if os.path.exists(checkpoint):
        from train_vit_only import ViT_Only
        model = ViT_Only(freeze_backbone=False).to(DEVICE)
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        models_dict["ViT-Only"] = model
        print(f"  Loaded: {checkpoint}")
    else:
        print(f"  MISSING: {checkpoint} — run train_vit_only.py first")

    # 4. ST-ViT (Our novel model)
    checkpoint = "best_st_vit_model.pth"
    if os.path.exists(checkpoint):
        from st_vit_model import ST_ViT
        model = ST_ViT(
            seq_len=SEQ_LEN, lstm_hidden=256, lstm_layers=2,
            vit_dim=512, vit_depth=4, vit_heads=8, freeze_cnn=False
        ).to(DEVICE)
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        models_dict["ST-ViT (Ours)"] = model
        print(f"  Loaded: {checkpoint}")
    else:
        print(f"  MISSING: {checkpoint} — run train_st_vit.py first")

    return models_dict


# ─── Print comparison table ─────────────────────────────────────────────────
def print_comparison_table(results):
    """Print a formatted comparison table to the console."""
    metrics = ["accuracy", "f1", "precision", "recall", "auc", "params_m"]
    labels = ["Accuracy", "F1 Score", "Precision", "Recall", "AUC-ROC", "Params (M)"]

    model_names = list(results.keys())

    # Header
    header = f"{'Metric':<14}"
    for name in model_names:
        header += f" | {name:>15}"
    print("\n" + "=" * len(header))
    print("  MODEL COMPARISON — DEEPFAKE DETECTION")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # Rows
    for metric, label in zip(metrics, labels):
        row = f"{label:<14}"
        values = [results[name].get(metric, 0) for name in model_names]

        for i, (name, val) in enumerate(zip(model_names, values)):
            if metric == "params_m":
                cell = f"{val:.1f}M"
            else:
                cell = f"{val:.4f}"

            # Bold the best value (highest metric, excluding params)
            if metric != "params_m" and val == max(values):
                cell = f"*{cell}*"

            row += f" | {cell:>15}"
        print(row)

    print("=" * len(header))
    print("  * = Best performing model for that metric")
    print()


# ─── Plot comparison chart ───────────────────────────────────────────────────
def plot_comparison(results):
    """Create a grouped bar chart comparing all models."""
    model_names = list(results.keys())
    metrics = ["accuracy", "f1", "precision", "recall", "auc"]
    metric_labels = ["Accuracy", "F1 Score", "Precision", "Recall", "AUC-ROC"]

    x = np.arange(len(metrics))
    width = 0.18
    n_models = len(model_names)

    # Color palette
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (name, color) in enumerate(zip(model_names, colors[:n_models])):
        values = [results[name].get(m, 0) for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name, color=color,
                      edgecolor="white", linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Metric", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Deepfake Detection — Model Comparison\n(Same Dataset, Same Training Config)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=200, bbox_inches="tight")
    print("Comparison chart saved to model_comparison.png")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Deepfake Detection — Model Comparison")
    print("=" * 60)

    if not os.path.exists(DATA_DIR):
        print(f"ERROR: '{DATA_DIR}' not found.")
        return

    # Load test data
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_dataset = VideoSequenceDataset(DATA_DIR, "test", SEQ_LEN, val_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()
    )
    num_test = len(test_dataset)
    print(f"\nTest set: {num_test} videos")

    # Load all models
    print("\nLoading model checkpoints...")
    models_dict = load_all_models()

    if len(models_dict) == 0:
        print("\nNo models found! Train at least one model first.")
        return

    # Evaluate all models on the same test set
    print(f"\nEvaluating {len(models_dict)} models on the test set...\n")
    results = {}
    for name, model in models_dict.items():
        print(f"  [{name}]")
        results[name] = evaluate_model(model, test_loader, num_test)

        # Free GPU memory between models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print comparison table
    print_comparison_table(results)

    # Plot comparison chart
    plot_comparison(results)

    print("Done! Open model_comparison.png to see the visual comparison.")


if __name__ == "__main__":
    main()
