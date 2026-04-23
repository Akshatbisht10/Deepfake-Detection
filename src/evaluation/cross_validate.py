"""
5-Fold Stratified Cross-Validation for Deepfake Detection Models.

For research paper publication: reports mean +/- std across all folds
for Accuracy, F1, Precision, Recall, and AUC-ROC.

Evaluates: CNN-Only, CNN+LSTM, ViT-Only, ST-ViT (Ours)
"""

import os, copy, time, gc, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, Subset
from torch.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.st_vit_model import ST_ViT
from src.training.train_cnn_only import CNN_Only
from src.training.train_cnn_lstm_v2 import CNN_LSTM_V2
from src.training.train_vit_only import ViT_Only

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = "dataset_sequences"
IMG_SIZE = 224
SEQ_LEN = 20
BATCH_SIZE = 2
ACCUM_STEPS = 4
LR = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
EPOCHS = 15           # Slightly fewer for CV efficiency
UNFREEZE_EPOCH = 4
NUM_WORKERS = 0
N_FOLDS = 5
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

# Which models to evaluate (set False to skip slow ones)
RUN_MODELS = {
    "CNN-Only": False,        # Already done — results in cv_results.json
    "CNN+LSTM": False,        # Already done — results in cv_results.json
    "ViT-Only": True,         # Running now
    "ST-ViT (Ours)": False,   # Already done — results in cv_results.json
}


# ─── Dataset that accepts explicit sample list ──────────────────────────────
class VideoSequenceDatasetCV(Dataset):
    """Dataset that takes explicit list of (video_dir, label) pairs."""

    def __init__(self, samples, seq_len=SEQ_LEN, transform=None):
        self.samples = samples
        self.seq_len = seq_len
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_dir, label = self.samples[idx]
        frame_files = sorted(
            [f for f in os.listdir(video_dir) if f.endswith(".jpg")],
            key=lambda x: int(os.path.splitext(x)[0]),
        )
        frames = []
        for fname in frame_files[:self.seq_len]:
            img = Image.open(os.path.join(video_dir, fname)).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        if len(frames) < self.seq_len:
            pad = torch.zeros_like(frames[0]) if frames else torch.zeros(3, IMG_SIZE, IMG_SIZE)
            while len(frames) < self.seq_len:
                frames.append(pad)

        sequence = torch.stack(frames)
        return sequence, torch.tensor(label, dtype=torch.float32)


# ─── Collect ALL video sequences ────────────────────────────────────────────
def collect_all_samples():
    """Pool all videos from train/val/test into one list."""
    LABEL_MAP = {"real": 0, "fake": 1}
    all_samples = []

    for split in ["train", "val", "test"]:
        for label_name in ["real", "fake"]:
            label_dir = os.path.join(DATA_DIR, split, label_name)
            if not os.path.isdir(label_dir):
                continue
            for video_name in sorted(os.listdir(label_dir)):
                video_dir = os.path.join(label_dir, video_name)
                if os.path.isdir(video_dir):
                    jpgs = [f for f in os.listdir(video_dir) if f.endswith(".jpg")]
                    if len(jpgs) > 0:
                        all_samples.append((video_dir, LABEL_MAP[label_name]))

    return all_samples


# ─── Label Smoothing Loss ────────────────────────────────────────────────────
class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        smooth = targets * (1 - self.smoothing) + self.smoothing / 2
        return self.bce(logits, smooth)


# ─── Transforms ─────────────────────────────────────────────────────────────
def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ─── Model factory ──────────────────────────────────────────────────────────
def create_model(name):
    if name == "CNN-Only":
        return CNN_Only(freeze_cnn=True)
    elif name == "CNN+LSTM":
        return CNN_LSTM_V2(freeze_cnn=True)
    elif name == "ViT-Only":
        return ViT_Only(freeze_backbone=True)
    elif name == "ST-ViT (Ours)":
        return ST_ViT(seq_len=SEQ_LEN, lstm_hidden=256, lstm_layers=2,
                      vit_dim=512, vit_depth=4, vit_heads=8, freeze_cnn=True)


def unfreeze_model(model, name):
    if name == "CNN-Only":
        model.unfreeze_cnn()
    elif name == "CNN+LSTM":
        model.unfreeze_cnn()
    elif name == "ViT-Only":
        model.unfreeze_backbone()
    elif name == "ST-ViT (Ours)":
        model.unfreeze_cnn()


def get_param_groups(model, name, lr_backbone, lr_rest):
    if name == "ViT-Only":
        return model.get_parameter_groups(lr_backbone, lr_rest)
    else:
        return model.get_parameter_groups(lr_backbone, lr_rest)


# ─── Train one fold ─────────────────────────────────────────────────────────
def train_one_fold(model, model_name, train_loader, val_loader, train_size, val_size):
    """Train model for EPOCHS, return best model state dict."""
    criterion = LabelSmoothingBCE(LABEL_SMOOTHING)
    scaler = GradScaler("cuda", enabled=USE_AMP)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

    best_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    for epoch in range(EPOCHS):
        # Progressive unfreezing
        if epoch == UNFREEZE_EPOCH:
            unfreeze_model(model, model_name)
            optimizer = optim.AdamW(
                get_param_groups(model, model_name, LR / 10, LR),
                weight_decay=WEIGHT_DECAY,
            )
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
            scaler = GradScaler("cuda", enabled=USE_AMP)

        # Train phase
        model.train()
        optimizer.zero_grad()
        accum_count = 0

        for sequences, labels in tqdm(train_loader, desc=f"  ep{epoch}", leave=False):
            sequences = sequences.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True).unsqueeze(1)

            with autocast("cuda", enabled=USE_AMP):
                logits = model(sequences)
                loss = criterion(logits, labels) / ACCUM_STEPS

            scaler.scale(loss).backward()
            accum_count += 1
            if accum_count % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        if accum_count % ACCUM_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        scheduler.step(epoch)

        # Val phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True).unsqueeze(1)
                with autocast("cuda", enabled=USE_AMP):
                    logits = model(sequences)
                    loss = nn.BCEWithLogitsLoss()(logits, labels)
                val_loss += loss.item() * sequences.size(0)

        val_loss /= val_size
        if val_loss < best_loss:
            best_loss = val_loss
            best_wts = copy.deepcopy(model.state_dict())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.load_state_dict(best_wts)
    return model


# ─── Evaluate on test fold ──────────────────────────────────────────────────
def evaluate_fold(model, test_loader):
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(DEVICE, non_blocking=True)
            with autocast("cuda", enabled=USE_AMP):
                logits = model(sequences)
            probs = torch.sigmoid(logits.float()).squeeze()
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
            all_labels.extend(labels.numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())

    preds = [1 if p > 0.5 else 0 for p in all_probs]
    metrics = {
        "accuracy": accuracy_score(all_labels, preds),
        "f1": f1_score(all_labels, preds, zero_division=0),
        "precision": precision_score(all_labels, preds, zero_division=0),
        "recall": recall_score(all_labels, preds, zero_division=0),
    }
    try:
        metrics["auc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        metrics["auc"] = 0.0
    return metrics


# ─── Main Cross-Validation Loop ─────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  5-Fold Stratified Cross-Validation — Deepfake Detection")
    print("  For Research Paper Publication")
    print("=" * 70)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        gpu = torch.cuda.get_device_name(0)
        print(f"  GPU: {gpu} | FP16: Enabled")

    # Collect all samples
    all_samples = collect_all_samples()
    labels_array = np.array([s[1] for s in all_samples])
    n_real = int((labels_array == 0).sum())
    n_fake = int((labels_array == 1).sum())
    print(f"\n  Total videos: {len(all_samples)} (real: {n_real}, fake: {n_fake})")
    print(f"  Folds: {N_FOLDS} | Epochs/fold: {EPOCHS}")

    train_tf, val_tf = get_transforms()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Results storage — load existing results so new models merge in
    all_results = {}
    if os.path.exists("cv_results.json"):
        with open("cv_results.json", "r") as f:
            all_results = json.load(f)
        print(f"\n  Loaded existing results for: {list(all_results.keys())}")
    models_to_run = [name for name, run in RUN_MODELS.items() if run]

    for model_name in models_to_run:
        print(f"\n{'=' * 70}")
        print(f"  MODEL: {model_name}")
        print(f"{'=' * 70}")

        fold_metrics = []
        fold_start = time.time()

        for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(np.zeros(len(all_samples)), labels_array)):
            print(f"\n  --- Fold {fold_idx + 1}/{N_FOLDS} ---")

            # Split train_val into train (90%) and val (10%)
            np.random.seed(SEED + fold_idx)
            np.random.shuffle(train_val_idx)
            val_split = int(len(train_val_idx) * 0.1)
            val_idx = train_val_idx[:val_split]
            train_idx = train_val_idx[val_split:]

            train_samples = [all_samples[i] for i in train_idx]
            val_samples = [all_samples[i] for i in val_idx]
            test_samples = [all_samples[i] for i in test_idx]

            print(f"    Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

            train_ds = VideoSequenceDatasetCV(train_samples, SEQ_LEN, train_tf)
            val_ds = VideoSequenceDatasetCV(val_samples, SEQ_LEN, val_tf)
            test_ds = VideoSequenceDatasetCV(test_samples, SEQ_LEN, val_tf)

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                      num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                     num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

            # Create fresh model
            model = create_model(model_name).to(DEVICE)

            # Train
            model = train_one_fold(model, model_name, train_loader, val_loader,
                                   len(train_samples), len(val_samples))

            # Evaluate on held-out test fold
            metrics = evaluate_fold(model, test_loader)
            fold_metrics.append(metrics)
            print(f"    Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | "
                  f"AUC: {metrics['auc']:.4f}")

            # Cleanup
            del model, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        elapsed = time.time() - fold_start
        print(f"\n  {model_name} completed in {elapsed / 60:.1f} minutes")

        # Aggregate fold results
        metric_names = ["accuracy", "f1", "precision", "recall", "auc"]
        result = {}
        for m in metric_names:
            values = [fm[m] for fm in fold_metrics]
            result[f"{m}_mean"] = float(np.mean(values))
            result[f"{m}_std"] = float(np.std(values))
            result[f"{m}_values"] = values

        all_results[model_name] = result

        # Print per-model summary
        print(f"\n  {model_name} -- 5-Fold Summary:")
        for m in metric_names:
            mean = result[f"{m}_mean"]
            std = result[f"{m}_std"]
            print(f"    {m:>12s}: {mean:.4f} +/- {std:.4f}")

        # Save incrementally so results aren't lost if interrupted
        with open("cv_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  [Saved incremental results to cv_results.json]")

    # ─── Final Comparison Table ──────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  FINAL RESULTS: 5-Fold Stratified Cross-Validation")
    print("=" * 90)

    metric_labels = ["Accuracy", "F1 Score", "Precision", "Recall", "AUC-ROC"]
    metric_keys = ["accuracy", "f1", "precision", "recall", "auc"]
    names = list(all_results.keys())

    header = f"{'Metric':<14}"
    for name in names:
        header += f" | {name:>22}"
    print(header)
    print("-" * len(header))

    for label, key in zip(metric_labels, metric_keys):
        row = f"{label:<14}"
        for name in names:
            mean = all_results[name][f"{key}_mean"]
            std = all_results[name][f"{key}_std"]
            row += f" | {mean:.4f} +/- {std:.4f}  "
        print(row)
    print("=" * len(header))

    # ─── Save results to JSON ────────────────────────────────────────────
    with open("cv_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to cv_results.json")

    # ─── Plot comparison ─────────────────────────────────────────────────
    plot_cv_results(all_results)


def plot_cv_results(results):
    """Grouped bar chart with error bars for cross-validation results."""
    names = list(results.keys())
    metrics = ["accuracy", "f1", "precision", "recall", "auc"]
    labels = ["Accuracy", "F1", "Precision", "Recall", "AUC-ROC"]

    x = np.arange(len(metrics))
    width = 0.18
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (name, color) in enumerate(zip(names, colors[:len(names)])):
        means = [results[name][f"{m}_mean"] for m in metrics]
        stds = [results[name][f"{m}_std"] for m in metrics]
        offset = (i - len(names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, label=name,
                      color=color, edgecolor="white", linewidth=0.5,
                      capsize=3, error_kw={"linewidth": 1.5})
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xlabel("Metric", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Deepfake Detection — 5-Fold Cross-Validation Comparison\n(mean +/- std)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("cv_model_comparison.png", dpi=200, bbox_inches="tight")
    print("Cross-validation comparison chart saved to cv_model_comparison.png")


if __name__ == "__main__":
    main()
