"""
ST-ViT (Spatiotemporal Vision Transformer) Training Script — V2 (Improved).

Improvements over V1:
  - 7x more training data (363 videos/class vs 50)
  - Stronger data augmentation (RandomErasing, RandomAffine, GaussianBlur)
  - Cosine annealing learning rate scheduler
  - Weight decay (AdamW) for L2 regularization
  - Label smoothing (0.1) to reduce overconfidence
  - Gradient accumulation with effective batch = 8

Architecture:
  Stage 1 - CNN       : ResNet-50 (pretrained) -> 2048-dim feature per frame
  Stage 2 - BiLSTM    : 2-layer bidirectional LSTM -> 512-dim temporal tokens
  Stage 3 - ViT       : 4-layer Transformer encoder -> global context
  Stage 4 - Classifier: FC(512->256) -> ReLU -> Dropout(0.5) -> FC(256->1)

Training Strategy:
  - Epochs 0-4 : CNN frozen, train LSTM + ViT + classifier only
  - Epoch 5+   : Unfreeze CNN, fine-tune end-to-end (CNN at LR/10)
  - Optimizer  : AdamW with weight decay + cosine annealing LR
  - Loss       : BCEWithLogitsLoss with label smoothing

Input : batch of video sequences  (B, T, C, H, W)
Output: binary prediction per video (real / fake)
"""

import os
import copy
import time
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the dataset class from the existing CNN+LSTM script
from train_cnn_lstm import VideoSequenceDataset

# Import the ST-ViT model
from st_vit_model import ST_ViT, count_parameters


# ─── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = "dataset_sequences"
IMG_SIZE = 224
SEQ_LEN = 20
BATCH_SIZE = 2              # Micro-batch (fits in 4GB VRAM with FP16)
ACCUM_STEPS = 4             # Gradient accumulation -> effective batch = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4         # L2 regularization
LABEL_SMOOTHING = 0.1       # Reduce overconfidence
EPOCHS = 20                 # More epochs for larger dataset
UNFREEZE_CNN_EPOCH = 5      # Epoch at which to unfreeze CNN backbone
NUM_WORKERS = 0             # 0 workers on 8GB RAM to avoid paging file exhaustion
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

# Model hyperparameters
LSTM_HIDDEN = 256
LSTM_LAYERS = 2
VIT_DIM = 512
VIT_DEPTH = 4
VIT_HEADS = 8


# ─── Label Smoothing BCE Loss ───────────────────────────────────────────────
class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    """
    BCEWithLogitsLoss with label smoothing.
    Instead of hard labels (0/1), uses soft labels:
      0 -> smoothing/2
      1 -> 1 - smoothing/2
    This prevents the model from becoming overconfident.
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # Smooth: 0 -> 0.05, 1 -> 0.95 (for smoothing=0.1)
        smooth_targets = targets * (1 - self.smoothing) + self.smoothing / 2
        return self.bce(logits, smooth_targets)


# ─── Hardware Setup ──────────────────────────────────────────────────────────
def setup_hardware():
    """Configure CUDA for optimal performance."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()
        gc.collect()

        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU             : {gpu_name}")
        print(f"  VRAM            : {vram_gb:.1f} GB")
        print(f"  Mixed Precision : Enabled (FP16)")
        print(f"  cuDNN Benchmark : Enabled")
        print(f"  TF32            : Enabled")
    else:
        print("  GPU             : None (using CPU)")
        print("  Mixed Precision : Disabled")

    print(f"  CPU Cores       : {os.cpu_count()}")
    print(f"  DataLoader Workers: {NUM_WORKERS}")
    print(f"  Micro-batch     : {BATCH_SIZE}")
    print(f"  Accumulation    : {ACCUM_STEPS}x (effective batch = {BATCH_SIZE * ACCUM_STEPS})")


# ─── Data Loading ────────────────────────────────────────────────────────────
def get_data_loaders():
    """Build train / val / test DataLoaders with strong augmentation."""
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Stronger augmentation pipeline for training
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),  # Resize larger
        transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),         # Random crop back
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.05),              # Rare vertical flip
        transforms.RandomRotation(degrees=10),               # Slight rotation
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),                          # Slight shift
            scale=(0.95, 1.05),                              # Slight zoom
        ),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.RandomGrayscale(p=0.05),                 # Occasional grayscale
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),  # Simulate compression
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),  # Cutout-style
    ])

    datasets_ = {
        "train": VideoSequenceDataset(DATA_DIR, "train", SEQ_LEN, train_transform),
        "val":   VideoSequenceDataset(DATA_DIR, "val",   SEQ_LEN, val_transform),
        "test":  VideoSequenceDataset(DATA_DIR, "test",  SEQ_LEN, val_transform),
    }

    loaders = {
        split: DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
        for split, ds in datasets_.items()
    }

    return loaders, {s: len(d) for s, d in datasets_.items()}


# ─── Plotting ────────────────────────────────────────────────────────────────
def plot_history(history):
    """Save training & validation accuracy/loss/F1/LR plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(len(history["train_acc"]))

    # Accuracy
    axes[0, 0].plot(epochs, history["train_acc"], "bo-", label="Train Acc", markersize=3)
    axes[0, 0].plot(epochs, history["val_acc"], "ro-", label="Val Acc", markersize=3)
    axes[0, 0].set_title("Accuracy")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Loss
    axes[0, 1].plot(epochs, history["train_loss"], "bo-", label="Train Loss", markersize=3)
    axes[0, 1].plot(epochs, history["val_loss"], "ro-", label="Val Loss", markersize=3)
    axes[0, 1].set_title("Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # F1 Score
    axes[1, 0].plot(epochs, history["train_f1"], "bo-", label="Train F1", markersize=3)
    axes[1, 0].plot(epochs, history["val_f1"], "ro-", label="Val F1", markersize=3)
    axes[1, 0].set_title("F1 Score")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # AUC-ROC
    axes[1, 1].plot(epochs, history["train_auc"], "bo-", label="Train AUC", markersize=3)
    axes[1, 1].plot(epochs, history["val_auc"], "ro-", label="Val AUC", markersize=3)
    axes[1, 1].set_title("AUC-ROC")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Mark unfreeze epoch
    for ax in axes.flat:
        ax.axvline(x=UNFREEZE_CNN_EPOCH, color="green", linestyle="--",
                   alpha=0.5, label="CNN Unfreeze")

    plt.suptitle("ST-ViT Training History (V2 — Full Dataset)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("st_vit_training_history.png", dpi=150)
    print("Training history saved to st_vit_training_history.png")


# ─── Training Loop ──────────────────────────────────────────────────────────
def train_model(model, loaders, sizes, criterion, num_epochs=EPOCHS):
    """
    Training loop with progressive CNN unfreezing, mixed precision,
    cosine annealing LR, and gradient accumulation.
    """
    since = time.time()
    best_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_acc = 0.0

    history = {
        "train_loss": [], "train_acc": [], "train_f1": [], "train_auc": [],
        "val_loss": [],   "val_acc": [],   "val_f1": [],   "val_auc": [],
    }

    # Mixed precision scaler
    scaler = GradScaler("cuda", enabled=USE_AMP)

    # AdamW optimizer with weight decay (L2 regularization)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # Cosine annealing LR scheduler — restarts every 5 epochs
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs - 1}")
        print("-" * 50)

        # Print GPU memory + current LR
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"  GPU Memory: {alloc:.0f}MB allocated / {reserved:.0f}MB reserved")

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.6f}")

        # ── Progressive unfreezing ──
        if epoch == UNFREEZE_CNN_EPOCH:
            print("\n" + "=" * 50)
            print("  UNFREEZING CNN BACKBONE (ResNet-50)")
            print("  CNN will now be fine-tuned at LR/10")
            print("=" * 50 + "\n")

            model.unfreeze_cnn()

            # Create new optimizer with separate learning rates + weight decay
            optimizer = optim.AdamW(
                model.get_parameter_groups(
                    lr_cnn=LR / 10,
                    lr_rest=LR,
                ),
                weight_decay=WEIGHT_DECAY,
            )
            # New scheduler for the new optimizer
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=5, T_mult=2, eta_min=1e-6
            )
            scaler = GradScaler("cuda", enabled=USE_AMP)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_correct = 0
            all_labels, all_probs = [], []

            optimizer.zero_grad()
            accum_count = 0

            for batch_idx, (sequences, labels) in enumerate(
                tqdm(loaders[phase], desc=f"{phase} ep{epoch}")
            ):
                sequences = sequences.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True).unsqueeze(1)

                with torch.set_grad_enabled(phase == "train"):
                    with autocast("cuda", enabled=USE_AMP):
                        logits = model(sequences)
                        loss = criterion(logits, labels)
                        if phase == "train":
                            loss_scaled = loss / ACCUM_STEPS

                    if phase == "train":
                        scaler.scale(loss_scaled).backward()
                        accum_count += 1

                        if accum_count % ACCUM_STEPS == 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=1.0
                            )
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()

                probs = torch.sigmoid(logits.float())
                preds = (probs > 0.5).float()
                running_loss += loss.item() * sequences.size(0)
                running_correct += (preds == labels).sum().item()

                all_labels.extend(labels.cpu().numpy().flatten())
                all_probs.extend(probs.detach().cpu().numpy().flatten())

            # Handle remaining accumulated gradients
            if phase == "train" and accum_count % ACCUM_STEPS != 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Step the LR scheduler after training phase
            if phase == "train":
                scheduler.step(epoch)

            epoch_loss = running_loss / sizes[phase]
            epoch_acc = running_correct / sizes[phase]

            all_preds_binary = [1 if p > 0.5 else 0 for p in all_probs]
            try:
                epoch_auc = roc_auc_score(all_labels, all_probs)
            except ValueError:
                epoch_auc = 0.0
            try:
                epoch_f1 = f1_score(all_labels, all_preds_binary)
            except ValueError:
                epoch_f1 = 0.0

            frozen_str = " [CNN frozen]" if model.freeze_cnn and phase == "train" else ""
            print(f"  {phase}  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}  "
                  f"F1: {epoch_f1:.4f}  AUC: {epoch_auc:.4f}{frozen_str}")

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)
            history[f"{phase}_f1"].append(epoch_f1)
            history[f"{phase}_auc"].append(epoch_auc)

            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), "best_st_vit_model.pth")
                print("  -> Saved best model checkpoint")

        # Free GPU memory between epochs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - since
    print(f"\nTraining complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}  Loss: {best_loss:.4f}")

    model.load_state_dict(best_wts)
    return model, history


# ─── Evaluation ──────────────────────────────────────────────────────────────
def evaluate(model, loader, sizes, criterion):
    """Run evaluation on the test set and print all metrics."""
    model.eval()
    running_loss = 0.0
    running_correct = 0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for sequences, labels in tqdm(loader, desc="Testing"):
            sequences = sequences.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True).unsqueeze(1)

            with autocast("cuda", enabled=USE_AMP):
                logits = model(sequences)
                loss = criterion(logits, labels)

            probs = torch.sigmoid(logits.float())
            preds = (probs > 0.5).float()
            running_loss += loss.item() * sequences.size(0)
            running_correct += (preds == labels).sum().item()

            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs.detach().cpu().numpy().flatten())

    test_loss = running_loss / sizes["test"]
    test_acc = running_correct / sizes["test"]

    all_preds_binary = [1 if p > 0.5 else 0 for p in all_probs]
    try:
        test_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        test_auc = 0.0
    try:
        test_f1 = f1_score(all_labels, all_preds_binary)
    except ValueError:
        test_f1 = 0.0
    try:
        test_precision = precision_score(all_labels, all_preds_binary)
    except ValueError:
        test_precision = 0.0
    try:
        test_recall = recall_score(all_labels, all_preds_binary)
    except ValueError:
        test_recall = 0.0

    print(f"\n{'=' * 50}")
    print(f"  ST-ViT TEST RESULTS (V2)")
    print(f"  Loss      : {test_loss:.4f}")
    print(f"  Accuracy  : {test_acc:.4f}")
    print(f"  F1 Score  : {test_f1:.4f}")
    print(f"  Precision : {test_precision:.4f}")
    print(f"  Recall    : {test_recall:.4f}")
    print(f"  AUC-ROC   : {test_auc:.4f}")
    print(f"{'=' * 50}")

    return {
        "loss": test_loss,
        "accuracy": test_acc,
        "f1": test_f1,
        "precision": test_precision,
        "recall": test_recall,
        "auc": test_auc,
    }


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  ST-ViT V2: Spatiotemporal Vision Transformer")
    print("  Deepfake Detection Training (Full Dataset)")
    print("=" * 60)

    setup_hardware()

    print(f"  CNN Backbone      : ResNet-50 (frozen -> unfreeze at epoch {UNFREEZE_CNN_EPOCH})")
    print(f"  LSTM              : BiLSTM, {LSTM_LAYERS} layers, {LSTM_HIDDEN} hidden")
    print(f"  ViT               : {VIT_DEPTH} layers, {VIT_HEADS} heads, dim={VIT_DIM}")
    print(f"  Sequence Length   : {SEQ_LEN} frames")
    print(f"  Epochs            : {EPOCHS}")
    print(f"  Learning Rate     : {LR} (cosine annealing)")
    print(f"  Weight Decay      : {WEIGHT_DECAY}")
    print(f"  Label Smoothing   : {LABEL_SMOOTHING}")
    print("=" * 60)

    if not os.path.exists(DATA_DIR):
        print(f"\nERROR: Dataset directory '{DATA_DIR}' not found.")
        print("Run  python preprocess_sequences.py  first.")
        return

    # Load data
    loaders, sizes = get_data_loaders()
    print(f"\nDataset sizes: {sizes}")

    # Build model
    model = ST_ViT(
        seq_len=SEQ_LEN,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        vit_dim=VIT_DIM,
        vit_depth=VIT_DEPTH,
        vit_heads=VIT_HEADS,
        freeze_cnn=True,
    ).to(DEVICE)

    count_parameters(model)

    # Loss function with label smoothing
    criterion = LabelSmoothingBCEWithLogitsLoss(smoothing=LABEL_SMOOTHING)

    # Train
    trained_model, history = train_model(
        model, loaders, sizes, criterion, EPOCHS
    )

    # Save final model
    torch.save(trained_model.state_dict(), "final_st_vit_model.pth")
    print("Final model saved to final_st_vit_model.pth")

    # Plot training history
    plot_history(history)

    # Evaluate on test set (use standard BCE for fair test evaluation)
    test_criterion = nn.BCEWithLogitsLoss()
    test_results = evaluate(trained_model, loaders["test"], sizes, test_criterion)

    # Print comparison
    print("\n" + "=" * 60)
    print("  To compare with baseline CNN+LSTM:")
    print("  Run  python train_cnn_lstm.py  and compare metrics.")
    print("=" * 60)


if __name__ == "__main__":
    main()
