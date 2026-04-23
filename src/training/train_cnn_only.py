"""
CNN-Only Baseline for Deepfake Detection.

Architecture:
  ResNet-50 (pretrained) -> Global Average Pool -> FC(2048->256) -> ReLU
  -> Dropout(0.5) -> FC(256->1)

Strategy:
  Each frame is classified independently. Video-level prediction is the
  average of per-frame sigmoid probabilities across the 20-frame sequence.

This model has NO temporal modeling — it serves as a pure spatial baseline
to demonstrate the value of LSTM and ViT temporal components in ST-ViT.

Training config matches ST-ViT V2 for fair comparison:
  - AdamW + weight decay, label smoothing, cosine annealing LR, FP16
  - Same dataset (dataset_sequences, 150 videos/class)
  - Same metrics (Accuracy, F1, Precision, Recall, AUC-ROC)
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
from torchvision import transforms, models
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.training.train_cnn_lstm import VideoSequenceDataset


# ─── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = "dataset_sequences"
IMG_SIZE = 224
SEQ_LEN = 20
BATCH_SIZE = 2
ACCUM_STEPS = 4
LR = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
EPOCHS = 20
UNFREEZE_CNN_EPOCH = 5
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()


# ─── Model ───────────────────────────────────────────────────────────────────
class CNN_Only(nn.Module):
    """
    ResNet-50 frame-level classifier.
    Processes each frame independently, then averages predictions
    across the sequence for a video-level decision.
    """

    def __init__(self, freeze_cnn=True):
        super().__init__()
        self.freeze_cnn = freeze_cnn

        # ResNet-50 backbone (remove final FC)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn_feat_dim = resnet.fc.in_features  # 2048
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

        # Init classifier weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def unfreeze_cnn(self):
        self.freeze_cnn = False
        for param in self.cnn.parameters():
            param.requires_grad = True
        print("[CNN-Only] CNN backbone unfrozen for fine-tuning")

    def get_parameter_groups(self, lr_cnn, lr_rest):
        cnn_params = list(self.cnn.parameters())
        other_params = [
            p for n, p in self.named_parameters()
            if not n.startswith("cnn.") and p.requires_grad
        ]
        return [
            {"params": cnn_params, "lr": lr_cnn},
            {"params": other_params, "lr": lr_rest},
        ]

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) — batch of video sequences
        Returns:
            logits: (B, 1) — video-level logit (averaged across frames)
        """
        B, T, C, H, W = x.shape

        # Process all frames through CNN
        x = x.view(B * T, C, H, W)
        if self.freeze_cnn:
            with torch.no_grad():
                features = self.cnn(x)
        else:
            features = self.cnn(x)
        features = features.view(B * T, -1)  # (B*T, 2048)

        # Per-frame logits
        frame_logits = self.classifier(features)  # (B*T, 1)
        frame_logits = frame_logits.view(B, T, 1)  # (B, T, 1)

        # Average logits across frames for video-level prediction
        video_logits = frame_logits.mean(dim=1)  # (B, 1)
        return video_logits


# ─── Label Smoothing Loss ────────────────────────────────────────────────────
class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        smooth_targets = targets * (1 - self.smoothing) + self.smoothing / 2
        return self.bce(logits, smooth_targets)


# ─── Hardware Setup ──────────────────────────────────────────────────────────
def setup_hardware():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()
        gc.collect()
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {gpu_name} ({vram_gb:.1f} GB) | FP16: Enabled")
    else:
        print("  GPU: None (CPU mode)")


# ─── Data Loading ────────────────────────────────────────────────────────────
def get_data_loaders():
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_transform = transforms.Compose([
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

    datasets_ = {
        "train": VideoSequenceDataset(DATA_DIR, "train", SEQ_LEN, train_transform),
        "val":   VideoSequenceDataset(DATA_DIR, "val",   SEQ_LEN, val_transform),
        "test":  VideoSequenceDataset(DATA_DIR, "test",  SEQ_LEN, val_transform),
    }
    loaders = {
        split: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=(split == "train"),
                          num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
        for split, ds in datasets_.items()
    }
    return loaders, {s: len(d) for s, d in datasets_.items()}


# ─── Plotting ────────────────────────────────────────────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(len(history["train_acc"]))

    axes[0, 0].plot(epochs, history["train_acc"], "bo-", label="Train", ms=3)
    axes[0, 0].plot(epochs, history["val_acc"], "ro-", label="Val", ms=3)
    axes[0, 0].set_title("Accuracy"); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history["train_loss"], "bo-", label="Train", ms=3)
    axes[0, 1].plot(epochs, history["val_loss"], "ro-", label="Val", ms=3)
    axes[0, 1].set_title("Loss"); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history["train_f1"], "bo-", label="Train", ms=3)
    axes[1, 0].plot(epochs, history["val_f1"], "ro-", label="Val", ms=3)
    axes[1, 0].set_title("F1 Score"); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history["train_auc"], "bo-", label="Train", ms=3)
    axes[1, 1].plot(epochs, history["val_auc"], "ro-", label="Val", ms=3)
    axes[1, 1].set_title("AUC-ROC"); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.axvline(x=UNFREEZE_CNN_EPOCH, color="green", linestyle="--", alpha=0.5)

    plt.suptitle("CNN-Only Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("cnn_only_training_history.png", dpi=150)
    print("Training history saved to cnn_only_training_history.png")


# ─── Training Loop ──────────────────────────────────────────────────────────
def train_model(model, loaders, sizes, criterion, num_epochs=EPOCHS):
    since = time.time()
    best_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_acc = 0.0

    history = {
        "train_loss": [], "train_acc": [], "train_f1": [], "train_auc": [],
        "val_loss": [],   "val_acc": [],   "val_f1": [],   "val_auc": [],
    }

    scaler = GradScaler("cuda", enabled=USE_AMP)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs - 1}")
        print("-" * 50)

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**2
            print(f"  GPU Memory: {alloc:.0f}MB | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Progressive unfreezing
        if epoch == UNFREEZE_CNN_EPOCH:
            print("\n  >>> UNFREEZING CNN BACKBONE <<<\n")
            model.unfreeze_cnn()
            optimizer = optim.AdamW(
                model.get_parameter_groups(lr_cnn=LR / 10, lr_rest=LR),
                weight_decay=WEIGHT_DECAY,
            )
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
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
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()

                probs = torch.sigmoid(logits.float())
                preds = (probs > 0.5).float()
                running_loss += loss.item() * sequences.size(0)
                running_correct += (preds == labels).sum().item()
                all_labels.extend(labels.cpu().numpy().flatten())
                all_probs.extend(probs.detach().cpu().numpy().flatten())

            if phase == "train" and accum_count % ACCUM_STEPS != 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if phase == "train":
                scheduler.step(epoch)

            epoch_loss = running_loss / sizes[phase]
            epoch_acc = running_correct / sizes[phase]
            all_preds_binary = [1 if p > 0.5 else 0 for p in all_probs]
            try: epoch_auc = roc_auc_score(all_labels, all_probs)
            except ValueError: epoch_auc = 0.0
            try: epoch_f1 = f1_score(all_labels, all_preds_binary)
            except ValueError: epoch_f1 = 0.0

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
                torch.save(model.state_dict(), "best_cnn_only_model.pth")
                print("  -> Saved best model checkpoint")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - since
    print(f"\nTraining complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}  Loss: {best_loss:.4f}")
    model.load_state_dict(best_wts)
    return model, history


# ─── Evaluation ──────────────────────────────────────────────────────────────
def evaluate(model, loader, sizes):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    all_labels, all_probs = [], []
    criterion = nn.BCEWithLogitsLoss()

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
    try: test_auc = roc_auc_score(all_labels, all_probs)
    except ValueError: test_auc = 0.0
    try: test_f1 = f1_score(all_labels, all_preds_binary)
    except ValueError: test_f1 = 0.0
    try: test_prec = precision_score(all_labels, all_preds_binary)
    except ValueError: test_prec = 0.0
    try: test_rec = recall_score(all_labels, all_preds_binary)
    except ValueError: test_rec = 0.0

    print(f"\n{'=' * 50}")
    print(f"  CNN-Only TEST RESULTS")
    print(f"  Loss      : {test_loss:.4f}")
    print(f"  Accuracy  : {test_acc:.4f}")
    print(f"  F1 Score  : {test_f1:.4f}")
    print(f"  Precision : {test_prec:.4f}")
    print(f"  Recall    : {test_rec:.4f}")
    print(f"  AUC-ROC   : {test_auc:.4f}")
    print(f"{'=' * 50}")
    return {"loss": test_loss, "accuracy": test_acc, "f1": test_f1,
            "precision": test_prec, "recall": test_rec, "auc": test_auc}


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  CNN-Only Baseline — Deepfake Detection")
    print("=" * 60)
    setup_hardware()

    if not os.path.exists(DATA_DIR):
        print(f"ERROR: '{DATA_DIR}' not found. Run preprocess_sequences.py first.")
        return

    loaders, sizes = get_data_loaders()
    print(f"Dataset sizes: {sizes}")

    model = CNN_Only(freeze_cnn=True).to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")

    criterion = LabelSmoothingBCEWithLogitsLoss(smoothing=LABEL_SMOOTHING)
    trained_model, history = train_model(model, loaders, sizes, criterion, EPOCHS)

    torch.save(trained_model.state_dict(), "final_cnn_only_model.pth")
    plot_history(history)
    evaluate(trained_model, loaders["test"], sizes)


if __name__ == "__main__":
    main()
