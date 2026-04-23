"""
CNN + LSTM Deepfake Detection Training Script.

Architecture:
  - CNN backbone : ResNet-18 (pretrained, frozen) → 512-dim feature per frame
  - LSTM         : 512 → 256 hidden, 2 layers
  - Classifier   : 256 → 1 (sigmoid for binary classification)

Input : batch of video sequences  (B, T, C, H, W)
Output: binary prediction per video (real / fake)
"""

import os
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = "dataset_sequences"
IMG_SIZE = 224
SEQ_LEN = 20           # Must match preprocess_sequences.py
BATCH_SIZE = 4          # Sequences are memory-heavy
LR = 1e-4
EPOCHS = 10
LSTM_HIDDEN = 256
LSTM_LAYERS = 2
FREEZE_CNN = True       # Freeze ResNet weights (train only LSTM + FC)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Dataset ─────────────────────────────────────────────────────────────────
class VideoSequenceDataset(Dataset):
    """
    Loads sequences of face images from a folder structure:
        root/<split>/<label>/<video_name>/{0..N}.jpg

    Returns padded sequences of length `seq_len` and a binary label.
    """

    LABEL_MAP = {"real": 0, "fake": 1}

    def __init__(self, root_dir, split="train", seq_len=SEQ_LEN, transform=None):
        self.seq_len = seq_len
        self.transform = transform
        self.samples = []  # List of (video_folder_path, label)

        split_dir = os.path.join(root_dir, split)
        for label_name in ["real", "fake"]:
            label_dir = os.path.join(split_dir, label_name)
            if not os.path.isdir(label_dir):
                continue
            for video_name in sorted(os.listdir(label_dir)):
                video_dir = os.path.join(label_dir, video_name)
                if os.path.isdir(video_dir):
                    # Only include if the folder has at least 1 image
                    jpgs = [f for f in os.listdir(video_dir) if f.endswith(".jpg")]
                    if len(jpgs) > 0:
                        self.samples.append(
                            (video_dir, self.LABEL_MAP[label_name])
                        )

        print(f"[{split}] Loaded {len(self.samples)} video sequences "
              f"(real: {sum(1 for _, l in self.samples if l == 0)}, "
              f"fake: {sum(1 for _, l in self.samples if l == 1)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_dir, label = self.samples[idx]

        # Load available frames in order
        frame_files = sorted(
            [f for f in os.listdir(video_dir) if f.endswith(".jpg")],
            key=lambda x: int(os.path.splitext(x)[0]),
        )

        frames = []
        for fname in frame_files[: self.seq_len]:
            img = Image.open(os.path.join(video_dir, fname)).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        # Pad with zeros if fewer frames than seq_len
        if len(frames) < self.seq_len:
            pad = torch.zeros_like(frames[0]) if frames else torch.zeros(3, IMG_SIZE, IMG_SIZE)
            while len(frames) < self.seq_len:
                frames.append(pad)

        sequence = torch.stack(frames)  # (T, C, H, W)
        return sequence, torch.tensor(label, dtype=torch.float32)


# ─── Model ───────────────────────────────────────────────────────────────────
class CNN_LSTM(nn.Module):
    """
    CNN (ResNet-18) + LSTM for video-level deepfake detection.

    Forward pass:
        1. For each frame in the sequence, extract CNN features (512-d).
        2. Feed the feature sequence into a 2-layer LSTM.
        3. Take the last hidden state → FC → 1 logit.
    """

    def __init__(
        self,
        hidden_size=LSTM_HIDDEN,
        num_layers=LSTM_LAYERS,
        freeze_cnn=FREEZE_CNN,
    ):
        super().__init__()

        # ── CNN backbone (ResNet-18, remove final FC) ──
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn_feat_dim = resnet.fc.in_features  # 512
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # up to avgpool

        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

        # ── LSTM ──
        self.lstm = nn.LSTM(
            input_size=self.cnn_feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
        )

        # ── Classifier head ──
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W)  – batch of video sequences
        Returns:
            logits: (B, 1)
        """
        B, T, C, H, W = x.shape

        # Reshape to (B*T, C, H, W) for CNN
        x = x.view(B * T, C, H, W)
        with torch.no_grad() if FREEZE_CNN else torch.enable_grad():
            features = self.cnn(x)  # (B*T, 512, 1, 1)
        features = features.view(B, T, -1)  # (B, T, 512)

        # LSTM
        lstm_out, _ = self.lstm(features)  # (B, T, hidden)
        last_hidden = lstm_out[:, -1, :]   # (B, hidden) — last time step

        # Classify
        logits = self.classifier(last_hidden)  # (B, 1)
        return logits


# ─── Training Utilities ─────────────────────────────────────────────────────
def plot_history(history):
    """Save training & validation accuracy/loss plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(len(history["train_acc"]))

    axes[0].plot(epochs, history["train_acc"], "bo-", label="Train Acc")
    axes[0].plot(epochs, history["val_acc"], "ro-", label="Val Acc")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, history["train_loss"], "bo-", label="Train Loss")
    axes[1].plot(epochs, history["val_loss"], "ro-", label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("cnn_lstm_training_history.png")
    print("Training history saved to cnn_lstm_training_history.png")


def get_data_loaders():
    """Build train / val / test DataLoaders."""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),  # ImageNet stats
    ])
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    datasets_ = {
        "train": VideoSequenceDataset(DATA_DIR, "train", SEQ_LEN, train_transform),
        "val":   VideoSequenceDataset(DATA_DIR, "val",   SEQ_LEN, transform),
        "test":  VideoSequenceDataset(DATA_DIR, "test",  SEQ_LEN, transform),
    }

    loaders = {
        split: DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=2,
            pin_memory=True,
        )
        for split, ds in datasets_.items()
    }

    return loaders, {s: len(d) for s, d in datasets_.items()}


def train_model(model, loaders, sizes, criterion, optimizer, num_epochs=EPOCHS):
    """Standard PyTorch training loop with best-model checkpointing."""
    since = time.time()
    best_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_acc = 0.0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs - 1}")
        print("-" * 40)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_correct = 0
            all_labels, all_preds = [], []

            for sequences, labels in tqdm(loaders[phase], desc=f"{phase} ep{epoch}"):
                sequences = sequences.to(DEVICE)          # (B, T, C, H, W)
                labels = labels.to(DEVICE).unsqueeze(1)    # (B, 1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    logits = model(sequences)
                    loss = criterion(logits, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                preds = (torch.sigmoid(logits) > 0.5).float()
                running_loss += loss.item() * sequences.size(0)
                running_correct += (preds == labels).sum().item()

                all_labels.extend(labels.cpu().numpy().flatten())
                all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())

            epoch_loss = running_loss / sizes[phase]
            epoch_acc = running_correct / sizes[phase]

            try:
                epoch_auc = roc_auc_score(all_labels, all_preds)
            except ValueError:
                epoch_auc = 0.0

            print(f"  {phase}  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}  AUC: {epoch_auc:.4f}")

            if phase == "train":
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc)
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc)

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_acc = epoch_acc
                    best_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), "best_cnn_lstm_model.pth")
                    print("  ↳ Saved best model checkpoint")

    elapsed = time.time() - since
    print(f"\nTraining complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}  Loss: {best_loss:.4f}")

    model.load_state_dict(best_wts)
    return model, history


# ─── Evaluation ──────────────────────────────────────────────────────────────
def evaluate(model, loader, sizes, criterion):
    """Run evaluation on the test set and print metrics."""
    model.eval()
    running_loss = 0.0
    running_correct = 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for sequences, labels in tqdm(loader, desc="Testing"):
            sequences = sequences.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            logits = model(sequences)
            loss = criterion(logits, labels)

            preds = (torch.sigmoid(logits) > 0.5).float()
            running_loss += loss.item() * sequences.size(0)
            running_correct += (preds == labels).sum().item()

            all_labels.extend(labels.cpu().numpy().flatten())
            all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())

    test_loss = running_loss / sizes["test"]
    test_acc = running_correct / sizes["test"]

    try:
        test_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        test_auc = 0.0

    print(f"\n{'=' * 40}")
    print(f"  TEST RESULTS")
    print(f"  Loss : {test_loss:.4f}")
    print(f"  Acc  : {test_acc:.4f}")
    print(f"  AUC  : {test_auc:.4f}")
    print(f"{'=' * 40}")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  CNN + LSTM Deepfake Detection Training")
    print(f"  Device : {DEVICE}")
    print(f"  Frozen CNN : {FREEZE_CNN}")
    print("=" * 60)

    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Dataset directory '{DATA_DIR}' not found.")
        print("Run  python preprocess_sequences.py  first.")
        return

    loaders, sizes = get_data_loaders()
    print(f"Dataset sizes: {sizes}")

    model = CNN_LSTM().to(DEVICE)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )

    trained_model, history = train_model(
        model, loaders, sizes, criterion, optimizer, EPOCHS
    )

    # Save final model
    torch.save(trained_model.state_dict(), "final_cnn_lstm_model.pth")
    print("Final model saved to final_cnn_lstm_model.pth")

    # Plot history
    plot_history(history)

    # Evaluate on test set
    evaluate(trained_model, loaders["test"], sizes, criterion)


if __name__ == "__main__":
    main()
