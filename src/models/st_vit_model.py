"""
Spatiotemporal Vision Transformer (ST-ViT) for Deepfake Detection.

Novel architecture that chains three complementary stages:
  1. CNN  (ResNet-50)        → spatial feature extraction per frame
  2. BiLSTM (2-layer)        → temporal modeling across frames
  3. ViT Encoder             → global context via multi-head self-attention
  4. Classification Head     → binary real / fake prediction

Key novelty:  the Vision Transformer operates on **temporally enriched
feature tokens** produced by the BiLSTM, not on raw image patches.
This gives the self-attention mechanism access to both spatial (CNN)
and sequential (LSTM) information, enabling it to capture long-range
temporal dependencies that neither CNN nor LSTM can model alone.

Input : (B, T, C, H, W)  – batch of video frame sequences
Output: (B, 1)            – binary logit per video
"""

import math
import torch
import torch.nn as nn
from torchvision import models


# ─── Configuration defaults ─────────────────────────────────────────────────
CNN_FEAT_DIM = 2048          # ResNet-50 feature dimension
LSTM_HIDDEN = 256            # Per-direction hidden size (total = 512 for BiLSTM)
LSTM_LAYERS = 2
VIT_DIM = 512                # Transformer model dimension
VIT_DEPTH = 4                # Number of transformer encoder layers
VIT_HEADS = 8                # Number of attention heads
VIT_MLP_RATIO = 4            # FFN expansion ratio
VIT_DROPOUT = 0.1            # Dropout inside transformer
CLASSIFIER_DROPOUT = 0.5     # Dropout in classification head


# ─── Vision Transformer Encoder ─────────────────────────────────────────────
class ViTEncoder(nn.Module):
    """
    Lightweight Vision Transformer encoder that processes a sequence of
    temporal feature tokens (not image patches).

    Adds:
      - A learnable [CLS] token for aggregating global information
      - Learnable positional embeddings to retain temporal order
      - Standard Transformer encoder layers with multi-head self-attention

    The [CLS] token output is used as the global context representation
    for downstream classification.
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int = VIT_DIM,
        depth: int = VIT_DEPTH,
        num_heads: int = VIT_HEADS,
        mlp_ratio: int = VIT_MLP_RATIO,
        dropout: float = VIT_DROPOUT,
    ):
        super().__init__()

        self.d_model = d_model

        # Learnable [CLS] token — prepended to the sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional embeddings for [CLS] + T temporal tokens
        self.pos_embedding = nn.Parameter(
            torch.randn(1, seq_len + 1, d_model)
        )
        self.pos_dropout = nn.Dropout(dropout)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm for more stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=depth
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model) — temporally enriched feature tokens
        Returns:
            cls_output: (B, d_model) — global context from the [CLS] token
        """
        B, T, _ = x.shape

        # Expand [CLS] token for the batch
        cls_tokens = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)

        # Prepend [CLS] to sequence
        x = torch.cat([cls_tokens, x], dim=1)            # (B, T+1, d_model)

        # Add positional embeddings
        x = x + self.pos_embedding[:, : T + 1, :]
        x = self.pos_dropout(x)

        # Transformer encoder
        x = self.encoder(x)                              # (B, T+1, d_model)

        # Extract [CLS] token output
        cls_output = self.norm(x[:, 0])                  # (B, d_model)

        return cls_output


# ─── Full ST-ViT Model ──────────────────────────────────────────────────────
class ST_ViT(nn.Module):
    """
    Spatiotemporal Vision Transformer (ST-ViT).

    Pipeline:
        Video frames  →  CNN (per-frame)  →  BiLSTM (temporal)
                      →  ViT Encoder (global context)  →  FC (classify)

    Stages:
        1. ResNet-50 backbone extracts 2048-dim spatial features per frame
        2. 2-layer Bidirectional LSTM models temporal dynamics (output: 512-dim)
        3. Vision Transformer encoder with [CLS] token captures global context
        4. Two-layer FC head with dropout for binary classification
    """

    def __init__(
        self,
        seq_len: int = 20,
        cnn_feat_dim: int = CNN_FEAT_DIM,
        lstm_hidden: int = LSTM_HIDDEN,
        lstm_layers: int = LSTM_LAYERS,
        vit_dim: int = VIT_DIM,
        vit_depth: int = VIT_DEPTH,
        vit_heads: int = VIT_HEADS,
        vit_mlp_ratio: int = VIT_MLP_RATIO,
        vit_dropout: float = VIT_DROPOUT,
        classifier_dropout: float = CLASSIFIER_DROPOUT,
        freeze_cnn: bool = True,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.freeze_cnn = freeze_cnn

        # ── Stage 1: CNN Backbone (ResNet-50, pretrained) ──
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn_feat_dim = resnet.fc.in_features  # 2048
        # Remove the final classification layer, keep up to avgpool
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

        # ── Stage 2: Bidirectional LSTM ──
        self.lstm = nn.LSTM(
            input_size=self.cnn_feat_dim,       # 2048
            hidden_size=lstm_hidden,             # 256
            num_layers=lstm_layers,              # 2
            batch_first=True,
            bidirectional=True,                  # Key: bidirectional
            dropout=0.3 if lstm_layers > 1 else 0.0,
        )
        # BiLSTM output dimension: lstm_hidden * 2 = 512
        lstm_out_dim = lstm_hidden * 2

        # ── Projection: LSTM output → ViT input dimension ──
        # Linear projection if dimensions differ, identity otherwise
        if lstm_out_dim != vit_dim:
            self.projection = nn.Sequential(
                nn.Linear(lstm_out_dim, vit_dim),
                nn.LayerNorm(vit_dim),
                nn.GELU(),
            )
        else:
            self.projection = nn.LayerNorm(vit_dim)

        # ── Stage 3: Vision Transformer Encoder ──
        self.vit = ViTEncoder(
            seq_len=seq_len,
            d_model=vit_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            mlp_ratio=vit_mlp_ratio,
            dropout=vit_dropout,
        )

        # ── Stage 4: Classification Head ──
        self.classifier = nn.Sequential(
            nn.Linear(vit_dim, vit_dim // 2),    # 512 → 256
            nn.ReLU(),
            nn.Dropout(classifier_dropout),       # 0.5
            nn.Linear(vit_dim // 2, 1),           # 256 → 1
        )

        # Initialize weights for non-pretrained components
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for LSTM, ViT, and classifier layers."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def unfreeze_cnn(self):
        """Unfreeze CNN backbone for end-to-end fine-tuning."""
        self.freeze_cnn = False
        for param in self.cnn.parameters():
            param.requires_grad = True
        print("[ST-ViT] CNN backbone unfrozen for fine-tuning")

    def get_parameter_groups(self, lr_cnn: float, lr_rest: float):
        """
        Return parameter groups with different learning rates.
        CNN gets a lower LR for gentle fine-tuning after unfreezing.

        Args:
            lr_cnn:  learning rate for CNN backbone
            lr_rest: learning rate for LSTM + ViT + classifier

        Returns:
            list of parameter group dicts for torch.optim
        """
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
            x: (B, T, C, H, W) — batch of video frame sequences

        Returns:
            logits: (B, 1) — binary classification logit
        """
        B, T, C, H, W = x.shape

        # ── Stage 1: CNN feature extraction (per frame) ──
        x = x.view(B * T, C, H, W)
        if self.freeze_cnn:
            with torch.no_grad():
                cnn_features = self.cnn(x)          # (B*T, 2048, 1, 1)
        else:
            cnn_features = self.cnn(x)              # (B*T, 2048, 1, 1)
        cnn_features = cnn_features.view(B, T, -1)  # (B, T, 2048)

        # ── Stage 2: Bidirectional LSTM ──
        lstm_out, _ = self.lstm(cnn_features)        # (B, T, 512)

        # ── Projection to ViT dimension ──
        vit_input = self.projection(lstm_out)        # (B, T, vit_dim)

        # ── Stage 3: Vision Transformer global context ──
        global_ctx = self.vit(vit_input)             # (B, vit_dim)

        # ── Stage 4: Classification ──
        logits = self.classifier(global_ctx)         # (B, 1)

        return logits


# ─── Utility ─────────────────────────────────────────────────────────────────
def count_parameters(model):
    """Print a summary of model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"\n{'=' * 50}")
    print(f"  ST-ViT Parameter Summary")
    print(f"  Total      : {total:>12,}")
    print(f"  Trainable  : {trainable:>12,}")
    print(f"  Frozen     : {frozen:>12,}")
    print(f"{'=' * 50}\n")

    return total, trainable


if __name__ == "__main__":
    # Quick shape verification
    print("Testing ST-ViT forward pass...")

    model = ST_ViT(seq_len=20, freeze_cnn=True)
    count_parameters(model)

    # Dummy batch: 2 videos, 20 frames each, 224x224 RGB
    dummy = torch.randn(2, 20, 3, 224, 224)
    logits = model(dummy)

    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {logits.shape}")
    print(f"Output values: {logits.detach().numpy().flatten()}")
    assert logits.shape == (2, 1), f"Expected (2, 1), got {logits.shape}"
    print("\n[PASS] Forward pass test PASSED")
