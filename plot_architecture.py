"""
ST-ViT Architecture Diagram — Black & White, IEEE 2-column format.
Fits a single column width (~3.5 inches) of an A4 2-column paper.
Style matches the user's existing CNN+LSTM diagram.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# IEEE single column: ~3.5 inches wide, aspect ratio ~1.4
fig, ax = plt.subplots(1, 1, figsize=(3.5, 5.0))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14.5)
ax.axis("off")

# All black/white/gray colors
BLACK = "#000000"
DGRAY = "#444444"
MGRAY = "#888888"
LGRAY = "#CCCCCC"
WHITE = "#FFFFFF"
VLGRAY = "#E8E8E8"

# ─── Helper functions ────────────────────────────────────────────────────────
def box(x, y, w, h, label=None, fontsize=5.5, fill=WHITE, lw=0.8, sublabel=None):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                           facecolor=fill, edgecolor=BLACK, linewidth=lw, zorder=3)
    ax.add_patch(rect)
    if label:
        dy = 0.12 if sublabel else 0
        ax.text(x + w/2, y + h/2 + dy, label, fontsize=fontsize,
                ha="center", va="center", color=BLACK, fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.2, sublabel, fontsize=4.5,
                ha="center", va="center", color=DGRAY, zorder=4)

def arrow_down(x, y1, y2, label=None):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=0.8, mutation_scale=8), zorder=2)
    if label:
        ax.text(x + 0.15, (y1 + y2) / 2, label, fontsize=4, color=DGRAY,
                va="center", ha="left", fontfamily="monospace")

def arrow_right(x1, x2, y, label=None):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=0.8, mutation_scale=8), zorder=2)
    if label:
        ax.text((x1 + x2) / 2, y + 0.15, label, fontsize=4, color=DGRAY,
                ha="center", va="bottom", fontfamily="monospace")

def arrow_left(x1, x2, y):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=0.6, mutation_scale=6), zorder=2)

def draw_frame(x, y, w, h):
    """Draw a simple face placeholder (gray box with lines)."""
    rect = Rectangle((x, y), w, h, facecolor=LGRAY, edgecolor=BLACK, linewidth=0.6, zorder=3)
    ax.add_patch(rect)
    # Simple face outline
    cx, cy = x + w/2, y + h/2
    # Head circle
    circle = plt.Circle((cx, cy + 0.05), w * 0.28, fill=False, edgecolor=DGRAY, linewidth=0.4, zorder=4)
    ax.add_patch(circle)
    # Eyes
    ax.plot([cx - w*0.12, cx - w*0.06], [cy + 0.12, cy + 0.12], color=DGRAY, lw=0.4, zorder=4)
    ax.plot([cx + w*0.06, cx + w*0.12], [cy + 0.12, cy + 0.12], color=DGRAY, lw=0.4, zorder=4)
    # Mouth
    ax.plot([cx - w*0.08, cx + w*0.08], [cy - 0.05, cy - 0.05], color=DGRAY, lw=0.4, zorder=4)


# ═══════════════════════════════════════════════════════════════════════════════
# ROW 1: Input Frames (top)
# ═══════════════════════════════════════════════════════════════════════════════
y_frames = 13.0
ax.text(0.4, y_frames + 0.55, "Frame 1", fontsize=4.5, ha="center", va="center", color=BLACK)
ax.text(2.4, y_frames + 0.55, "Frame 2", fontsize=4.5, ha="center", va="center", color=BLACK)
ax.text(5.0, y_frames + 0.2, "...", fontsize=7, ha="center", va="center", color=BLACK)
ax.text(7.4, y_frames + 0.55, "Frame N-1", fontsize=4.5, ha="center", va="center", color=BLACK)
ax.text(9.4, y_frames + 0.55, "Frame N", fontsize=4.5, ha="center", va="center", color=BLACK)

# Draw frame placeholders
fw, fh = 1.2, 1.0
draw_frame(-0.2, y_frames - 0.7, fw, fh)
draw_frame(1.8, y_frames - 0.7, fw, fh)
draw_frame(6.8, y_frames - 0.7, fw, fh)
draw_frame(8.8, y_frames - 0.7, fw, fh)

# Arrows down from frames
arrow_down(0.4, y_frames - 0.75, 11.3)
arrow_down(2.4, y_frames - 0.75, 11.3)
arrow_down(7.4, y_frames - 0.75, 11.3)
arrow_down(9.4, y_frames - 0.75, 11.3)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 2: CNN Backbone (ResNet-50)
# ═══════════════════════════════════════════════════════════════════════════════
y_cnn = 10.3
box(0.5, y_cnn, 9.0, 0.95, "Stage 1: Convolutional Neural Network (ResNet-50)",
    fontsize=5.5, fill=VLGRAY, sublabel="Pretrained spatial feature extractor — per frame")

# Arrows down from CNN
y_feat = 9.3
arrow_down(0.4, y_cnn, y_feat + 0.6)
arrow_down(2.4, y_cnn, y_feat + 0.6)
arrow_down(7.4, y_cnn, y_feat + 0.6)
arrow_down(9.4, y_cnn, y_feat + 0.6)

# Feature vectors
box(-0.1, y_feat, 1.1, 0.55, "Feature\nVector 1", fontsize=4.5, fill=WHITE)
box(1.9, y_feat, 1.1, 0.55, "Feature\nVector 2", fontsize=4.5, fill=WHITE)
ax.text(5.0, y_feat + 0.3, "...", fontsize=7, ha="center", va="center", color=BLACK)
box(6.8, y_feat, 1.2, 0.55, "Feature\nVector N-1", fontsize=4.5, fill=WHITE)
box(8.8, y_feat, 1.1, 0.55, "Feature\nVector N", fontsize=4.5, fill=WHITE)

ax.text(5.0, y_feat - 0.2, "(2048-dimensional per frame)", fontsize=4, color=MGRAY, ha="center")

# Arrows down from features to BiLSTM
y_lstm_top = 8.15
arrow_down(0.45, y_feat, y_lstm_top + 0.85)
arrow_down(2.45, y_feat, y_lstm_top + 0.85)
arrow_down(7.4, y_feat, y_lstm_top + 0.85)
arrow_down(9.35, y_feat, y_lstm_top + 0.85)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 3: BiLSTM
# ═══════════════════════════════════════════════════════════════════════════════
y_lstm = 7.15
box(0.5, y_lstm, 9.0, 1.8, fill=VLGRAY, lw=0.8)
ax.text(5.0, y_lstm + 1.55, "Stage 2: Bidirectional LSTM Sequence Modeling",
        fontsize=5.5, fontweight="bold", ha="center", va="center", color=BLACK, zorder=4)
ax.text(5.0, y_lstm + 1.25, "(2 layers, 256 hidden units per direction)",
        fontsize=4.5, ha="center", va="center", color=DGRAY, zorder=4)

# LSTM cells
cell_w, cell_h = 1.3, 0.6
cells_x = [0.8, 2.8, 6.5, 8.3]
cells_labels = ["LSTM₁", "LSTM₂", "LSTM_{N-1}", "LSTM_N"]
for i, (cx, cl) in enumerate(zip(cells_x, cells_labels)):
    box(cx, y_lstm + 0.3, cell_w, cell_h, cl, fontsize=5, fill=WHITE)

# Forward arrows
arrow_right(2.1, 2.8, y_lstm + 0.6)
arrow_right(4.1, 4.8, y_lstm + 0.6)
ax.text(5.5, y_lstm + 0.6, "...", fontsize=6, ha="center", va="center", color=BLACK)
arrow_right(5.9, 6.5, y_lstm + 0.6)

# Backward arrows (below)
arrow_left(2.8, 2.1, y_lstm + 0.45)
arrow_left(6.5, 5.9, y_lstm + 0.45)
ax.text(5.5, y_lstm + 0.35, "...", fontsize=6, ha="center", va="center", color=BLACK)
arrow_left(4.8, 4.1, y_lstm + 0.45)

# Arrows down to temporal features
y_tfeat = 6.1
arrow_down(5.0, y_lstm, y_tfeat + 0.55)

# Temporal features output
box(2.5, y_tfeat, 5.0, 0.5, "Temporal Feature Tokens (512-dim per position)",
    fontsize=5, fill=WHITE)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 4: Projection Layer
# ═══════════════════════════════════════════════════════════════════════════════
y_proj = 5.15
arrow_down(5.0, y_tfeat, y_proj + 0.5)
box(2.8, y_proj, 4.4, 0.45, "Linear Projection + LayerNorm + GELU",
    fontsize=5, fill=WHITE)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 5: Vision Transformer Encoder
# ═══════════════════════════════════════════════════════════════════════════════
y_vit = 2.7
arrow_down(5.0, y_proj, y_vit + 2.3)

box(0.5, y_vit, 9.0, 2.3, fill=VLGRAY, lw=0.8)
ax.text(5.0, y_vit + 2.05, "Stage 3: Vision Transformer Encoder (×4 layers)",
        fontsize=5.5, fontweight="bold", ha="center", va="center", color=BLACK, zorder=4)

# [CLS] token
box(0.8, y_vit + 1.15, 1.5, 0.55, "[CLS] Token", fontsize=5, fill=WHITE)
ax.text(1.55, y_vit + 0.95, "(learnable)", fontsize=3.5, ha="center", color=MGRAY)

# Positional Embedding
box(2.7, y_vit + 1.15, 2.3, 0.55, "Positional\nEmbeddings", fontsize=5, fill=WHITE)

# Multi-Head Self Attention
box(5.5, y_vit + 1.15, 3.7, 0.55, "Multi-Head Self-Attention\n(8 heads, 512-dim)", fontsize=5, fill=WHITE)

# FFN
box(0.8, y_vit + 0.2, 8.4, 0.55, "Feed-Forward Network (GELU) + Residual Connections + LayerNorm",
    fontsize=5, fill=WHITE)

# Output: CLS output
y_cls = 1.8
arrow_down(5.0, y_vit, y_cls + 0.55)
box(2.5, y_cls, 5.0, 0.5, "Global Context Vector (512-dim from [CLS])",
    fontsize=5, fill=WHITE)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 6: Classification Head
# ═══════════════════════════════════════════════════════════════════════════════
y_class = 0.8
arrow_down(5.0, y_cls, y_class + 0.65)

box(1.0, y_class, 5.0, 0.6, "Stage 4: FC(512→256) → ReLU → Dropout → FC(256→1)",
    fontsize=5, fill=VLGRAY, sublabel="Classification Head")

# Output
arrow_right(6.0, 7.0, y_class + 0.3)

# Real / Fake output
box(7.0, y_class + 0.25, 1.2, 0.35, "Real", fontsize=5.5, fill=WHITE)
box(8.5, y_class + 0.25, 1.2, 0.35, "Fake", fontsize=5.5, fill=LGRAY)

ax.text(5.0, 0.35, "Output: Real vs Fake Video Classification",
        fontsize=4.5, ha="center", color=DGRAY, style="italic")

# ═══════════════════════════════════════════════════════════════════════════════
plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
plt.savefig("stvit_architecture_bw.png", dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("Saved: stvit_architecture_bw.png (300 DPI, B&W, single-column)")
