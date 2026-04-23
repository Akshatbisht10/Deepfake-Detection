"""
Publication-quality comparison charts for the research paper.
Reads cv_results.json and generates polished figures.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Use a clean, academic font style
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['font.size'] = 12
rcParams['axes.linewidth'] = 1.2

# Load results
with open("cv_results.json", "r") as f:
    results = json.load(f)

# Explicit order: ST-ViT last for emphasis
model_order = ["CNN-Only", "CNN+LSTM", "ViT-Only", "ST-ViT (Ours)"]
model_names = [m for m in model_order if m in results]
# Display names (remove '(Ours)')
display_names = {"ST-ViT (Ours)": "ST-ViT"}
metrics = ["accuracy", "f1", "precision", "recall", "auc"]
metric_labels = ["Accuracy", "F1 Score", "Precision", "Recall", "AUC-ROC"]

# ─── Figure 1: Grouped Bar Chart with Error Bars ────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6.5))

x = np.arange(len(metrics))
width = 0.18
colors = ["#E8505B", "#14A76C", "#45B7D1", "#3D5A80"]

for i, (name, color) in enumerate(zip(model_names, colors[:len(model_names)])):
    means = [results[name][f"{m}_mean"] for m in metrics]
    stds = [results[name][f"{m}_std"] for m in metrics]
    offset = (i - len(model_names) / 2 + 0.5) * width
    label = display_names.get(name, name)

    bars = ax.bar(x + offset, means, width, yerr=stds, label=label,
                  color=color, edgecolor="white", linewidth=1.2,
                  capsize=0, error_kw={"linewidth": 1.5},
                  alpha=0.9, zorder=3)

    # Value labels above bars
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.015,
                f"{m:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
                color=color)

ax.set_ylabel("Score", fontsize=14, fontweight="bold", labelpad=10)
ax.set_title("Deepfake Detection: 5-Fold Cross-Validation Comparison",
             fontsize=15, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=12, fontweight="medium")
ax.set_ylim(0.70, 1.08)
ax.legend(loc="upper left", fontsize=11, framealpha=0.95, edgecolor="#ccc",
          fancybox=True, shadow=False)
ax.grid(axis="y", alpha=0.25, linestyle="--", zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=11)

plt.tight_layout()
plt.savefig("paper_comparison_bar.png", dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("Saved: paper_comparison_bar.png")


# ─── Figure 2: Radar / Spider Chart ─────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # close the polygon

colors_radar = ["#E8505B", "#14A76C", "#45B7D1", "#3D5A80"]
fills = [0.1, 0.1, 0.1, 0.25]

for i, (name, color, fill_alpha) in enumerate(zip(model_names, colors_radar[:len(model_names)], fills[:len(model_names)])):
    values = [results[name][f"{m}_mean"] for m in metrics]
    values += values[:1]
    label = display_names.get(name, name)

    ax2.plot(angles, values, "o-", linewidth=2.5, label=label, color=color,
             markersize=7, zorder=3)
    ax2.fill(angles, values, alpha=fill_alpha, color=color, zorder=2)

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(metric_labels, fontsize=12, fontweight="medium")
ax2.set_ylim(0.75, 1.0)
ax2.set_yticks([0.80, 0.85, 0.90, 0.95, 1.0])
ax2.set_yticklabels(["0.80", "0.85", "0.90", "0.95", "1.00"], fontsize=9, color="#555")
ax2.set_title("Model Performance Radar\n(5-Fold Cross-Validation Mean)",
              fontsize=14, fontweight="bold", pad=25)
ax2.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), fontsize=11,
           framealpha=0.95, edgecolor="#ccc")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("paper_comparison_radar.png", dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("Saved: paper_comparison_radar.png")


# ─── Figure 3: Per-Fold Accuracy Box Plot ────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(10, 6))

data = [results[name]["accuracy_values"] for name in model_names]
bp = ax3.boxplot(data, labels=[display_names.get(n, n) for n in model_names], patch_artist=True, widths=0.5,
                 medianprops=dict(color="white", linewidth=2),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5),
                 flierprops=dict(marker="o", markersize=8))

box_colors = ["#E8505B", "#14A76C", "#45B7D1", "#3D5A80"]
for patch, color in zip(bp["boxes"], box_colors[:len(model_names)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
    patch.set_edgecolor("white")
    patch.set_linewidth(1.5)

# Overlay individual fold points
for i, (name, color) in enumerate(zip(model_names, box_colors[:len(model_names)])):
    vals = results[name]["accuracy_values"]
    jitter = np.random.normal(0, 0.04, len(vals))
    ax3.scatter([i + 1 + j for j in jitter], vals, color="white",
                edgecolor=color, s=60, zorder=5, linewidth=1.5)

ax3.set_ylabel("Accuracy", fontsize=14, fontweight="bold", labelpad=10)
ax3.set_title("Per-Fold Accuracy Distribution (5-Fold CV)",
              fontsize=15, fontweight="bold", pad=15)
ax3.grid(axis="y", alpha=0.25, linestyle="--")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.tick_params(axis='both', which='major', labelsize=12)
ax3.set_ylim(0.70, 1.0)

plt.tight_layout()
plt.savefig("paper_comparison_boxplot.png", dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("Saved: paper_comparison_boxplot.png")

print("\nAll publication figures generated successfully!")
