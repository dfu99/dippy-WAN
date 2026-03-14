"""Generate visualization for PACE cluster migration objective (obj-011)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Job timeline
ax = axes[0]
ax.set_title("PACE Job Submission Timeline", fontsize=13, fontweight='bold')

jobs = [
    ("Job 4932682", 0, 0.03, "red", "FAILED\nNo torch module"),
    ("Job 4932774", 1, 24.5, "orange", "FAILED\nCUBLAS LoRA error"),
    ("Job 4933019", 2, None, "dodgerblue", "PENDING\nFix applied"),
]

for label, y, duration, color, status in jobs:
    if duration is not None:
        ax.barh(y, duration, height=0.5, color=color, alpha=0.8, edgecolor='black', linewidth=0.8)
        ax.text(duration + 0.5, y, f"{duration:.0f} min — {status}", va='center', fontsize=9)
    else:
        ax.barh(y, 2, height=0.5, color=color, alpha=0.5, edgecolor='black', linewidth=0.8, linestyle='--')
        ax.text(3, y, status, va='center', fontsize=9, color='dodgerblue')
    ax.text(-0.5, y, label, va='center', ha='right', fontsize=9, fontweight='bold')

ax.set_xlim(-8, 40)
ax.set_ylim(-0.8, 2.8)
ax.set_xlabel("Wall Time (minutes)")
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Right: Issues fixed
ax2 = axes[1]
ax2.set_title("Issues Diagnosed & Fixed", fontsize=13, fontweight='bold')
ax2.axis('off')

issues = [
    ("1. No Python venv", "System Python 3.9 has no torch.\nFix: Create venv with anaconda3 module,\ninstall PyTorch from cu128 wheel index", "green"),
    ("2. CUBLAS LoRA error", "peft fuse_lora() fails on CUDA bf16.\nFix: Fuse LoRA on CPU before\n.to('cuda') — avoids CUBLAS matmul", "green"),
    ("3. GPU policy", "Local RTX 3060 crashes system.\nFix: All inference via PACE SLURM.\nMemory + planning updated", "green"),
]

for i, (title, desc, color) in enumerate(issues):
    y = 0.85 - i * 0.33
    ax2.add_patch(mpatches.FancyBboxPatch((0.02, y - 0.12), 0.96, 0.28,
                  boxstyle="round,pad=0.02", facecolor='#e8f5e9', edgecolor=color, linewidth=1.5))
    ax2.text(0.06, y + 0.08, title, fontsize=11, fontweight='bold', color='#2e7d32',
             transform=ax2.transAxes)
    ax2.text(0.06, y - 0.06, desc, fontsize=8.5, color='#333',
             transform=ax2.transAxes, verticalalignment='top', linespacing=1.4)

plt.tight_layout()
plt.savefig("figures/pace_migration_timeline.png", dpi=150, bbox_inches='tight')
print("Saved figures/pace_migration_timeline.png")
