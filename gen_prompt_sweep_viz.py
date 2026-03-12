"""Generate comparison visualization for prompt engineering sweep (obj-007)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os

variants = [
    ("basic", "Basic: 'waves hello with right hand'", "gs=6.0"),
    ("detailed", "Detailed: 'raises arm high and waves...'", "gs=6.0"),
    ("action", "Action: 'raises both arms overhead, big waving'", "gs=6.0"),
    ("gs9", "Detailed prompt", "gs=9.0"),
    ("gs12", "Detailed prompt", "gs=12.0"),
]

frame_indices = [0, 16, 32, 48]
frame_labels = ["Frame 0", "Frame 16", "Frame 32 (mid)", "Frame 48 (end)"]

fig, axes = plt.subplots(5, 4, figsize=(16, 20))
fig.suptitle("Prompt Engineering Sweep — CogVideoX-5B on RTX 3060\n"
             "Goal: maximize arm/body gesture motion", fontsize=16, fontweight='bold', y=0.98)

for row, (key, desc, gs) in enumerate(variants):
    for col, fi in enumerate(frame_indices):
        ax = axes[row][col]
        fpath = f"results/prompt_{key}_f{fi:03d}.png"
        if os.path.exists(fpath):
            img = Image.open(fpath)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "missing", ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(f"{desc}\n({gs})", fontsize=9, rotation=0, labelpad=180, ha='left', va='center')
        if row == 0:
            ax.set_title(frame_labels[col], fontsize=11)

# Add result annotations
results = [
    "Facial only — no arm motion",
    "Minimal motion — slight face change",
    "BEST — both arms raised overhead!",
    "Less motion, character shrinks",
    "Blurry, minimal motion",
]
for row, result in enumerate(results):
    color = 'green' if 'BEST' in result else ('red' if 'Blurry' in result else 'orange')
    axes[row][3].annotate(result, xy=(1.05, 0.5), xycoords='axes fraction',
                          fontsize=10, fontweight='bold', color=color, va='center')

plt.tight_layout(rect=[0.15, 0.02, 0.85, 0.96])
plt.savefig('results/prompt_sweep_comparison.png', dpi=150, bbox_inches='tight')
print("Saved results/prompt_sweep_comparison.png")
