"""Visualize the multi-sentence demo results (obj-013)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle("Dippy Demo — 5-Sentence Charades Chain (WAN 14B, PACE A100)", fontsize=13, fontweight='bold')

sentences = [
    "He jumped",
    "She laughed",
    "She laughed at him",
    "He ran",
    "He ran because\nit started raining",
]

for i, (ax, sent) in enumerate(zip(axes, sentences)):
    img_path = f"results/demo_s{i+1}_fwd_mid.png"
    if os.path.exists(img_path):
        img = Image.open(img_path)
        ax.imshow(img)
    ax.set_title(f'"{sent}"', fontsize=9, style='italic')
    ax.axis('off')
    ax.text(0.5, -0.05, f"Scene {i+1}", transform=ax.transAxes, ha='center', fontsize=8, color='gray')

fig.text(0.5, 0.01, "481 frames | 20s at 24fps | 46 min inference | 53GB VRAM | PyTorch 2.6+cu124",
         fontsize=9, ha='center', family='monospace',
         bbox=dict(boxstyle='round', facecolor='#e8f5e9', alpha=0.8))

plt.tight_layout(rect=[0, 0.06, 1, 0.92])
plt.savefig("figures/demo_trajectory_keyframes.png", dpi=150, bbox_inches='tight')
print("Saved figures/demo_trajectory_keyframes.png")
