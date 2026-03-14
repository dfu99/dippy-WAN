"""Visualize PACE WAN 14B inference results (obj-011)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("WAN 2.1 14B — PACE A100 Inference Results", fontsize=14, fontweight='bold')

# Forward frames
fwd_files = sorted([f for f in os.listdir("results") if f.startswith("pace_wan14b_fwd_")])
for i, fname in enumerate(fwd_files[:4]):
    img = Image.open(f"results/{fname}")
    axes[0, i].imshow(img)
    frame_num = fname.split("_f")[1].split(".")[0]
    axes[0, i].set_title(f"Forward f{frame_num}", fontsize=10)
    axes[0, i].axis('off')

# Reset frames
rst_files = sorted([f for f in os.listdir("results") if f.startswith("pace_wan14b_rst_")])
for i, fname in enumerate(rst_files[:4]):
    img = Image.open(f"results/{fname}")
    axes[1, i].imshow(img)
    frame_num = fname.split("_f")[1].split(".")[0]
    axes[1, i].set_title(f"Reset f{frame_num}", fontsize=10)
    axes[1, i].axis('off')

# Add metrics text
metrics_text = (
    "Backend: WAN 2.1 14B + CausVid LoRA\n"
    "GPU: A100 80GB PCIe\n"
    "Peak VRAM: 53.13 GB\n"
    "Forward: 313s (49 frames)\n"
    "Reset: 279s (49 frames)\n"
    "Total: 592s (~10 min)\n"
    "PyTorch 2.6+cu124"
)
fig.text(0.98, 0.02, metrics_text, fontsize=9, family='monospace',
         ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='#e3f2fd', alpha=0.8))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig("figures/pace_wan14b_results.png", dpi=150, bbox_inches='tight')
print("Saved figures/pace_wan14b_results.png")
