"""Visualize GPU compatibility findings for CogVideoX (obj-012)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.set_title("CogVideoX-5B GPU Compatibility: Flash Attention Required", fontsize=13, fontweight='bold')
ax.axis('off')

gpus = [
    ("RTX 6000 (Turing)", "SM 7.5", "24 GB", "No", "OOM (113GB attn)", "#ffcdd2", "X"),
    ("V100 (Volta)", "SM 7.0", "32 GB", "No", "Won't work", "#ffcdd2", "X"),
    ("RTX 3060 (Ampere)", "SM 8.6", "12 GB", "Yes", "4.0 GB peak, 6 min", "#c8e6c9", "✓"),
    ("RTX 3090 (Ampere)", "SM 8.6", "24 GB", "Yes", "~4 GB peak, faster", "#c8e6c9", "✓"),
    ("A100 (Ampere)", "SM 8.0", "80 GB", "Yes", "Works (tested)", "#c8e6c9", "✓"),
]

headers = ["GPU", "Arch", "VRAM", "Flash Attn", "CogVideoX Result", ""]
col_x = [0.02, 0.25, 0.38, 0.48, 0.60, 0.92]

# Header
for j, h in enumerate(headers[:-1]):
    ax.text(col_x[j], 0.92, h, fontsize=10, fontweight='bold', transform=ax.transAxes, va='center')
ax.plot([0.02, 0.98], [0.87, 0.87], color='black', linewidth=1, transform=ax.transAxes)

for i, (name, arch, vram, flash, result, color, icon) in enumerate(gpus):
    y = 0.78 - i * 0.15
    ax.add_patch(mpatches.FancyBboxPatch((0.01, y - 0.05), 0.97, 0.12,
                 boxstyle="round,pad=0.01", facecolor=color, alpha=0.5,
                 transform=ax.transAxes))
    ax.text(col_x[0], y, name, fontsize=9.5, transform=ax.transAxes, va='center')
    ax.text(col_x[1], y, arch, fontsize=9.5, transform=ax.transAxes, va='center')
    ax.text(col_x[2], y, vram, fontsize=9.5, transform=ax.transAxes, va='center')
    ax.text(col_x[3], y, flash, fontsize=9.5, transform=ax.transAxes, va='center',
            fontweight='bold', color='#2e7d32' if flash == 'Yes' else '#c62828')
    ax.text(col_x[4], y, result, fontsize=9, transform=ax.transAxes, va='center')
    ax.text(col_x[5], y, icon, fontsize=14, transform=ax.transAxes, va='center', ha='center',
            color='#2e7d32' if icon == '✓' else '#c62828', fontweight='bold')

# RunPod recommendation
ax.text(0.5, 0.02, "RunPod: RTX 3090 (Ampere, 24GB, $0.22/hr) — best value for CogVideoX-5B",
        fontsize=10, transform=ax.transAxes, ha='center', va='bottom',
        bbox=dict(boxstyle='round', facecolor='#e3f2fd', alpha=0.8))

plt.tight_layout()
plt.savefig("figures/gpu_compatibility.png", dpi=150, bbox_inches='tight')
print("Saved figures/gpu_compatibility.png")
