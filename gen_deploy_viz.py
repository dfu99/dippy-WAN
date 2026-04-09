"""Generate deployment architecture visualization (obj-009)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(1, 1, figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')
fig.suptitle("Dippy-WAN Deployment Options", fontsize=16, fontweight='bold')

# Deployment options as boxes
options = [
    (0.5, 3.5, "Local RTX 3060", "cogvideo5b\n10 steps, 6 min/clip\n4GB VRAM\n$0/hr", "#90EE90", "VERIFIED"),
    (4, 3.5, "Google Colab T4", "cogvideo5b (int8)\n~5GB VRAM\nFree tier\nNotebook ready", "#FFE4B5", "USER RUNS"),
    (7.5, 3.5, "RunPod RTX 3090", "wan14b\n4 steps, ~30s/clip\n26GB VRAM\n$0.22/hr spot", "#ADD8E6", "SCRIPT READY"),
    (11, 3.5, "PACE A100", "wan14b\n4 steps, ~20s/clip\n80GB VRAM\nFree (GT)", "#DDA0DD", "SLURM READY"),
]

for x, y, title, desc, color, status in options:
    rect = patches.FancyBboxPatch((x, y-2), 2.8, 2.3, boxstyle="round,pad=0.1",
                                   facecolor=color, edgecolor='gray', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x+1.4, y+0.1, title, ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(x+1.4, y-0.9, desc, ha='center', va='center', fontsize=8, family='monospace')
    # Status badge
    badge_color = 'green' if 'VERIFIED' in status else ('orange' if 'USER' in status else 'blue')
    ax.text(x+1.4, y-1.7, status, ha='center', va='center', fontsize=9,
            fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=badge_color, alpha=0.8))

# Arrow from deploy script
ax.annotate('deploy_runpod.sh', xy=(8.9, 1.3), xytext=(8.9, 0.5),
            ha='center', fontsize=9, fontweight='bold', color='blue',
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))

# Cost comparison
ax.text(7, 0.2, "Cost per 100 clips: Local=$0 | Colab=Free | RunPod~$0.60 | PACE=Free",
        ha='center', va='center', fontsize=9, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('figures/deployment_options.png', dpi=150, bbox_inches='tight')
print("Saved figures/deployment_options.png")
