"""Generate visualization comparing old vs new prompt pipeline results (obj-008)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle("Pipeline Prompt Upgrade Validation — CogVideoX-5B on RTX 3060\n"
             "Old (basic) vs New (action-style) forward+reset pipeline",
             fontsize=14, fontweight='bold', y=0.98)

# Old pipeline frames (from obj-006)
old_frames = [0, 24, 48, 72, 96]
old_labels = ["F0 (start)", "F24 (mid-fwd)", "F48 (end-fwd)", "F72 (mid-rst)", "F96 (end)"]
for col, (fi, label) in enumerate(zip(old_frames, old_labels)):
    ax = axes[0][col]
    fpath = f"results/pipeline_frame_{fi:03d}.png"
    if os.path.exists(fpath):
        ax.imshow(Image.open(fpath))
    else:
        ax.text(0.5, 0.5, "missing", ha='center', va='center', transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    if col == 0:
        ax.set_ylabel("Old prompt\n(basic)", fontsize=12, fontweight='bold', color='red')
    ax.set_title(label, fontsize=10)

# New pipeline frames
new_frames = [0, 24, 32, 48, 96]
new_labels = ["F0 (start)", "F24 (wave)", "F32 (gesture)", "F48 (end-fwd)", "F96 (end)"]
for col, (fi, label) in enumerate(zip(new_frames, new_labels)):
    ax = axes[1][col]
    fpath = f"results/new_prompt_f{fi:03d}.png"
    if os.path.exists(fpath):
        ax.imshow(Image.open(fpath))
    else:
        ax.text(0.5, 0.5, "missing", ha='center', va='center', transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    if col == 0:
        ax.set_ylabel("New prompt\n(action-style)", fontsize=12, fontweight='bold', color='green')
    ax.set_title(label, fontsize=10)

# Annotations
fig.text(0.92, 0.72, "Facial only\nNo arm motion", fontsize=11, color='red',
         fontweight='bold', va='center')
fig.text(0.92, 0.28, "Arms raised!\nFull-body gesture", fontsize=11, color='green',
         fontweight='bold', va='center')

plt.tight_layout(rect=[0, 0.02, 0.90, 0.94])
plt.savefig('results/pipeline_prompt_upgrade.png', dpi=150, bbox_inches='tight')
print("Saved results/pipeline_prompt_upgrade.png")
