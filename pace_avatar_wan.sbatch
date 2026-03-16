#!/bin/bash
#SBATCH -J dippy-avatar-wan
#SBATCH -A gts-yke8
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH -C A100-80GB
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -q inferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu
#SBATCH -o slurm-%j-avatar-wan.out
#SBATCH -e slurm-%j-avatar-wan.err

# ── Dippy avatar test: WAN 14B on PACE A100 ──
# Tests all avatars in avatars/ with forward+reset on "He jumped".
# Expected: ~10 min per avatar (forward+reset), ~30 min for 3 avatars.

set -eo pipefail

module load cuda
module load anaconda3

PROJECT_DIR="$HOME/scratch/dippy-WAN"
CACHE_DIR="$HOME/scratch/hf_cache"
RESULTS_DIR="$PROJECT_DIR/results"
VENV_DIR="$HOME/scratch/dippy_venv"

cd "$PROJECT_DIR"

export HF_HUB_CACHE="$CACHE_DIR"
export HF_HOME="$CACHE_DIR"

mkdir -p "$RESULTS_DIR" "$CACHE_DIR"

echo "=== Dippy Avatar Test: WAN 14B ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "===================================="

# Reuse existing venv (created by prior PACE jobs)
source "$VENV_DIR/bin/activate"

echo "Python: $(which python) ($(python --version))"
echo "Torch: $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"

python pace_avatar_test.py \
  --backend wan14b \
  --cache-dir "$CACHE_DIR" \
  --output-dir "$RESULTS_DIR" \
  --avatar-dir avatars \
  --num-frames 49 \
  --steps 4 \
  --guidance-scale 1.0 \
  --seed 42 \
  --sentence "He jumped"

echo "=== Done ==="
ls -la "$RESULTS_DIR"/avatar_wan14b_*.{png,mp4} 2>/dev/null || echo "(no avatar results)"
