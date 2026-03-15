#!/bin/bash
#SBATCH -J dippy-demo
#SBATCH -A gts-yke8
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH -C A100-80GB
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -q inferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu
#SBATCH -o slurm-%j-dippy-demo.out
#SBATCH -e slurm-%j-dippy-demo.err

# ── Dippy multi-sentence demo on PACE A100 ──
# Generates a 5-sentence chained charades demo with WAN 14B.
# 10 inference passes (forward+reset per sentence), ~50 min total.

set -eo pipefail

module load cuda
module load anaconda3

PROJECT_DIR="$HOME/p-yke8-0/dippy-WAN"
CACHE_DIR="$HOME/scratch/hf_cache"
RESULTS_DIR="$PROJECT_DIR/results"
VENV_DIR="$HOME/scratch/dippy_venv"

cd "$PROJECT_DIR"

export HF_HUB_CACHE="$CACHE_DIR"
export HF_HOME="$CACHE_DIR"

mkdir -p "$RESULTS_DIR" "$CACHE_DIR"

echo "=== Dippy Multi-Sentence Demo ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "================================="

# Reuse existing venv
source "$VENV_DIR/bin/activate"

echo "Python: $(which python) ($(python --version))"
echo "Torch: $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"

python pace_demo_trajectory.py \
  --backend wan14b \
  --cache-dir "$CACHE_DIR" \
  --output-dir "$RESULTS_DIR" \
  --num-frames 49 \
  --steps 30 \
  --guidance-scale 6.0 \
  --seed 42

echo "=== Done ==="
ls -la "$RESULTS_DIR"/demo_*.{png,mp4} 2>/dev/null || echo "(no demo files)"
