#!/bin/bash
#SBATCH -J dippy-batch
#SBATCH -A gts-yke8
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH -C A100-80GB
#SBATCH --mem=128G
#SBATCH -t 01:00:00
#SBATCH -q inferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu
#SBATCH -o slurm-%A_%a-batch.out
#SBATCH -e slurm-%A_%a-batch.err
#SBATCH --array=0-49

# ── Dippy batch pre-render: top 50 sentences ──
# Each array task renders one sentence (3-segment: setup→action→reset).
# ~90s inference per sentence on A100.
# Usage: sbatch pace_batch_render.sbatch

set -eo pipefail

module load cuda
module load anaconda3

PROJECT_DIR="$HOME/scratch/dippy-WAN"
CACHE_DIR="$HOME/scratch/hf_cache"
OUTPUT_DIR="$PROJECT_DIR/cache_clips"
VENV_DIR="$HOME/scratch/dippy_venv"

cd "$PROJECT_DIR"

export HF_HUB_CACHE="$CACHE_DIR"
export HF_HOME="$CACHE_DIR"

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

echo "=== Dippy Batch Render: task $SLURM_ARRAY_TASK_ID / 50 ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "================================================="

source "$VENV_DIR/bin/activate"

echo "Python: $(which python) ($(python --version))"

python pace_batch_render.py \
  --sentence-index "$SLURM_ARRAY_TASK_ID" \
  --sentence-file sentences_top50.json \
  --backend wan14b \
  --cache-dir "$CACHE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --image avatars/Perplexity_neutral.png \
  --num-frames 49 \
  --steps 4 \
  --guidance-scale 0.7 \
  --seed 42

echo "=== Task $SLURM_ARRAY_TASK_ID done ==="
