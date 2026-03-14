#!/bin/bash
#SBATCH -J dippy-cogvid-rtx6k
#SBATCH -A gts-yke8
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -q inferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu
#SBATCH -o slurm-%j-dippy-rtx6k.out
#SBATCH -e slurm-%j-dippy-rtx6k.err

# ── Dippy CogVideoX-5B on PACE RTX 6000 (24GB) ──
# Tests CogVideoX-5B on a 24GB GPU — the economical RunPod-equivalent config.
# WAN 14B won't fit on 24GB (needs 53GB VRAM).

set -eo pipefail

module load cuda
module load anaconda3

# ── Paths ──
PROJECT_DIR="$HOME/p-yke8-0/dippy-WAN"
CACHE_DIR="$HOME/scratch/hf_cache"
RESULTS_DIR="$PROJECT_DIR/results"
VENV_DIR="$HOME/scratch/dippy_venv"

cd "$PROJECT_DIR"

export HF_HUB_CACHE="$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export DIPPY_BACKEND="cogvideo5b"

mkdir -p "$RESULTS_DIR" "$CACHE_DIR"

# ── Environment ──
echo "=== PACE Dippy CogVideoX-5B on RTX 6000 ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "Backend: $DIPPY_BACKEND"
echo "Cache: $CACHE_DIR"
echo "============================================="

# ── Virtual environment (reuse existing from A100 job) ──
VENV_MARKER="$VENV_DIR/.torch_version"
WANT_TORCH="2.6"

if [ ! -d "$VENV_DIR" ] || [ ! -f "$VENV_MARKER" ] || ! grep -q "$WANT_TORCH" "$VENV_MARKER"; then
    echo "Creating venv at $VENV_DIR (PyTorch $WANT_TORCH) ..."
    rm -rf "$VENV_DIR"
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    pip install \
      diffusers==0.36.0 transformers==5.1.0 accelerate==1.12.0 \
      huggingface_hub==1.4.1 \
      ftfy peft imageio-ffmpeg opencv-python safetensors sentencepiece \
      tiktoken protobuf torchao
    echo "$WANT_TORCH" > "$VENV_MARKER"
else
    source "$VENV_DIR/bin/activate"
    # Ensure torchao is installed for int8 quantization
    pip install -q torchao 2>/dev/null || true
fi

echo "Python: $(which python) ($(python --version))"
echo "Torch: $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"

# ── Run inference ──
python pace_test_inference.py \
  --backend "$DIPPY_BACKEND" \
  --cache-dir "$CACHE_DIR" \
  --output-dir "$RESULTS_DIR" \
  --num-frames 49 \
  --steps 10 \
  --guidance-scale 6.0 \
  --seed 42

echo "=== Done. Results in $RESULTS_DIR ==="
ls -la "$RESULTS_DIR"/pace_*.png "$RESULTS_DIR"/pace_*.mp4 2>/dev/null || echo "(no output files found)"
