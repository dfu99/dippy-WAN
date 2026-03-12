#!/bin/bash
# ── RunPod / Vast.ai Deploy Script for Dippy-WAN ────────────────────────────
#
# Deploys the Dippy-WAN Gradio app on a cloud GPU instance.
#
# Recommended instance:
#   - RunPod: RTX 3090 spot ($0.22/hr) or A100 ($1.64/hr)
#   - Vast.ai: RTX 3090 (~$0.20/hr)
#   - Minimum 24GB VRAM for wan14b, 12GB for cogvideo5b
#
# Usage:
#   # On the cloud instance after SSH:
#   git clone https://github.com/<your-repo>/dippy-WAN.git
#   cd dippy-WAN
#   bash deploy_runpod.sh [backend]
#
#   # backend: wan14b (default), cogvideo5b, ltx2b
#
# The script will:
#   1. Install dependencies
#   2. Set up HF cache directory
#   3. Download model weights
#   4. Launch Gradio app with public share link
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

BACKEND="${1:-wan14b}"
CACHE_DIR="${HF_HUB_CACHE:-/workspace/hf_cache}"
OPENAI_KEY_FILE="${OPENAI_KEY_FILE:-}"

echo "═══════════════════════════════════════════════════════════"
echo " Dippy-WAN Deploy — Backend: ${BACKEND}"
echo " Cache: ${CACHE_DIR}"
echo "═══════════════════════════════════════════════════════════"

# ── Check GPU ────────────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    echo ""
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "WARNING: nvidia-smi not found. GPU may not be available."
fi

# ── Install dependencies ─────────────────────────────────────────────────────
echo "Installing Python dependencies..."
pip install -q --upgrade \
    diffusers==0.36.0 \
    transformers==5.1.0 \
    accelerate==1.12.0 \
    huggingface_hub==1.4.1 \
    gradio==6.5.1 \
    spaces ftfy peft imageio-ffmpeg opencv-python \
    safetensors sentencepiece openai tiktoken protobuf

# torchao for CogVideoX int8 quantization (optional, may fail on some setups)
if [ "$BACKEND" = "cogvideo5b" ]; then
    echo "Installing torchao for int8 quantization..."
    pip install -q torchao==0.11.0 || echo "torchao install failed — will use bf16 fallback"
fi

echo "Dependencies installed."

# ── Set up cache ─────────────────────────────────────────────────────────────
mkdir -p "$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export TOKENIZERS_PARALLELISM=false

# ── Set backend ──────────────────────────────────────────────────────────────
export DIPPY_BACKEND="$BACKEND"

# ── VRAM check ───────────────────────────────────────────────────────────────
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
if [ "$BACKEND" = "wan14b" ] && [ "$VRAM_MB" -lt 22000 ]; then
    echo ""
    echo "WARNING: wan14b needs ~26GB VRAM but this GPU has ${VRAM_MB}MB."
    echo "Consider using cogvideo5b instead: bash deploy_runpod.sh cogvideo5b"
    echo ""
fi

# ── OpenAI API key (for LLM sentence generation) ────────────────────────────
if [ -n "$OPENAI_KEY_FILE" ] && [ -f "$OPENAI_KEY_FILE" ]; then
    export OPENAI_API_KEY=$(cat "$OPENAI_KEY_FILE")
    echo "OpenAI API key loaded from $OPENAI_KEY_FILE"
elif [ -n "${OPENAI_API_KEY:-}" ]; then
    echo "OpenAI API key found in environment."
else
    echo "NOTE: No OPENAI_API_KEY set. LLM sentence generation will be unavailable."
    echo "Set it with: export OPENAI_API_KEY=sk-..."
fi

# ── Pre-download model weights ───────────────────────────────────────────────
echo ""
echo "Pre-downloading model weights (this may take a while on first run)..."
python3 -c "
import os
os.environ['HF_HUB_CACHE'] = '$CACHE_DIR'
backend = '$BACKEND'

if backend == 'wan14b':
    from huggingface_hub import snapshot_download, hf_hub_download
    print('Downloading WAN 2.1 14B...')
    snapshot_download('Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', cache_dir='$CACHE_DIR')
    print('Downloading CausVid LoRA...')
    hf_hub_download('Kijai/WanVideo_comfy', 'Wan21_CausVid_14B_T2V_lora_rank32.safetensors', cache_dir='$CACHE_DIR')
elif backend == 'cogvideo5b':
    from huggingface_hub import snapshot_download
    print('Downloading CogVideoX-5B-I2V...')
    snapshot_download('THUDM/CogVideoX-5b-I2V', cache_dir='$CACHE_DIR')
elif backend == 'ltx2b':
    from huggingface_hub import snapshot_download
    print('Downloading LTX-Video 2B...')
    snapshot_download('Lightricks/LTX-Video-0.9.7-dev', cache_dir='$CACHE_DIR')
print('Download complete.')
"

# ── Launch ───────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo " Launching Dippy-WAN (${BACKEND})"
echo " Share link will appear below"
echo "═══════════════════════════════════════════════════════════"
echo ""

python3 dippy-app.py
