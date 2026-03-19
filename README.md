# Dippy-WAN: Animated Charades for Language Learning

Generate loopable animation clips where a character acts out sentences. Each sentence produces a forward pass (action) and a reset pass (return to neutral), creating seamless clips that chain into trajectories.

## Example

```
Avatar Image → "He jumped" → forward clip → reset clip → loop
                                                          ↓
             → "She waved" → forward clip → reset clip → chain...
```

## Quick Start

```bash
pip install diffusers==0.36.0 transformers==5.1.0 accelerate==1.12.0 \
  huggingface_hub==1.4.1 gradio==6.5.1 \
  spaces ftfy peft imageio-ffmpeg opencv-python safetensors sentencepiece openai

python dippy-app.py
```

Or open `Dippy_WAN.ipynb` in Google Colab.

## Backends

| Backend | Model | Min GPU | Speed |
|---------|-------|---------|-------|
| `wan14b` | WAN 2.1 14B + CausVid LoRA | A100 / RTX 3090 | ~2 min/clip |
| `cogvideo5b` | CogVideoX-5B-I2V | T4 (free Colab) | ~6 min/clip |

Select via `DIPPY_BACKEND=wan14b` env var or the Gradio dropdown.

## API

```bash
python api.py

curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"sentence": "He jumped", "image_path": "avatars/Perplexity.png"}'
```

## Key Files

| File | Purpose |
|------|---------|
| `dippy-app.py` | Gradio UI and generation loop |
| `backends.py` | Model loading and inference |
| `api.py` | REST API |
| `clip_cache.py` | Clip caching by sentence+avatar |

## How It Works

The reset pass uses WAN's `last_image` parameter to condition on both the action pose (first frame) and the original avatar (last frame), producing a smooth return to neutral without a hard cut.

Clips are cached by `(sentence, backend, avatar_hash)` for instant replay.

## License

Research use. See model licenses: [WAN 2.1](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers), [CogVideoX](https://huggingface.co/THUDM/CogVideoX-5b-I2V).
