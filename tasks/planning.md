# Planning — dippy-WAN

## Current Priorities

1. **Test LTX-Video 2B locally** — Second low-VRAM backend, should also fit on RTX 3060
2. **Quality comparison** — Generate same sentences across backends, compare output quality
3. **Optimize CogVideoX settings** — Reduce steps (30→15?) to cut generation time on local GPU
4. **RunPod/Vast.ai deployment** — Create a lightweight deploy script for cheap WAN 14B runs

## Testing Options (RESOLVED — do not revisit)

- **Local RTX 3060 12GB**: VERIFIED — `cogvideo5b` runs at 4.04GB peak VRAM, 6 min/clip at 10 steps.
- **Google Colab T4**: ANSWERED — Claude CANNOT connect to Colab (no API exists). Notebook `Dippy_WAN.ipynb` is ready. User runs manually after `git push`.
- **PACE cluster**: A100 nodes for `wan14b` testing via SLURM.

## Next Steps

- [ ] Test `ltx2b` locally on RTX 3060 (retry — first attempt OOM-killed during model download; set HF_TOKEN first)
- [ ] Colab T4 testing — user must run notebook manually after `git push`
- [ ] Create side-by-side comparison (same avatar + sentence, multiple backends)
- [ ] Test RunPod spot instance with WAN 14B ($0.22/hr RTX 3090)
- [ ] Design alinakai API integration (sentence → clip endpoint)

## Recently Completed

- **CogVideoX step reduction** — 10 steps (6 min) matches 30 steps (23 min) quality; 4x speedup for local dev
- **Colab connectivity question resolved** — no API exists; notebook is ready, user runs manually after git push
- **CogVideoX-5B local test on RTX 3060** — 4.04GB peak VRAM, 23 min generation, sequential CPU offload works
- Updated Dippy_WAN.ipynb to support multi-backend selection
- Multi-backend architecture (`backends.py`) with WAN 14B, CogVideoX 5B, LTX-Video 2B
- Refactored `dippy-app.py` to use backend abstraction with UI dropdown
- Backend selection via `DIPPY_BACKEND` env var
- README.md with project overview, milestones, cost comparison
- Forward + reset looping pipeline (original implementation)
- Gradio UI with clip navigation and timeline
- LLM sentence generation via GPT-4o-mini
