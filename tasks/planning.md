# Planning — dippy-WAN

## Current Priorities

1. **Test with real character images** — Replace stick figure with actual avatar to validate production quality
2. **RunPod/Vast.ai deployment** — Create a lightweight deploy script for cheap WAN 14B runs
3. **Full Gradio app local test** — Run `dippy-app.py` end-to-end locally with CogVideoX-5B
4. **Design alinakai API integration** — sentence → clip endpoint

## Testing Options (RESOLVED — do not revisit)

- **Local RTX 3060 12GB**: VERIFIED — `cogvideo5b` runs at 4.04GB peak VRAM, 6 min/clip at 10 steps.
- **Google Colab T4**: ANSWERED — Claude CANNOT connect to Colab (no API exists). Notebook `Dippy_WAN.ipynb` is ready. User runs manually after `git push`.
- **PACE cluster**: A100 nodes for `wan14b` testing via SLURM.

## Next Steps
- [ ] Colab T4 testing — user must run notebook manually after `git push`
- [ ] Create side-by-side comparison (same avatar + sentence, multiple backends)
- [ ] Test RunPod spot instance with WAN 14B ($0.22/hr RTX 3090)
- [ ] Design alinakai API integration (sentence → clip endpoint)

## Recently Completed

- **LTX-Video 2B local test** — Runs at 0.97GB VRAM but minimal motion output; CogVideoX is superior for this use case
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
