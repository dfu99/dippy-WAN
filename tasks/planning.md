# Planning — dippy-WAN

## Current Priorities

1. **Test multi-backend system** — Run CogVideoX-5B and LTX-Video 2B on Colab T4 to verify they work end-to-end
2. **Quality comparison** — Generate same sentences across all 3 backends, compare output quality
3. **Optimize CogVideoX settings** — Find best steps/guidance/resolution for T4 with int8 quantization
4. **RunPod/Vast.ai deployment** — Create a lightweight deploy script for cheap WAN 14B runs

## Next Steps

- [ ] Run `DIPPY_BACKEND=cogvideo5b python dippy-app.py` on free Colab T4
- [ ] Run `DIPPY_BACKEND=ltx2b python dippy-app.py` on free Colab T4
- [ ] Create side-by-side comparison video (same avatar, same sentences, 3 backends)
- [ ] Profile VRAM usage for each backend on T4
- [ ] Test RunPod spot instance with WAN 14B ($0.22/hr RTX 3090)
- [ ] Design alinakai API integration (sentence → clip endpoint)
- [x] Update Dippy_WAN.ipynb to support backend selection

## Recently Completed

- Multi-backend architecture (`backends.py`) with WAN 14B, CogVideoX 5B, LTX-Video 2B
- Refactored `dippy-app.py` to use backend abstraction with UI dropdown
- Backend selection via `DIPPY_BACKEND` env var
- README.md with project overview, milestones, cost comparison
- Forward + reset looping pipeline (original implementation)
- Gradio UI with clip navigation and timeline
- LLM sentence generation via GPT-4o-mini
