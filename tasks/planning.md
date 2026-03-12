# Planning — dippy-WAN

## Current Priorities

1. **Test multi-backend system** — Run CogVideoX-5B and LTX-Video 2B on Colab T4 to verify they work end-to-end
2. **Quality comparison** — Generate same sentences across all 3 backends, compare output quality
3. **Optimize CogVideoX settings** — Find best steps/guidance/resolution for T4 with int8 quantization
4. **RunPod/Vast.ai deployment** — Create a lightweight deploy script for cheap WAN 14B runs

## Testing Options

- **Google Colab T4**: Notebook is ready (`Dippy_WAN.ipynb`). Must be run manually — no programmatic Colab API exists. Auto-detects GPU and selects backend.
- **Local RTX 3060 12GB**: Can run `cogvideo5b` (int8, ~5GB VRAM) and possibly `ltx2b` (~8GB). Not enough for `wan14b` (26GB).
- **PACE cluster**: A100 nodes available for `wan14b` testing.

## Next Steps

- [ ] Run `DIPPY_BACKEND=cogvideo5b python dippy-app.py` on free Colab T4 (manual — notebook ready)
- [ ] Run `DIPPY_BACKEND=ltx2b python dippy-app.py` on free Colab T4 (manual)
- [ ] Test `ltx2b` locally on RTX 3060
- [ ] Create side-by-side comparison video (same avatar, same sentences, 3 backends)
- [ ] Profile VRAM usage for each backend on T4
- [ ] Test RunPod spot instance with WAN 14B ($0.22/hr RTX 3090)
- [ ] Design alinakai API integration (sentence → clip endpoint)

## Recently Completed

- **CogVideoX-5B local test on RTX 3060** — 4.04GB peak VRAM, 23 min generation, sequential CPU offload works
- Updated Dippy_WAN.ipynb to support multi-backend selection
- Multi-backend architecture (`backends.py`) with WAN 14B, CogVideoX 5B, LTX-Video 2B
- Refactored `dippy-app.py` to use backend abstraction with UI dropdown
- Backend selection via `DIPPY_BACKEND` env var
- README.md with project overview, milestones, cost comparison
- Forward + reset looping pipeline (original implementation)
- Gradio UI with clip navigation and timeline
- LLM sentence generation via GPT-4o-mini
