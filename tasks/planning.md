# Planning — dippy-WAN

## Current Priorities

1. **Deploy orchestrator for testing** — Run orchestrator frontend locally, seed cache with demo clips, test full pipeline end-to-end.
2. **LLM scene description generation** — Auto-generate setup/action prompts for arbitrary sentences via GPT-4o-mini (replaces hand-crafted SCENE_SETUP/SCENE_ACTION dicts).
3. **alinakai integration** — Connect orchestrator API to alinakai's trajectory generation. Wire POST /select-trajectory into the learning flow.

## GPU Policy

- **Local RTX 3060**: DO NOT USE for inference or training — crashes the system. Only for tiny tests (<100MB VRAM).
- **PACE cluster**: Primary compute target. A100/H100 nodes via SLURM. Account: gts-yke8, user: dfu71.
- **Google Colab T4**: User runs notebook manually after `git push`.
- **RunPod/Vast.ai**: Spot instances for on-demand testing.

## Next Steps
- [ ] Colab T4 testing — user must run notebook manually after `git push`
- [ ] Create side-by-side comparison (same avatar + sentence, multiple backends)
- [ ] Test RunPod spot instance with `deploy_runpod.sh` ($0.22/hr RTX 3090)
- [ ] Add async job queue to API for production use
- [ ] Client SDK / example for alinakai integration

## Recently Completed

- **E2E integration test passed** — 50 segments → 3 branches → selection → 5 segments stitched → 30.2s video. Fixed stitcher for mpeg4 clips. Visualization generated. Committed 4496d26, pushed.
- **Requirements + README with integration guide** — orchestrator/requirements.txt + README.md: setup, API docs, config table, alinakai integration code sample. Committed 4561580.
- **Test suite: 32 tests all passing** — orchestrator/tests/test_engine.py: segment_db (12), embeddings (4), trajectory_engine (6), regen_scheduler (6), stitcher (4). Committed 1b7c8bb.
- **Centralized config system** — orchestrator/config.py: Settings dataclass with env var overrides. Wired into app.py, scene_gen.py, frontend.py. Committed f5ab675.
- **Pydantic API contract models** — orchestrator/models.py: 16 typed models for alinakai integration. app.py refactored to use response_model. Committed 714aee9.
- **LLM scene description generator** — orchestrator/scene_gen.py: GPT-4o-mini generates setup/action prompts for any sentence. Disk cache, fallback mode. Integrated into pace_batch_render.py. Committed 694685d.
- **50 clips ingested into segment DB** — All 50 batch renders (job 5389538) completed, fetched, and ingested via orchestrator/ingest.py. DB has 50 segments with embeddings. Committed 76b335e.
- **Batch pre-render top 50 sentences** — SLURM array job 5389538 (tasks 0-49), one A100 per sentence. 3-segment pipeline, guidance 0.7, Perplexity neutral avatar. Output to cache_clips/. Committed 6c36709.
- **Standalone orchestrator built** — `orchestrator/` package: segment vector DB, trajectory alignment engine, regen scheduler, video stitcher, FastAPI + Gradio UI. Committed 6a5125a, pushed. All components tested.
- **3-segment pipeline validated** — Job 5268474: setup→action→reset per sentence. 721 frames, 30s at 24fps, 6.8 min inference. Neutral bald avatar with wigs/costumes/backgrounds. Guidance 0.7. Committed and pushed (f377828).
- **Rich prompt + neutral avatar trajectory validated** — Job 5143042: guidance 0.7, bald avatar with wigs/accessories, stage-prop backgrounds. Avatar morphs (wigs, headbands, expressions, backgrounds) while preserving identity. Major improvement over generic prompts.
- **Multi-sentence trajectory with last_image** — Job 5055600 completed. 5 sentences, 481 frames, 20s at 24fps, 4.7 min inference. WAN 14B + Perplexity avatar with last_image conditioning on all reset passes. `results/demo_trajectory_wan14b.mp4`
- **README simplified** — 209→62 lines. Removed internal plans, milestones, methodology. Committed and pushed (54442df).
- **last_image loop closure validated** — WAN's native first+last frame conditioning works. MSE 2088→775 (63% reduction), similarity 96.8%→98.8%. Fixed diffusers 0.36.0 batch dim bug with encode_image monkey-patch. Jobs: 5050210 (failed), 5052890 (success).
- **Avatar loop closure + last_image conditioning** — Ran 3 avatars × 2 backends on PACE. LTX unusable. WAN+Perplexity best. Discovered faked loop closure. Found & implemented WAN's native `last_image` parameter.
- **PACE data migration to scratch** — Moved all data from `~/p-yke8-0/dippy-WAN/` to `~/scratch/dippy-WAN/`. Updated all 5 sbatch scripts, CLAUDE.md, README.md, lessons.md. Old p-yke8-0 path deprecated.
- **Video clip caching system** — `clip_cache.py` with ClipCache class, integrated into `dippy-app.py`. Cache by (sentence, backend, avatar hash). Normalized matching, file-based persistence, hit rate display.
- **Avatar test scripts ready** — pace_avatar_test.py + SLURM scripts for WAN (A100) and LTX (A100). 3 AI-generated avatars (ChatGPT, Gemini, Perplexity). Blocked on VPN for submission.
- **5-sentence demo video** — Job 4953901 completed on A100. 481 frames, 20s at 24fps, 46 min inference. `results/demo_trajectory_wan14b.mp4`
- **PACE RTX 6000 test** — CogVideoX-5B CANNOT run on RTX 6000 (Turing). Needs Ampere+ for Flash Attention. Tested 5 job iterations. RunPod recommendation: RTX 3090 ($0.22/hr).
- **PACE WAN 14B inference SUCCESS** — Job 4934779 completed on A100. Forward+reset in 10 min, 53GB VRAM. Fixed: venv setup, LoRA CPU fusion, PyTorch 2.10 CUBLAS bug (downgraded to 2.6), 128G RAM for model loading, disk quota cleanup.
- **alinakai API integration** — `api.py` with FastAPI REST endpoint: POST /generate (sentence+image→clip), GET /clips, GET /backends, GET /health
- **RunPod deploy script** — `deploy_runpod.sh` created with one-command deployment, VRAM checks, weight pre-download
- **Gradio local launch** — UI wired with CogVideoX-5B defaults, backend-aware duration/steps/guidance, validated with pipeline test
- **Prompt engineering sweep** — Action-style prompts ("raises both arms overhead") produce dramatic gestures; basic/detailed prompts only change faces. Higher guidance (9, 12) hurts. gs=6.0 is optimal.
- **Forward+reset pipeline test** — Full loop works locally, 11.6 min/sentence, loop closes correctly
- **Cartoon avatar test** — CogVideoX animates character with prompt-appropriate wave/point gestures, style preserved
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
