# Dippy Orchestrator

Trajectory alignment, video segment caching, and stitching for language learning animations. Standalone module designed for integration with alinakai.

## What It Does

Given multiple sentence trajectory branches from an LLM, the orchestrator:

1. Searches its cache of pre-rendered video segments
2. Selects the branch with the best cache coverage
3. Flags poorly-aligned segments for regeneration, prioritized by urgency
4. Stitches cached segments into a continuous video

## Setup

```bash
pip install -r orchestrator/requirements.txt
apt install ffmpeg  # for video stitching
```

## Quick Start

```bash
# Start API server
uvicorn orchestrator.app:app --host 0.0.0.0 --port 8100

# Start Gradio UI (for testing)
python -m orchestrator.frontend
```

## API Endpoints

### `POST /select-trajectory`

Select the best trajectory branch from candidates.

```json
{
  "branches": [
    ["He jumped", "She laughed", "He ran"],
    ["She waved", "He cried", "They danced"]
  ],
  "user_proficiency": 0.5,
  "similarity_threshold": 0.75
}
```

Returns: chosen branch, per-sentence alignment scores, and a regeneration plan.

### `POST /stitch`

Stitch segments into a single video.

```json
{
  "segment_ids": ["abc123", "def456", "ghi789"],
  "crossfade_s": 0.3,
  "fps": 24
}
```

### `GET /segments`

List all cached segments.

### `POST /segments`

Add a segment to the cache.

```json
{
  "sentence": "He jumped",
  "video_path": "/path/to/clip.mp4",
  "backend": "wan14b"
}
```

### `POST /search`

Find cached segments similar to a sentence.

### `GET /health`

Check service status and segment count.

## Ingesting Pre-Rendered Clips

```bash
# Batch ingest from rendered clips directory
python -m orchestrator.ingest --clips-dir cache_clips/ --db data/segments.db
```

Each clip needs a `clip_NNN_meta.json` file (produced by `pace_batch_render.py`).

## Configuration

All settings via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DIPPY_DB_PATH` | `data/segments.db` | Segment database path |
| `DIPPY_STITCH_DIR` | `data/stitched` | Stitched video output |
| `DIPPY_SCENE_CACHE` | `data/scene_cache` | LLM prompt cache |
| `DIPPY_SIM_THRESHOLD` | `0.75` | Minimum cosine similarity for cache hit |
| `DIPPY_GEN_TIME_S` | `70.0` | Expected seconds to regenerate one segment |
| `DIPPY_API_PORT` | `8100` | API server port |
| `OPENAI_API_KEY` | — | For LLM scene descriptions (optional) |

## alinakai Integration

```python
import requests

# alinakai generates trajectory branches based on user proficiency
branches = [
    ["He jumped", "She laughed", "He ran"],
    ["She waved", "He cried", "They danced"],
]

# Ask orchestrator to pick the best branch
resp = requests.post("http://localhost:8100/select-trajectory", json={
    "branches": branches,
    "user_proficiency": 0.6,
})
result = resp.json()

chosen = result["selection"]["chosen_branch"]
print(f"Branch {chosen['branch_index']}: confidence={chosen['confidence']}")

# Get segment IDs for stitching
segment_ids = [m["segment_id"] for m in chosen["matches"] if m["segment_id"]]

# Stitch into a video
resp = requests.post("http://localhost:8100/stitch", json={
    "segment_ids": segment_ids,
})
print(f"Video: {resp.json()['output_path']}")

# Check what needs regeneration
regen = result["regeneration_plan"]
for item in regen["items"]:
    print(f"Regen needed: '{item['sentence']}' (priority={item['priority_score']})")
```

## Testing

```bash
pytest orchestrator/tests/ -v
```

## Architecture

```
orchestrator/
├── config.py          — Centralized settings (env vars + defaults)
├── models.py          — Pydantic API contract (16 typed models)
├── segment_db.py      — SQLite + numpy vector DB with embeddings
├── trajectory_engine.py — Branch selection by coverage + confidence
├── regen_scheduler.py — TTR-based regeneration priority queue
├── stitcher.py        — ffmpeg video concatenation with crossfade
├── scene_gen.py       — GPT-4o-mini prompt generation with disk cache
├── ingest.py          — Batch clip ingestion from rendered outputs
├── app.py             — FastAPI endpoints
├── frontend.py        — Gradio test UI (4 tabs)
├── seed_cache.py      — Demo data seeder
└── tests/
    └── test_engine.py — 32 unit tests
```
