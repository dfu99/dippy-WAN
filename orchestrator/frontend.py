"""
Dippy Orchestrator Frontend — Gradio test interface.

Provides a visual way to test trajectory selection, browse cached segments,
stitch videos, and seed the cache before alinakai integration.

Usage:
    python -m orchestrator.frontend
"""

import json
import os
from pathlib import Path

import gradio as gr

from .segment_db import SegmentDB
from .trajectory_engine import TrajectoryEngine
from .regen_scheduler import RegenScheduler
from .stitcher import stitch_segments

from .config import settings

DB_PATH = settings.db_path
STITCH_DIR = settings.stitch_dir


def _get_db():
    return SegmentDB(DB_PATH)


# ── Tab 1: Trajectory Selection ─────────────────────────────────────────────


def select_trajectory(branches_json: str, proficiency: float, threshold: float):
    """Evaluate branches and return selection results."""
    try:
        branches = json.loads(branches_json)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}", "", ""

    if not isinstance(branches, list) or not all(
        isinstance(b, list) for b in branches
    ):
        return "Expected a list of lists, e.g. [[\"He jumped\", \"She ran\"], ...]", "", ""

    db = _get_db()
    engine = TrajectoryEngine(db, similarity_threshold=threshold)
    selection = engine.select_trajectory(branches)

    scheduler = RegenScheduler()
    regen = scheduler.build_plan(
        gaps=selection.chosen.gaps,
        user_proficiency=proficiency,
        current_confidence=selection.chosen.confidence,
        total_sentences=len(selection.chosen.sentences),
    )

    # Format chosen branch
    chosen = selection.chosen
    chosen_text = f"**Branch {chosen.branch_index}** — confidence: {chosen.confidence:.3f}, coverage: {chosen.coverage:.1%}\n\n"
    for m in chosen.matches:
        icon = "✅" if not m.needs_regeneration else "❌"
        cached = f"→ cached: \"{m.segment_sentence}\" (sim={m.similarity:.3f})" if m.segment_id else "→ NO MATCH"
        chosen_text += f"{icon} **{m.position}.** \"{m.sentence}\" {cached}\n"

    # Format all branches comparison
    all_text = ""
    for b in selection.all_branches:
        marker = " ← CHOSEN" if b.branch_index == chosen.branch_index else ""
        all_text += f"Branch {b.branch_index}: conf={b.confidence:.3f}, coverage={b.coverage:.1%}{marker}\n"

    # Format regen plan
    if regen.items:
        regen_text = f"**{len(regen.items)} segments need regeneration** ({regen.critical_count} critical)\n"
        regen_text += f"Total regen time: {regen.total_regen_time_s:.0f}s\n\n"
        for item in regen.items:
            crit = "🔴 CRITICAL" if item.is_critical else "🟡"
            regen_text += (
                f"{crit} \"{item.sentence}\" (pos {item.position}) — "
                f"priority={item.priority_score:.3f}, "
                f"reach={item.time_to_reach_s:.0f}s, regen={item.time_to_regen_s:.0f}s\n"
            )
    else:
        regen_text = "✅ All segments cached — no regeneration needed!"

    db.close()
    return chosen_text, all_text, regen_text


# ── Tab 2: Segment Browser ──────────────────────────────────────────────────


def list_segments():
    db = _get_db()
    segs = db.get_all_segments()
    db.close()
    if not segs:
        return "No segments in cache. Use the 'Seed Cache' tab to add some."
    text = f"**{len(segs)} segments cached:**\n\n"
    for s in segs:
        exists = "📁" if Path(s["video_path"]).exists() else "⚠️ MISSING"
        text += f"- `{s['id']}` — \"{s['sentence']}\" [{s['backend']}] {exists}\n"
    return text


def search_segments(query: str, top_k: int):
    if not query.strip():
        return "Enter a sentence to search."
    db = _get_db()
    results = db.search_nearest(query, top_k=int(top_k))
    db.close()
    if not results:
        return "No results found."
    text = f"**Top {len(results)} matches for \"{query}\":**\n\n"
    for r in results:
        seg = r["segment"]
        text += f"- **{r['similarity']:.4f}** — \"{seg['sentence']}\" (`{seg['id']}`)\n"
    return text


# ── Tab 3: Stitcher ─────────────────────────────────────────────────────────


def stitch_selected(segment_ids_text: str, crossfade: float):
    ids = [s.strip() for s in segment_ids_text.split(",") if s.strip()]
    if not ids:
        return "Enter comma-separated segment IDs.", None

    db = _get_db()
    video_paths = []
    for sid in ids:
        seg = db.get_segment(sid)
        if seg is None:
            db.close()
            return f"Segment {sid} not found.", None
        if not Path(seg["video_path"]).exists():
            db.close()
            return f"Video file missing: {seg['video_path']}", None
        video_paths.append(seg["video_path"])
    db.close()

    output_dir = Path(STITCH_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / f"stitched_{'_'.join(ids[:3])}.mp4")

    try:
        result = stitch_segments(
            video_paths=video_paths,
            output_path=output_path,
            crossfade_duration=crossfade,
        )
        info = (
            f"Stitched {result['segments_used']} segments → {result['duration_s']:.1f}s\n"
            f"Output: {result['output_path']}"
        )
        return info, result["output_path"]
    except Exception as e:
        return f"Stitch failed: {e}", None


# ── Tab 4: Seed Cache ────────────────────────────────────────────────────────


def seed_from_results():
    """Seed the DB with existing demo clips from results/."""
    from .seed_cache import seed_demo_clips
    count = seed_demo_clips(DB_PATH)
    return f"Seeded {count} segments into the database."


def add_single_segment(sentence: str, video_path: str, backend: str):
    if not sentence.strip() or not video_path.strip():
        return "Both sentence and video path are required."
    db = _get_db()
    sid = db.add_segment(
        sentence=sentence.strip(),
        video_path=video_path.strip(),
        backend=backend,
    )
    db.close()
    return f"Added segment `{sid}` for \"{sentence}\""


# ── Build UI ─────────────────────────────────────────────────────────────────


EXAMPLE_BRANCHES = json.dumps(
    [
        ["He jumped", "She laughed", "He ran"],
        ["She waved", "He cried", "They danced"],
        ["He jumped", "He ran", "She laughed at him"],
    ],
    indent=2,
)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Dippy Orchestrator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎭 Dippy Orchestrator\nTrajectory selection, segment caching, and video stitching")

        with gr.Tab("🎯 Trajectory Selection"):
            gr.Markdown("Paste trajectory branches as JSON. The system selects the branch with the best cached segment coverage.")
            branches_input = gr.Textbox(
                label="Trajectory Branches (JSON)",
                value=EXAMPLE_BRANCHES,
                lines=8,
            )
            with gr.Row():
                proficiency = gr.Slider(0, 1, value=0.5, step=0.1, label="User Proficiency")
                threshold = gr.Slider(0, 1, value=0.75, step=0.05, label="Similarity Threshold")
            select_btn = gr.Button("Select Best Trajectory", variant="primary")
            chosen_output = gr.Markdown(label="Chosen Branch")
            branches_output = gr.Textbox(label="All Branches Comparison", lines=5)
            regen_output = gr.Markdown(label="Regeneration Plan")
            select_btn.click(
                select_trajectory,
                [branches_input, proficiency, threshold],
                [chosen_output, branches_output, regen_output],
            )

        with gr.Tab("📦 Segment Browser"):
            list_btn = gr.Button("Refresh Segment List")
            segments_display = gr.Markdown()
            list_btn.click(list_segments, [], [segments_display])
            gr.Markdown("---")
            search_input = gr.Textbox(label="Search by sentence", placeholder="He jumped")
            search_k = gr.Slider(1, 20, value=5, step=1, label="Top K")
            search_btn = gr.Button("Search")
            search_output = gr.Markdown()
            search_btn.click(
                search_segments, [search_input, search_k], [search_output]
            )

        with gr.Tab("🎬 Stitcher"):
            gr.Markdown("Enter comma-separated segment IDs to stitch into a single video.")
            stitch_ids = gr.Textbox(label="Segment IDs (comma-separated)", placeholder="abc123, def456, ghi789")
            crossfade_slider = gr.Slider(0, 1.0, value=0.3, step=0.1, label="Crossfade (seconds)")
            stitch_btn = gr.Button("Stitch Video", variant="primary")
            stitch_info = gr.Textbox(label="Stitch Result")
            stitch_video = gr.Video(label="Stitched Video")
            stitch_btn.click(
                stitch_selected,
                [stitch_ids, crossfade_slider],
                [stitch_info, stitch_video],
            )

        with gr.Tab("🌱 Seed Cache"):
            gr.Markdown("Populate the segment database with existing demo clips or add new segments manually.")
            seed_btn = gr.Button("Seed from results/ directory", variant="primary")
            seed_output = gr.Textbox(label="Seed Result")
            seed_btn.click(seed_from_results, [], [seed_output])
            gr.Markdown("---")
            gr.Markdown("### Add Single Segment")
            add_sentence = gr.Textbox(label="Sentence", placeholder="He jumped")
            add_path = gr.Textbox(label="Video Path", placeholder="results/demo_trajectory_wan14b.mp4")
            add_backend = gr.Dropdown(["wan14b", "cogvideo5b", "ltx2b"], value="wan14b", label="Backend")
            add_btn = gr.Button("Add Segment")
            add_output = gr.Textbox(label="Result")
            add_btn.click(
                add_single_segment,
                [add_sentence, add_path, add_backend],
                [add_output],
            )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7861)
