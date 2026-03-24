"""
Unit tests for the Dippy Orchestrator components.

Tests: segment_db CRUD + search, trajectory_engine branch selection,
regen_scheduler priority ordering, stitcher (mock ffmpeg).

Run: pytest orchestrator/tests/test_engine.py -v
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from orchestrator.segment_db import SegmentDB, encode_sentence, _hash_embed
from orchestrator.trajectory_engine import TrajectoryEngine
from orchestrator.regen_scheduler import RegenScheduler
from orchestrator.stitcher import stitch_segments


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary segment database."""
    db = SegmentDB(str(tmp_path / "test.db"))
    yield db
    db.close()


@pytest.fixture
def seeded_db(tmp_db):
    """DB pre-loaded with 5 test segments."""
    sentences = [
        "He jumped",
        "She laughed",
        "He ran fast",
        "She cried",
        "He ate food",
    ]
    for s in sentences:
        tmp_db.add_segment(
            sentence=s,
            video_path=f"/tmp/fake_{s.replace(' ', '_')}.mp4",
            backend="wan14b",
        )
    return tmp_db


# ── segment_db tests ─────────────────────────────────────────────────────────


class TestSegmentDB:
    def test_add_and_count(self, tmp_db):
        assert tmp_db.count() == 0
        tmp_db.add_segment("He jumped", video_path="/tmp/test.mp4")
        assert tmp_db.count() == 1

    def test_add_returns_id(self, tmp_db):
        sid = tmp_db.add_segment("She ran", video_path="/tmp/test.mp4")
        assert isinstance(sid, str)
        assert len(sid) == 12

    def test_get_segment(self, tmp_db):
        sid = tmp_db.add_segment("He walked", video_path="/tmp/walk.mp4", backend="wan14b")
        seg = tmp_db.get_segment(sid)
        assert seg is not None
        assert seg["sentence"] == "He walked"
        assert seg["backend"] == "wan14b"
        assert seg["video_path"] == "/tmp/walk.mp4"

    def test_get_nonexistent(self, tmp_db):
        assert tmp_db.get_segment("nonexistent") is None

    def test_delete_segment(self, tmp_db):
        sid = tmp_db.add_segment("He cried", video_path="/tmp/cry.mp4")
        assert tmp_db.count() == 1
        assert tmp_db.delete_segment(sid) is True
        assert tmp_db.count() == 0
        assert tmp_db.delete_segment(sid) is False

    def test_get_all_segments(self, seeded_db):
        segs = seeded_db.get_all_segments()
        assert len(segs) == 5
        sentences = {s["sentence"] for s in segs}
        assert "He jumped" in sentences
        assert "She laughed" in sentences

    def test_search_nearest_exact_match(self, seeded_db):
        results = seeded_db.search_nearest("He jumped", top_k=1)
        assert len(results) == 1
        assert results[0]["segment"]["sentence"] == "He jumped"
        assert results[0]["similarity"] == pytest.approx(1.0, abs=0.01)

    def test_search_nearest_semantic(self, seeded_db):
        results = seeded_db.search_nearest("He leaped", top_k=3)
        assert len(results) == 3
        # "He jumped" should be in top results (semantically similar)
        sentences = [r["segment"]["sentence"] for r in results]
        assert "He jumped" in sentences

    def test_search_empty_db(self, tmp_db):
        results = tmp_db.search_nearest("anything", top_k=5)
        assert results == []

    def test_search_batch(self, seeded_db):
        results = seeded_db.search_batch(["He jumped", "She laughed"], top_k=1)
        assert len(results) == 2
        assert results[0][0]["segment"]["sentence"] == "He jumped"
        assert results[1][0]["segment"]["sentence"] == "She laughed"

    def test_metadata_stored(self, tmp_db):
        sid = tmp_db.add_segment(
            "He danced",
            video_path="/tmp/dance.mp4",
            metadata={"source": "test", "quality": 0.95},
        )
        seg = tmp_db.get_segment(sid)
        assert seg["metadata"]["source"] == "test"
        assert seg["metadata"]["quality"] == 0.95

    def test_normalization(self, tmp_db):
        sid = tmp_db.add_segment("  He   JUMPED  ", video_path="/tmp/j.mp4")
        seg = tmp_db.get_segment(sid)
        assert seg["sentence_normalized"] == "he jumped"


class TestEmbeddings:
    def test_hash_embed_deterministic(self):
        e1 = _hash_embed("He jumped")
        e2 = _hash_embed("He jumped")
        np.testing.assert_array_equal(e1, e2)

    def test_hash_embed_normalized(self):
        e = _hash_embed("She laughed")
        norm = np.linalg.norm(e)
        assert norm == pytest.approx(1.0, abs=0.01)

    def test_hash_embed_different(self):
        e1 = _hash_embed("He jumped")
        e2 = _hash_embed("She cooked dinner")
        # Should not be identical
        assert not np.allclose(e1, e2)

    def test_encode_sentence_returns_384d(self):
        e = encode_sentence("Test sentence")
        assert e.shape == (384,)
        assert e.dtype == np.float32


# ── trajectory_engine tests ──────────────────────────────────────────────────


class TestTrajectoryEngine:
    def test_select_best_branch(self, seeded_db):
        engine = TrajectoryEngine(seeded_db, similarity_threshold=0.75)
        selection = engine.select_trajectory([
            ["He jumped", "She laughed", "He ran fast"],  # all cached
            ["She cried", "He ate food", "They danced"],  # 2 cached, 1 miss
        ])
        # Branch 0 should win — full coverage
        assert selection.chosen.branch_index == 0
        assert selection.chosen.coverage == pytest.approx(1.0)

    def test_single_branch(self, seeded_db):
        engine = TrajectoryEngine(seeded_db, similarity_threshold=0.75)
        selection = engine.select_trajectory([["He jumped", "She laughed"]])
        assert len(selection.all_branches) == 1
        assert selection.chosen.branch_index == 0

    def test_empty_db_all_gaps(self, tmp_db):
        engine = TrajectoryEngine(tmp_db, similarity_threshold=0.75)
        selection = engine.select_trajectory([["He jumped", "She ran"]])
        assert selection.chosen.confidence == 0.0
        assert selection.chosen.coverage == 0.0
        assert len(selection.chosen.gaps) == 2
        assert all(m.needs_regeneration for m in selection.chosen.matches)

    def test_gaps_identified(self, seeded_db):
        engine = TrajectoryEngine(seeded_db, similarity_threshold=0.99)
        # With threshold=0.99, only exact matches pass
        selection = engine.select_trajectory([
            ["He jumped", "Something completely new"],
        ])
        gaps = selection.chosen.gaps
        # "Something completely new" should be a gap
        gap_sentences = [g.sentence for g in gaps]
        assert "Something completely new" in gap_sentences

    def test_matches_have_correct_positions(self, seeded_db):
        engine = TrajectoryEngine(seeded_db)
        result = engine.evaluate_branch(["He jumped", "She laughed", "He ran fast"])
        for i, m in enumerate(result.matches):
            assert m.position == i

    def test_confidence_is_mean_similarity(self, seeded_db):
        engine = TrajectoryEngine(seeded_db)
        result = engine.evaluate_branch(["He jumped", "She laughed"])
        sims = [m.similarity for m in result.matches]
        assert result.confidence == pytest.approx(np.mean(sims), abs=0.001)


# ── regen_scheduler tests ────────────────────────────────────────────────────


class TestRegenScheduler:
    def _make_gap(self, sentence, position, similarity=0.0):
        from orchestrator.trajectory_engine import SegmentMatch
        return SegmentMatch(
            sentence=sentence,
            position=position,
            segment_id=None,
            segment_sentence=None,
            similarity=similarity,
            video_path=None,
            needs_regeneration=True,
        )

    def test_empty_gaps(self):
        scheduler = RegenScheduler()
        plan = scheduler.build_plan(gaps=[], user_proficiency=0.5)
        assert len(plan.items) == 0
        assert plan.critical_count == 0

    def test_priority_order(self):
        scheduler = RegenScheduler()
        gaps = [
            self._make_gap("Late sentence", position=9, similarity=0.3),
            self._make_gap("Early sentence", position=1, similarity=0.1),
            self._make_gap("Middle sentence", position=5, similarity=0.2),
        ]
        plan = scheduler.build_plan(gaps, user_proficiency=0.5)
        # Earlier position + lower similarity → higher priority
        assert plan.items[0].sentence == "Early sentence"

    def test_critical_detection(self):
        scheduler = RegenScheduler(segment_gen_time_s=100.0)
        gaps = [
            self._make_gap("First sentence", position=0, similarity=0.0),
        ]
        # High proficiency = fast progression = user arrives quickly
        plan = scheduler.build_plan(gaps, user_proficiency=1.0)
        # position=0, proficiency=1.0 → time_to_reach=0s, regen=100s → critical
        assert plan.items[0].is_critical is True
        assert plan.critical_count == 1

    def test_not_critical_slow_user(self):
        scheduler = RegenScheduler(segment_gen_time_s=10.0)
        gaps = [
            self._make_gap("Late sentence", position=10, similarity=0.0),
        ]
        # Slow user, late position → plenty of time
        plan = scheduler.build_plan(gaps, user_proficiency=0.0)
        # position=10 × 30s/sentence ÷ 1.0 speed = 300s to reach, regen=10s
        assert plan.items[0].is_critical is False

    def test_total_regen_time(self):
        scheduler = RegenScheduler(segment_gen_time_s=50.0)
        gaps = [self._make_gap(f"S{i}", i) for i in range(3)]
        plan = scheduler.build_plan(gaps, user_proficiency=0.5)
        assert plan.total_regen_time_s == pytest.approx(150.0)

    def test_time_to_reach_proficiency_scaling(self):
        scheduler = RegenScheduler()
        # Beginner: position 5, speed 1.0x → 5 * 30 = 150s
        t_beginner = scheduler.estimate_time_to_reach(5, 0.0)
        # Fluent: position 5, speed 3.0x → 5 * 10 = 50s
        t_fluent = scheduler.estimate_time_to_reach(5, 1.0)
        assert t_beginner > t_fluent
        assert t_beginner == pytest.approx(150.0)
        assert t_fluent == pytest.approx(50.0)


# ── stitcher tests ───────────────────────────────────────────────────────────


class TestStitcher:
    def test_no_valid_paths_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No valid video segments"):
            stitch_segments(
                ["/nonexistent/a.mp4", "/nonexistent/b.mp4"],
                str(tmp_path / "out.mp4"),
            )

    def test_single_segment_copies(self, tmp_path):
        # Create a minimal video file
        fake_video = tmp_path / "single.mp4"
        fake_video.write_bytes(b"\x00" * 100)

        with patch("orchestrator.stitcher._check_ffmpeg", return_value=True), \
             patch("subprocess.run") as mock_run, \
             patch("orchestrator.stitcher._get_video_duration", return_value=2.0):
            mock_run.return_value = MagicMock(returncode=0)
            result = stitch_segments(
                [str(fake_video)],
                str(tmp_path / "out.mp4"),
            )
            assert result["segments_used"] == 1

    def test_missing_segments_warned(self, tmp_path):
        real = tmp_path / "real.mp4"
        real.write_bytes(b"\x00" * 100)

        with patch("orchestrator.stitcher._check_ffmpeg", return_value=True), \
             patch("subprocess.run") as mock_run, \
             patch("orchestrator.stitcher._get_video_duration", return_value=2.0):
            mock_run.return_value = MagicMock(returncode=0)
            result = stitch_segments(
                [str(real), "/nonexistent/b.mp4"],
                str(tmp_path / "out.mp4"),
            )
            assert result["segments_missing"] == 1  # 1 missing from input
            assert len(result["warnings"]) == 1
            assert "missing" in result["warnings"][0].lower()

    def test_ffmpeg_not_found_raises(self, tmp_path):
        fake = tmp_path / "v.mp4"
        fake.write_bytes(b"\x00" * 100)

        with patch("orchestrator.stitcher._check_ffmpeg", return_value=False):
            with pytest.raises(RuntimeError, match="ffmpeg not found"):
                stitch_segments([str(fake)], str(tmp_path / "out.mp4"))
