"""
Segment Vector Database — stores cached video segments with prompt embeddings.

Each segment represents one sentence's full 3-phase clip (setup→action→reset).
Segments are searchable by cosine similarity against sentence embeddings.

Uses SQLite for metadata and numpy for embedding storage/search.
Embeddings are generated via sentence-transformers (all-MiniLM-L6-v2).
"""

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np


EMBED_DIM = 384  # all-MiniLM-L6-v2 dimension (also used by fallback)
_encoder = None
_use_fallback = False


def get_encoder():
    """Lazy-load the sentence transformer encoder.

    Falls back to TF-IDF-style hashing if sentence-transformers is unavailable.
    """
    global _encoder, _use_fallback
    if _encoder is not None:
        return _encoder
    try:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    except (ImportError, Exception):
        _use_fallback = True
        _encoder = "fallback"
    return _encoder


def _hash_embed(sentence: str) -> np.ndarray:
    """Deterministic hash-based embedding (fallback when sentence-transformers unavailable).

    Produces a normalized vector from character n-gram hashes. Not as good as
    a trained model but sufficient for development/testing.
    """
    import hashlib
    vec = np.zeros(EMBED_DIM, dtype=np.float32)
    words = sentence.lower().strip().split()
    for w in words:
        for n in range(2, 5):  # char 2-4 grams
            for i in range(max(1, len(w) - n + 1)):
                gram = w[i : i + n]
                h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
                idx = h % EMBED_DIM
                vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def encode_sentence(sentence: str) -> np.ndarray:
    """Encode a sentence into a normalized embedding vector."""
    enc = get_encoder()
    if _use_fallback:
        return _hash_embed(sentence)
    emb = enc.encode(sentence, normalize_embeddings=True)
    return emb.astype(np.float32)


def encode_sentences(sentences: list[str]) -> np.ndarray:
    """Batch-encode sentences into normalized embedding vectors."""
    enc = get_encoder()
    if _use_fallback:
        return np.stack([_hash_embed(s) for s in sentences])
    embs = enc.encode(sentences, normalize_embeddings=True, batch_size=64)
    return embs.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalized vectors."""
    return float(np.dot(a, b))


def cosine_similarity_matrix(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix between query and corpus embeddings.

    Args:
        queries: (N, D) normalized embeddings
        corpus: (M, D) normalized embeddings

    Returns:
        (N, M) similarity matrix
    """
    return queries @ corpus.T


class SegmentDB:
    """SQLite-backed segment database with embedding search."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS segments (
        id TEXT PRIMARY KEY,
        sentence TEXT NOT NULL,
        sentence_normalized TEXT NOT NULL,
        embedding BLOB NOT NULL,
        video_path TEXT NOT NULL,
        backend TEXT DEFAULT 'wan14b',
        avatar_hash TEXT DEFAULT '',
        duration_s REAL DEFAULT 0.0,
        num_frames INTEGER DEFAULT 0,
        fps INTEGER DEFAULT 24,
        prompt_setup TEXT DEFAULT '',
        prompt_action TEXT DEFAULT '',
        prompt_reset TEXT DEFAULT '',
        metadata TEXT DEFAULT '{}',
        created_at REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_sentence ON segments(sentence_normalized);
    CREATE INDEX IF NOT EXISTS idx_backend ON segments(backend);
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self.SCHEMA)
        self._embedding_cache: Optional[tuple[list[str], np.ndarray]] = None

    def close(self):
        self._conn.close()

    def _invalidate_cache(self):
        self._embedding_cache = None

    def _normalize(self, sentence: str) -> str:
        return " ".join(sentence.lower().strip().split())

    def add_segment(
        self,
        sentence: str,
        video_path: str,
        backend: str = "wan14b",
        avatar_hash: str = "",
        duration_s: float = 0.0,
        num_frames: int = 0,
        fps: int = 24,
        prompt_setup: str = "",
        prompt_action: str = "",
        prompt_reset: str = "",
        metadata: Optional[dict] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> str:
        """Add a segment to the database. Returns segment ID."""
        segment_id = uuid.uuid4().hex[:12]
        if embedding is None:
            embedding = encode_sentence(sentence)
        emb_blob = embedding.tobytes()
        meta_json = json.dumps(metadata or {})

        self._conn.execute(
            """INSERT INTO segments
               (id, sentence, sentence_normalized, embedding, video_path,
                backend, avatar_hash, duration_s, num_frames, fps,
                prompt_setup, prompt_action, prompt_reset, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                segment_id,
                sentence,
                self._normalize(sentence),
                emb_blob,
                str(video_path),
                backend,
                avatar_hash,
                duration_s,
                num_frames,
                fps,
                prompt_setup,
                prompt_action,
                prompt_reset,
                meta_json,
                time.time(),
            ),
        )
        self._conn.commit()
        self._invalidate_cache()
        return segment_id

    def get_segment(self, segment_id: str) -> Optional[dict]:
        """Get a segment by ID."""
        row = self._conn.execute(
            "SELECT * FROM segments WHERE id = ?", (segment_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def get_all_segments(self) -> list[dict]:
        """Return all segments as dicts (without embeddings for speed)."""
        rows = self._conn.execute(
            "SELECT id, sentence, sentence_normalized, video_path, backend, "
            "avatar_hash, duration_s, num_frames, fps, created_at FROM segments"
        ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]

    def delete_segment(self, segment_id: str) -> bool:
        cursor = self._conn.execute(
            "DELETE FROM segments WHERE id = ?", (segment_id,)
        )
        self._conn.commit()
        self._invalidate_cache()
        return cursor.rowcount > 0

    def _load_all_embeddings(self) -> tuple[list[str], np.ndarray]:
        """Load all segment IDs and embeddings into memory for search."""
        if self._embedding_cache is not None:
            return self._embedding_cache
        rows = self._conn.execute("SELECT id, embedding FROM segments").fetchall()
        if not rows:
            return [], np.empty((0, 384), dtype=np.float32)
        ids = [r["id"] for r in rows]
        dim = len(np.frombuffer(rows[0]["embedding"], dtype=np.float32))
        embs = np.stack(
            [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
        )
        self._embedding_cache = (ids, embs)
        return ids, embs

    def search_nearest(
        self, sentence: str, top_k: int = 5, embedding: Optional[np.ndarray] = None
    ) -> list[dict]:
        """Find the top-k most similar segments to a sentence.

        Returns list of dicts with 'segment' and 'similarity' keys.
        """
        if embedding is None:
            embedding = encode_sentence(sentence)
        ids, corpus = self._load_all_embeddings()
        if len(ids) == 0:
            return []
        sims = cosine_similarity_matrix(embedding.reshape(1, -1), corpus)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            seg = self.get_segment(ids[idx])
            if seg:
                results.append({"segment": seg, "similarity": float(sims[idx])})
        return results

    def search_batch(
        self, sentences: list[str], top_k: int = 1
    ) -> list[list[dict]]:
        """Batch search: find nearest segments for multiple sentences at once."""
        embeddings = encode_sentences(sentences)
        ids, corpus = self._load_all_embeddings()
        if len(ids) == 0:
            return [[] for _ in sentences]
        sim_matrix = cosine_similarity_matrix(embeddings, corpus)
        results = []
        for i in range(len(sentences)):
            row_sims = sim_matrix[i]
            top_indices = np.argsort(row_sims)[::-1][:top_k]
            matches = []
            for idx in top_indices:
                seg = self.get_segment(ids[idx])
                if seg:
                    matches.append({
                        "segment": seg,
                        "similarity": float(row_sims[idx]),
                    })
            results.append(matches)
        return results

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        if "embedding" in d:
            del d["embedding"]  # Don't serialize raw bytes
        if "metadata" in d and isinstance(d["metadata"], str):
            d["metadata"] = json.loads(d["metadata"])
        return d
