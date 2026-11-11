"""Persistence: a vector database for search plus a record store for verification.

* ``VectorIndex`` - Qdrant collection holding one global CNN embedding per enrolled
  capture (cosine distance). Embedded on-disk mode by default; point ``qdrant_url``
  at a Qdrant server for production (HNSW index, concurrent clients).
* ``RecordStore`` - SQLite table with the full signed record of each capture: the
  embedding, local keypoints/descriptors for geometric verification, and the Ed25519
  signature. The vector DB is only an index; the record store is the source of truth.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import Config
from .fingerprint import LocalFeatures


@dataclass
class Hit:
    capture_id: str
    note_id: str
    score: float


class VectorIndex:
    def __init__(self, cfg: Config, dim: int):
        from qdrant_client import QdrantClient, models

        self._models = models
        self.collection = cfg.collection
        if cfg.qdrant_url:
            self.client = QdrantClient(url=cfg.qdrant_url)
        else:
            path = Path(cfg.db_dir) / "qdrant"
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
            if cfg.qdrant_url:  # payload indexes only exist in server mode
                self.client.create_payload_index(self.collection, "note_id", models.PayloadSchemaType.KEYWORD)
        else:
            have = self.client.get_collection(self.collection).config.params.vectors.size
            if have != dim:
                raise ValueError(f"collection {self.collection!r} holds {have}-d vectors, model produces {dim}-d; "
                                 "use a new collection or re-enroll")

    def add(self, capture_id: str, note_id: str, vector: np.ndarray, payload: dict | None = None) -> None:
        m = self._models
        point = m.PointStruct(id=capture_id, vector=vector.astype(np.float32).tolist(),
                              payload={"note_id": note_id, **(payload or {})})
        self.client.upsert(self.collection, points=[point])

    def search(self, vector: np.ndarray, k: int) -> list[Hit]:
        res = self.client.query_points(self.collection, query=vector.astype(np.float32).tolist(),
                                       limit=k, with_payload=True)
        return [Hit(str(p.id), p.payload["note_id"], float(p.score)) for p in res.points]

    def delete_note(self, note_id: str) -> None:
        m = self._models
        flt = m.Filter(must=[m.FieldCondition(key="note_id", match=m.MatchValue(value=note_id))])
        self.client.delete(self.collection, points_selector=m.FilterSelector(filter=flt))

    def count(self) -> int:
        return self.client.count(self.collection).count

    def close(self) -> None:
        self.client.close()


@dataclass
class Record:
    capture_id: str
    note_id: str
    created_at: str
    source: str
    feature_version: str
    embedding: np.ndarray
    local: LocalFeatures
    reg_inliers: int
    signature: bytes

    def digest_fields(self) -> tuple:
        pts, desc = self.local.to_bytes()
        return (self.capture_id, self.note_id, self.created_at, self.feature_version,
                self.embedding.astype(np.float32).tobytes(), pts, desc)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS captures (
    capture_id      TEXT PRIMARY KEY,
    note_id         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    source          TEXT,
    feature_version TEXT NOT NULL,
    embedding       BLOB NOT NULL,
    kp_points       BLOB NOT NULL,
    kp_desc         BLOB NOT NULL,
    reg_inliers     INTEGER,
    signature       BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_captures_note ON captures(note_id);
"""


class RecordStore:
    def __init__(self, cfg: Config):
        Path(cfg.db_dir).mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(Path(cfg.db_dir) / "records.sqlite")
        self.db.executescript(_SCHEMA)

    @staticmethod
    def new_capture_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def put(self, r: Record) -> None:
        pts, desc = r.local.to_bytes()
        with self.db:
            self.db.execute(
                "INSERT INTO captures VALUES (?,?,?,?,?,?,?,?,?,?)",
                (r.capture_id, r.note_id, r.created_at, r.source, r.feature_version,
                 r.embedding.astype(np.float32).tobytes(), pts, desc, r.reg_inliers, r.signature),
            )

    def get(self, capture_id: str) -> Record | None:
        row = self.db.execute("SELECT * FROM captures WHERE capture_id=?", (capture_id,)).fetchone()
        return self._row(row) if row else None

    def by_note(self, note_id: str) -> list[Record]:
        rows = self.db.execute("SELECT * FROM captures WHERE note_id=? ORDER BY created_at", (note_id,))
        return [self._row(r) for r in rows]

    def notes(self) -> list[tuple[str, int, str]]:
        return self.db.execute(
            "SELECT note_id, COUNT(*), MIN(created_at) FROM captures GROUP BY note_id ORDER BY MIN(created_at)"
        ).fetchall()

    def has_note(self, note_id: str) -> bool:
        return self.db.execute("SELECT 1 FROM captures WHERE note_id=? LIMIT 1", (note_id,)).fetchone() is not None

    def delete_note(self, note_id: str) -> int:
        with self.db:
            return self.db.execute("DELETE FROM captures WHERE note_id=?", (note_id,)).rowcount

    @staticmethod
    def _row(row) -> Record:
        (cid, nid, created, source, fv, emb, pts, desc, reg_inl, sig) = row
        return Record(cid, nid, created, source, fv, np.frombuffer(emb, np.float32).copy(),
                      LocalFeatures.from_bytes(pts, desc), reg_inl, sig)

    def close(self) -> None:
        self.db.close()
