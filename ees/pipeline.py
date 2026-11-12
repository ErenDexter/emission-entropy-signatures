"""End-to-end pipeline: the storing end (mint enrollment) and the user end (verification).

    photo ─► register to note template ─► canonical spray crop ─► fluorescence map
                                                                       │
                          ┌────────────────────────────────────────────┴───────┐
                          ▼                                                    ▼
                 CNN embedding (no FC)                              SIFT keypoints
                          │                                                    │
       enroll: ───► Qdrant vector index                     signed record (SQLite)
       verify: ───► top-k shortlist ──► geometric re-ranking against records ──► verdict
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .fingerprint import Fingerprint, FingerprintExtractor
from .imaging import load_image
from .ledger import Ledger, record_digest
from .matching import GeometricMatcher, MatchResult
from .registration import Registrar, Registration, RegistrationError
from .store import Record, RecordStore, VectorIndex
from .template import NoteTemplate


@dataclass
class Capture:
    source: str
    registration: Registration
    fingerprint: Fingerprint
    seconds: float


class Engine:
    """Deterministic processing shared by both ends: photo -> fingerprint."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.template = NoteTemplate.load(cfg.template_dir)
        self.registrar = Registrar(self.template, cfg)
        self.extractor = FingerprintExtractor(cfg)
        self.matcher = GeometricMatcher(cfg)
        self.feature_version = f"{self.template.name}:{cfg.feature_fingerprint()}"

    def process(self, image: str | Path | np.ndarray, source: str | None = None) -> Capture:
        t0 = time.perf_counter()
        if isinstance(image, np.ndarray):
            rgb, source = image, source or "<array>"
        else:
            rgb, source = load_image(image), source or str(image)
        reg = self.registrar.register(rgb)
        fp = self.extractor.extract(reg.crop, reg.mask)
        return Capture(source, reg, fp, time.perf_counter() - t0)


@dataclass
class Candidate:
    note_id: str
    capture_id: str
    cosine: float
    inliers: int
    match: MatchResult
    enrolled_points: np.ndarray   # keypoints of the enrolled capture (match.pairs index into these)
    trusted: bool = True          # signature valid and same feature version


@dataclass
class Stores:
    records: RecordStore
    index: VectorIndex
    ledger: Ledger

    @classmethod
    def open(cls, cfg: Config, dim: int, create_keys: bool = False) -> "Stores":
        return cls(RecordStore(cfg), VectorIndex(cfg, dim), Ledger(cfg.keys_dir, create=create_keys))


class _Service:
    def __init__(self, cfg: Config, engine: Engine | None = None, stores: Stores | None = None,
                 create_keys: bool = False):
        self.cfg = cfg
        self.engine = engine or Engine(cfg)
        # Embedded Qdrant allows one client per folder, so services in one process share stores.
        self.stores = stores or Stores.open(cfg, self.engine.extractor.cnn.dim, create_keys)
        self.records, self.index, self.ledger = self.stores.records, self.stores.index, self.stores.ledger
        self.snapshots = Path(cfg.db_dir) / "snapshots"

    def candidates(self, fp: Fingerprint, k: int, include_notes: tuple[str, ...] = (),
                   exclude_note: str | None = None) -> tuple[list[Candidate], list[str]]:
        """Shortlist by vector search, then geometrically verify every shortlisted capture."""
        warnings: list[str] = []
        hits = self.index.search(fp.embedding, k)
        records = [r for r in (self.records.get(h.capture_id) for h in hits) if r is not None]
        seen = {r.capture_id for r in records}
        for note in include_notes:
            records += [r for r in self.records.by_note(note) if r.capture_id not in seen]
        out = []
        for rec in records:
            if rec.note_id == exclude_note:
                continue
            trusted = True
            if rec.feature_version != self.engine.feature_version:
                warnings.append(f"{rec.note_id}: enrolled with feature version {rec.feature_version}, skipped")
                trusted = False
            elif not self.ledger.verify(record_digest(*rec.digest_fields()), rec.signature):
                warnings.append(f"{rec.note_id}: record signature INVALID - database tampering?")
                trusted = False
            m = self.engine.matcher.match(fp.local, rec.local) if trusted else MatchResult(0, 0)
            cos = float(np.dot(fp.embedding, rec.embedding))
            out.append(Candidate(rec.note_id, rec.capture_id, cos, m.inliers, m, rec.local.points, trusted))
        out.sort(key=lambda c: (c.trusted, c.inliers, c.cosine), reverse=True)
        return out, warnings

    def snapshot_path(self, capture_id: str) -> Path:
        return self.snapshots / f"{capture_id}.png"

    def close(self) -> None:
        self.index.close()
        self.records.close()


# ─────────────────────────────── storing end ────────────────────────────────

class EnrollmentError(RuntimeError):
    pass


@dataclass
class EnrollResult:
    note_id: str
    capture_ids: list[str]
    keypoints: list[int]
    reg_inliers: list[int]
    nearest_other: tuple[str, int] | None   # closest *other* enrolled note and its inliers
    warnings: list[str] = field(default_factory=list)


class Enroller(_Service):
    """Mint side: capture one or more photos of a freshly sprayed note and register it."""

    def __init__(self, cfg: Config, engine: Engine | None = None, stores: Stores | None = None):
        super().__init__(cfg, engine, stores, create_keys=True)
        if not self.ledger.can_sign:
            raise PermissionError("enrollment requires the mint private key")
        self.snapshots.mkdir(parents=True, exist_ok=True)

    def enroll(self, note_id: str, images: list[str | Path | np.ndarray], append: bool = False,
               allow_duplicate: bool = False) -> EnrollResult:
        if not images:
            raise EnrollmentError("no images given")
        if self.records.has_note(note_id) and not append:
            raise EnrollmentError(f"note {note_id!r} already enrolled (use append to add captures)")
        try:
            captures = [self.engine.process(img) for img in images]
        except RegistrationError as e:
            raise EnrollmentError(f"capture rejected: {e}") from e

        cfg = self.cfg
        # Quality gate 1: multiple captures of the same note must agree with each other.
        for c in captures[1:]:
            m = self.engine.matcher.match(captures[0].fingerprint.local, c.fingerprint.local)
            if m.inliers < cfg.accept_inliers:
                raise EnrollmentError(f"captures disagree ({c.source}: {m.inliers} inliers) - retake")

        # Quality gate 2: the pattern must not already belong to another note
        # (re-submitted note, or an accidental collision of two very similar sprays).
        nearest = None
        for c in captures:
            cands, _ = self.candidates(c.fingerprint, cfg.shortlist_k, exclude_note=note_id)
            if cands and (nearest is None or cands[0].inliers > nearest[1]):
                nearest = (cands[0].note_id, cands[0].inliers)
        if nearest and nearest[1] >= cfg.duplicate_inliers and not allow_duplicate:
            raise EnrollmentError(f"pattern already enrolled as note {nearest[0]!r} ({nearest[1]} inliers)")

        ids = []
        for c in captures:
            rec = Record(
                capture_id=RecordStore.new_capture_id(), note_id=note_id, created_at=RecordStore.now(),
                source=Path(c.source).name, feature_version=self.engine.feature_version,
                embedding=c.fingerprint.embedding, local=c.fingerprint.local,
                reg_inliers=c.registration.inliers, signature=b"",
            )
            rec.signature = self.ledger.sign(record_digest(*rec.digest_fields()))
            self.records.put(rec)
            self.index.add(rec.capture_id, note_id, rec.embedding,
                           {"created_at": rec.created_at, "source": rec.source})
            cv2.imwrite(str(self.snapshot_path(rec.capture_id)), c.fingerprint.fmap)
            ids.append(rec.capture_id)
        return EnrollResult(note_id, ids, [len(c.fingerprint.local) for c in captures],
                            [c.registration.inliers for c in captures], nearest)

    def delete(self, note_id: str) -> int:
        self.index.delete_note(note_id)
        return self.records.delete_note(note_id)


# ──────────────────────────────── user end ──────────────────────────────────

class Status(str, Enum):
    GENUINE = "GENUINE"          # pattern matches an enrolled note (and the claimed ID, if any)
    MISMATCH = "MISMATCH"        # pattern is genuine but belongs to a different note than claimed
    NOT_FOUND = "NOT_FOUND"      # pattern matches no enrolled note: counterfeit or unregistered
    UNREADABLE = "UNREADABLE"    # note / spray window could not be located in the photo


@dataclass
class Verdict:
    status: Status
    note_id: str | None
    inliers: int
    cosine: float
    message: str
    evidence: Candidate | None = None             # the enrolled capture the decision rests on
    candidates: list[Candidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    capture: Capture | None = None
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "status": self.status.value, "note_id": self.note_id, "inliers": self.inliers,
            "cosine": round(self.cosine, 4), "message": self.message, "warnings": self.warnings,
            "seconds": round(self.seconds, 3),
            "registration_inliers": self.capture.registration.inliers if self.capture else None,
            "candidates": [{"note_id": c.note_id, "inliers": c.inliers, "cosine": round(c.cosine, 4),
                            "trusted": c.trusted} for c in self.candidates[:5]],
        }


class Verifier(_Service):
    """User side: photograph a note from any angle and ask whether it is genuine."""

    def verify(self, image: str | Path | np.ndarray, claimed_note_id: str | None = None) -> Verdict:
        t0 = time.perf_counter()
        try:
            cap = self.engine.process(image)
        except RegistrationError as e:
            return Verdict(Status.UNREADABLE, None, 0, 0.0, str(e), seconds=time.perf_counter() - t0)
        return self.verify_capture(cap, claimed_note_id, t0)

    def verify_capture(self, cap: Capture, claimed_note_id: str | None = None,
                       t0: float | None = None) -> Verdict:
        t0 = time.perf_counter() if t0 is None else t0
        include = (claimed_note_id,) if claimed_note_id else ()
        cands, warnings = self.candidates(cap.fingerprint, self.cfg.shortlist_k, include_notes=include)
        # Best capture per note.
        per_note: dict[str, Candidate] = {}
        for c in cands:
            if c.trusted and c.note_id not in per_note:
                per_note[c.note_id] = c
        ranked = sorted(per_note.values(), key=lambda c: (c.inliers, c.cosine), reverse=True)
        thr = self.cfg.accept_inliers
        best = ranked[0] if ranked else None

        def verdict(status, c, msg):
            # NOT_FOUND still reports the closest candidate's scores, but never names a note.
            note = c.note_id if c and status != Status.NOT_FOUND else None
            return Verdict(status, note, c.inliers if c else 0, c.cosine if c else 0.0,
                           msg, c, cands, warnings, cap, time.perf_counter() - t0)

        if best is None:
            return verdict(Status.NOT_FOUND, None, "database is empty")
        if len(ranked) > 1 and ranked[1].inliers >= thr:
            warnings.append(f"second note {ranked[1].note_id} also matches ({ranked[1].inliers} inliers)")

        if claimed_note_id:
            claimed = per_note.get(claimed_note_id)
            if claimed and claimed.inliers >= thr:
                return verdict(Status.GENUINE, claimed, f"pattern matches note {claimed_note_id}")
            if best.inliers >= thr:
                return verdict(Status.MISMATCH, best,
                               f"pattern belongs to note {best.note_id}, not {claimed_note_id} (ID tampered?)")
            return verdict(Status.NOT_FOUND, claimed or best, "pattern matches no enrolled note")

        if best.inliers >= thr:
            return verdict(Status.GENUINE, best, f"pattern matches note {best.note_id}")
        return verdict(Status.NOT_FOUND, best, "pattern matches no enrolled note")
