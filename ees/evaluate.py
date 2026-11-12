"""Offline evaluation: enroll a gallery, verify queries taken from another angle.

Protocols (all in a fresh, separate database so the real one is never touched):

A. Identification  - every query is verified without a claimed ID (1:N search).
B. Claimed ID      - each identified query is re-checked claiming its own note
                     (must be GENUINE) and claiming a random *other* note, which simulates a
                     copied serial/QR on a different sprayed note (must be MISMATCH).
C. Unenrolled      - a random fraction of enrolled notes is deleted; queries of those
                     notes now stand in for counterfeits/unregistered notes (must be
                     NOT_FOUND). Queries of the remaining notes must still be GENUINE.

Without a ground-truth pairs file, protocol A's matches are reported as found; with
--pairs (CSV: query,gallery file names) identification accuracy is scored.

With all_pairs=True, every query is also matched droplet by droplet against every
enrolled note (not only its search shortlist) and the scores go to all_pairs_inliers.csv.
"""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np

from .config import Config
from .pipeline import Capture, EnrollmentError, Enroller, Engine, Status, Verifier
from .registration import RegistrationError


def _stem(p: str | Path) -> str:
    return Path(p).stem


def run_evaluation(cfg: Config, gallery: list[Path], queries: list[Path], out_dir: Path,
                   holdout: float = 0.5, seed: int = 0, pairs_csv: Path | None = None,
                   all_pairs: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.db_dir = out_dir / "db"
    if cfg.db_dir.exists() and any(cfg.db_dir.iterdir()):
        raise FileExistsError(f"{cfg.db_dir} is not empty - choose a new --out folder")
    cfg.keys_dir = out_dir / "keys"
    rng = random.Random(seed)
    engine = Engine(cfg)
    enroller = Enroller(cfg, engine)
    verifier = Verifier(cfg, engine, stores=enroller.stores)
    log = lambda msg: print(msg, flush=True)

    # ---- enrollment ------------------------------------------------------------------
    t0 = time.perf_counter()
    enrolled, enroll_fail, enroll_secs = [], {}, []
    for i, p in enumerate(gallery, 1):
        t = time.perf_counter()
        try:
            res = enroller.enroll(_stem(p), [p])
            enrolled.append(res.note_id)
            enroll_secs.append(time.perf_counter() - t)
        except EnrollmentError as e:
            enroll_fail[_stem(p)] = str(e)
        if i % 20 == 0 or i == len(gallery):
            log(f"  enrolled {len(enrolled)}/{i} ({time.perf_counter() - t0:.0f}s)")

    # ---- A: identification -------------------------------------------------------------
    caps: dict[str, Capture] = {}
    rows, genuine_scores, impostor_scores, verify_secs = [], [], [], []
    unreadable = {}
    for i, p in enumerate(queries, 1):
        t = time.perf_counter()
        try:
            cap = engine.process(p)
        except RegistrationError as e:
            unreadable[_stem(p)] = str(e)
            rows.append({"query": _stem(p), "status": Status.UNREADABLE.value, "note_id": "", "inliers": 0,
                         "cosine": 0.0, "second_inliers": 0, "message": str(e)})
            continue
        v = verifier.verify_capture(cap, t0=t)
        verify_secs.append(v.seconds)
        caps[_stem(p)] = cap
        others = [c.inliers for c in v.candidates if c.note_id != v.note_id]
        if v.status == Status.GENUINE:
            genuine_scores.append(v.inliers)
        impostor_scores.extend(others)
        rows.append({"query": _stem(p), "status": v.status.value, "note_id": v.note_id or "",
                     "inliers": v.inliers, "cosine": round(v.cosine, 4),
                     "second_inliers": max(others, default=0), "message": v.message})
        if i % 20 == 0 or i == len(queries):
            log(f"  verified {i}/{len(queries)}")
    found = {r["query"]: r["note_id"] for r in rows if r["status"] == Status.GENUINE.value}

    accuracy, truth = None, None
    if pairs_csv:
        truth = {_stem(a): _stem(b) for a, b in csv.reader(open(pairs_csv)) if a and not a.startswith("#")}
        scored = [q for q in truth if q in {r["query"] for r in rows}]
        correct = sum(found.get(q) == truth[q] for q in scored)
        accuracy = {"scored": len(scored), "correct": correct, "accuracy": correct / max(len(scored), 1)}

    # ---- every query against every enrolled note (optional) ------------------------------
    pairs_summary = None
    if all_pairs:
        same_note = truth or found
        records = {n: enroller.records.by_note(n) for n in enrolled}
        same, different = [], []
        with open(out_dir / "all_pairs_inliers.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["test_photo", "enrolled_note", "inliers", "same_note"])
            for i, (q, cap) in enumerate(caps.items(), 1):
                for note, recs in records.items():
                    n = max((engine.matcher.match(cap.fingerprint.local, r.local).inliers for r in recs), default=0)
                    is_same = same_note.get(q) == note
                    (same if is_same else different).append(n)
                    w.writerow([q, note, n, int(is_same)])
                if i % 20 == 0 or i == len(caps):
                    log(f"  all pairs: {i}/{len(caps)} queries")
        pairs_summary = {"comparisons": len(same) + len(different),
                         "same_note_pairs": len(same), "same_note_inliers_min": min(same, default=None),
                         "different_note_pairs": len(different),
                         "different_note_inliers_max": max(different, default=None)}

    # ---- B: claimed ID ------------------------------------------------------------------
    claim_ok = claim_swap_caught = claim_swap_false_accept = 0
    for q, note in found.items():
        if verifier.verify_capture(caps[q], claimed_note_id=note).status == Status.GENUINE:
            claim_ok += 1
        wrong = rng.choice([n for n in enrolled if n != note])
        s = verifier.verify_capture(caps[q], claimed_note_id=wrong).status
        claim_swap_caught += s == Status.MISMATCH
        claim_swap_false_accept += s == Status.GENUINE

    # ---- C: unenrolled notes ------------------------------------------------------------
    removed = set(rng.sample(sorted(set(found.values())), int(len(set(found.values())) * holdout)))
    for note in removed:
        enroller.delete(note)
    far_trials = frr_trials = false_accepts = false_rejects = 0
    for q, note in found.items():
        v = verifier.verify_capture(caps[q])
        if note in removed:
            far_trials += 1
            false_accepts += v.status == Status.GENUINE
        else:
            frr_trials += 1
            false_rejects += not (v.status == Status.GENUINE and v.note_id == note)

    gen, imp = np.array(genuine_scores), np.array(impostor_scores or [0])
    summary = {
        "gallery_images": len(gallery), "enrolled": len(enrolled), "enroll_failures": enroll_fail,
        "queries": len(queries), "unreadable": unreadable,
        "A_identification": {
            "genuine": sum(r["status"] == "GENUINE" for r in rows),
            "not_found": sum(r["status"] == "NOT_FOUND" for r in rows),
            "distinct_notes_matched": len(set(found.values())),
            "ground_truth": accuracy,
        },
        "scores": {
            "accept_inliers": cfg.accept_inliers,
            "genuine_inliers_min": int(gen.min()) if len(gen) else None,
            "genuine_inliers_median": float(np.median(gen)) if len(gen) else None,
            "impostor_inliers_max": int(imp.max()), "impostor_inliers_p99": float(np.percentile(imp, 99)),
            "impostor_comparisons": len(impostor_scores),
        },
        "all_pairs": pairs_summary,
        "B_claimed_id": {"claim_own_note_accepted": f"{claim_ok}/{len(found)}",
                         "claim_other_note_flagged_mismatch": f"{claim_swap_caught}/{len(found)}",
                         "claim_other_note_false_accept": claim_swap_false_accept},
        "C_unenrolled": {"removed_notes": len(removed),
                         "false_accepts": f"{false_accepts}/{far_trials}",
                         "false_rejects": f"{false_rejects}/{frr_trials}"},
        "timing_s": {"enroll_mean": float(np.mean(enroll_secs)) if enroll_secs else None,
                     "verify_mean": float(np.mean(verify_secs)) if verify_secs else None},
        "feature_version": engine.feature_version,
    }
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _plot(gen, np.array(impostor_scores), cfg.accept_inliers, out_dir / "score_distribution.png")
    enroller.close()
    return summary


def _plot(gen: np.ndarray, imp: np.ndarray, thr: int, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    top = max(gen.max() if len(gen) else 1, imp.max() if len(imp) else 1)
    bins = np.arange(0, top + 10, 5)
    ax.hist(imp, bins=bins, color="#c0392b", alpha=0.75, label=f"different notes (n={len(imp)})")
    ax.hist(gen, bins=bins, color="#2e86c1", alpha=0.75, label=f"same note, other angle (n={len(gen)})")
    ax.axvline(thr, color="black", ls="--", lw=1, label=f"accept threshold = {thr}")
    ax.set_yscale("log")
    ax.set_xlabel("geometrically consistent droplet matches (inliers)")
    ax.set_ylabel("count (log)")
    ax.legend()
    ax.set_title("EES verification scores")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
