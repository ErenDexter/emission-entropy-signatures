"""Command line interface.

    python -m pes build-template ...          (one-time setup per note design)
    python -m pes enroll   <images|folders>   (storing end - the mint)
    python -m pes verify   <images|folders>   (user end)
    python -m pes list | info | delete <id>
    python -m pes evaluate --gallery DIR --queries DIR
    python -m pes train --gallery DIR --queries DIR --pairs CSV [--save MODEL]
    python -m pes --weights MODEL ...         (use a trained embedding instead of frozen ImageNet)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config


def _expand(paths: list[str]) -> list[Path]:
    from .imaging import list_images

    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += list_images(p)
        elif p.exists():
            out.append(p)
        else:
            sys.exit(f"not found: {p}")
    return out


def cmd_build_template(cfg: Config, a: argparse.Namespace) -> None:
    import cv2
    import numpy as np

    from .imaging import gray_clahe, load_image
    from .registration import Registrar, RegistrationError
    from .template import NoteTemplate, print_free_mask

    t = NoteTemplate.from_reference(load_image(a.reference), crop_box=tuple(a.crop), roi_center=tuple(a.roi[:2]),
                                    roi_axes=tuple(a.roi[2:]), roi_angle=a.roi_angle, rotate=a.rotate,
                                    scale=a.scale, name=a.name)
    samples = _expand(a.mask_from) if a.mask_from else []
    if samples:
        step = max(1, len(samples) // a.mask_samples)
        reg = Registrar(t, cfg)
        grays = []
        for p in samples[::step]:
            try:
                grays.append(gray_clahe(reg.register(load_image(p)).crop))
            except RegistrationError as e:
                print(f"  skip {p.name}: {e}")
        if len(grays) < 10:
            sys.exit("need at least 10 registrable sample notes to find the printed design")
        t.canonical_mask = print_free_mask(grays, cfg.roi_margin)
        t.mask_roi_margin = cfg.roi_margin
        print(f"print-free mask from {len(grays)} notes: {np.mean(t.canonical_mask > 0):.0%} of canonical frame usable")
    t.save(cfg.template_dir)
    print(f"template '{t.name}' saved to {cfg.template_dir}")
    if a.preview:
        vis = cv2.cvtColor(t.image, cv2.COLOR_RGB2BGR)
        c = tuple(round(v) for v in t.roi_center)
        cv2.ellipse(vis, c, tuple(round(v) for v in t.roi_axes), t.roi_angle, 0, 360, (0, 255, 255), 3)
        cv2.imwrite(a.preview, vis)
        print(f"preview: {a.preview}")


def cmd_enroll(cfg: Config, a: argparse.Namespace) -> None:
    from .pipeline import EnrollmentError, Enroller

    paths = _expand(a.paths)
    enroller = Enroller(cfg)
    try:
        jobs = [(a.note_id, paths)] if a.note_id else [(p.stem, [p]) for p in paths]
        ok = 0
        for note_id, imgs in jobs:
            try:
                r = enroller.enroll(note_id, imgs, append=a.append, allow_duplicate=a.allow_duplicate)
                near = f"nearest other note {r.nearest_other[0]} ({r.nearest_other[1]} inliers)" if r.nearest_other else "first note"
                print(f"ENROLLED  {note_id:<20} captures={len(r.capture_ids)} keypoints={r.keypoints}  {near}")
                ok += 1
            except EnrollmentError as e:
                print(f"REJECTED  {note_id:<20} {e}")
        print(f"\n{ok}/{len(jobs)} notes enrolled; database now holds {enroller.index.count()} captures")
    finally:
        enroller.close()


def cmd_verify(cfg: Config, a: argparse.Namespace) -> None:
    import cv2

    from .pipeline import Verifier
    from .viz import render_verdict

    paths = _expand(a.paths)
    verifier = Verifier(cfg)
    if a.vis:
        Path(a.vis).mkdir(parents=True, exist_ok=True)
    try:
        for p in paths:
            v = verifier.verify(p, claimed_note_id=a.claim)
            if a.json:
                print(json.dumps({"image": str(p), **v.to_dict()}))
            else:
                print(f"{p.name:<22} {v.status.value:<10} note={v.note_id or '-':<14} "
                      f"inliers={v.inliers:<4} cos={v.cosine:.3f}  {v.seconds:.2f}s  {v.message}")
                for w in v.warnings:
                    print(f"{'':22} warning: {w}")
            if a.vis:
                enrolled = None
                if v.evidence is not None:
                    snap = verifier.snapshot_path(v.evidence.capture_id)
                    enrolled = cv2.imread(str(snap), cv2.IMREAD_GRAYSCALE) if snap.exists() else None
                render_verdict(v, enrolled, Path(a.vis) / f"{p.stem}_{v.status.value.lower()}.jpg")
    finally:
        verifier.close()


def cmd_list(cfg: Config, a: argparse.Namespace) -> None:
    from .store import RecordStore

    rs = RecordStore(cfg)
    notes = rs.notes()
    for note_id, n, created in notes:
        print(f"{note_id:<24} captures={n}  enrolled={created}")
    print(f"{len(notes)} notes")


def cmd_delete(cfg: Config, a: argparse.Namespace) -> None:
    from .pipeline import Enroller

    e = Enroller(cfg)
    try:
        print(f"deleted {e.delete(a.note_id)} captures of {a.note_id}")
    finally:
        e.close()


def cmd_info(cfg: Config, a: argparse.Namespace) -> None:
    from .ledger import Ledger
    from .pipeline import Engine
    from .store import RecordStore

    engine = Engine(cfg)
    rs = RecordStore(cfg)
    info = {
        "template": engine.template.name,
        "feature_version": engine.feature_version,
        "embedding_model": str(cfg.embed_weights) if cfg.embed_weights else f"frozen ImageNet {cfg.backbone}",
        "embedding_dim": engine.extractor.cnn.dim,
        "device": str(engine.extractor.cnn.device),
        "notes": len(rs.notes()),
        "db_dir": str(cfg.db_dir),
        "vector_db": cfg.qdrant_url or "embedded qdrant (" + str(Path(cfg.db_dir) / "qdrant") + ")",
    }
    try:
        led = Ledger(cfg.keys_dir)
        info.update(mint_key_id=led.public_key_id(), can_enroll=led.can_sign)
    except FileNotFoundError:
        info.update(mint_key_id=None, can_enroll=False)
    print(json.dumps(info, indent=2))


def cmd_evaluate(cfg: Config, a: argparse.Namespace) -> None:
    from datetime import datetime

    from .evaluate import run_evaluation

    out = Path(a.out) if a.out else Path("data") / "eval" / datetime.now().strftime("%Y%m%d-%H%M%S")
    s = run_evaluation(cfg, _expand(a.gallery), _expand(a.queries), out, holdout=a.holdout, seed=a.seed,
                       pairs_csv=Path(a.pairs) if a.pairs else None, all_pairs=a.all_pairs)
    print(json.dumps(s, indent=2))
    print(f"\nreport written to {out}")


def cmd_train(cfg: Config, a: argparse.Namespace) -> None:
    from datetime import datetime

    from .train import TrainConfig, run_training

    given = {"proj_dim": a.proj_dim, "iters": a.iters, "batch": a.batch, "seed": a.seed}
    tc = TrainConfig(**{k: v for k, v in given.items() if v is not None})
    out = Path(a.out) if a.out else Path("data") / "train" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_training(cfg, tc, _expand(a.gallery), _expand(a.queries), Path(a.pairs) if a.pairs else None, out,
                 cv=a.cv, save=Path(a.save) if a.save else None, cache_dir=Path(a.cache) if a.cache else None)
    print(f"\nreport written to {out / 'train_report.json'}")
    if a.save:
        print(f"use the model with:  python -m pes --weights {a.save} --db-dir <new folder> enroll ...")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="pes", description="Photoluminescent Entropy Signatures (PES): enrollment & verification for banknotes")
    p.add_argument("--config", help="JSON file overriding defaults in pes/config.py")
    p.add_argument("--db-dir", help="database folder (default: data/db)")
    p.add_argument("--weights", help="trained embedding from `pes train` (default: frozen ImageNet CNN); "
                                     "a database only holds records of one embedding model")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-template", help="setup: build the geometric anchor for a note design")
    b.add_argument("--reference", required=True, help="photo of a note used as the anchor")
    b.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270], help="make the reference upright")
    b.add_argument("--scale", type=float, default=0.5, help="resize the reference by this factor")
    b.add_argument("--crop", type=int, nargs=4, required=True, metavar=("X0", "Y0", "X1", "Y1"),
                   help="template area around the spray window (rotated+scaled reference pixels)")
    b.add_argument("--roi", type=float, nargs=4, required=True, metavar=("CX", "CY", "AX", "AY"),
                   help="spray window ellipse: centre and semi-axes (rotated+scaled reference pixels)")
    b.add_argument("--roi-angle", type=float, default=0.0)
    b.add_argument("--name", default="note")
    b.add_argument("--mask-from", nargs="*", help="photos/folders of different sprayed notes, used to "
                                                  "detect and exclude printed design inside the window")
    b.add_argument("--mask-samples", type=int, default=60)
    b.add_argument("--preview", help="write a preview image of the template and ROI")

    e = sub.add_parser("enroll", help="STORING END: register sprayed notes in the database")
    e.add_argument("paths", nargs="+", help="images and/or folders")
    e.add_argument("--note-id", help="treat all images as captures of this one note "
                                     "(default: one note per image, ID = file name)")
    e.add_argument("--append", action="store_true", help="add captures to an already enrolled note")
    e.add_argument("--allow-duplicate", action="store_true", help="skip the 'pattern already enrolled' check")

    v = sub.add_parser("verify", help="USER END: check photo(s) of a note taken from any angle")
    v.add_argument("paths", nargs="+", help="images and/or folders")
    v.add_argument("--claim", help="note ID read from the note (serial/QR) - enables ID-swap detection")
    v.add_argument("--vis", help="folder for visual evidence images")
    v.add_argument("--json", action="store_true", help="machine-readable output")

    sub.add_parser("list", help="list enrolled notes")
    d = sub.add_parser("delete", help="remove an enrolled note")
    d.add_argument("note_id")
    sub.add_parser("info", help="show configuration, model and database status")

    ev = sub.add_parser("evaluate", help="benchmark on a gallery/query split in a fresh database")
    ev.add_argument("--gallery", nargs="+", required=True, help="enrollment images/folders (one note each)")
    ev.add_argument("--queries", nargs="+", required=True, help="verification images/folders")
    ev.add_argument("--out", help="report folder (default: data/eval/<timestamp>)")
    ev.add_argument("--holdout", type=float, default=0.5, help="fraction of notes removed for the unenrolled test")
    ev.add_argument("--pairs", help="optional ground truth CSV: query_file,gallery_file")
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--all-pairs", action="store_true",
                    help="also match every query against every enrolled note (writes all_pairs_inliers.csv)")

    t = sub.add_parser("train", help="fine-tune the CNN embedding (metric learning, note-level cross-validation)")
    t.add_argument("--gallery", nargs="+", required=True, help="one photo per note (ID = file name)")
    t.add_argument("--queries", nargs="*", default=[], help="further photos of the same notes, matched by --pairs")
    t.add_argument("--pairs", help="results.csv written by `evaluate`, or a query_file,gallery_file CSV")
    t.add_argument("--cv", type=int, default=5, help="cross-validation folds over notes (0 = skip)")
    t.add_argument("--save", help="then train on all notes and save the model, e.g. data/models/pes_embed.pt")
    t.add_argument("--out", help="report folder (default: data/train/<timestamp>)")
    t.add_argument("--cache", help="registered spray maps cache (default: data/train_cache)")
    t.add_argument("--proj-dim", type=int, help="1x1-conv projection channels (default 128)")
    t.add_argument("--iters", type=int, help="training iterations (default 300)")
    t.add_argument("--batch", type=int, help="notes per batch (default 24)")
    t.add_argument("--seed", type=int)

    a = p.parse_args(argv)
    cfg = Config.load(a.config)
    if a.db_dir:
        cfg.db_dir = Path(a.db_dir)
    if a.weights:
        cfg.embed_weights = Path(a.weights)
    {"build-template": cmd_build_template, "enroll": cmd_enroll, "verify": cmd_verify, "list": cmd_list,
     "delete": cmd_delete, "info": cmd_info, "evaluate": cmd_evaluate, "train": cmd_train}[a.cmd](cfg, a)
