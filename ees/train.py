"""Train the CNN embedding - the vector-database key - with metric learning.

Why: verification re-ranks only the top-k notes of the embedding search, so as the
database grows the embedding decides whether a genuine note is found at all
(ees/scaling.py). Frozen ImageNet features see every spray as much the same texture:
on the 2-Taka set the shortlist needed to keep 99% recall grows to a sizeable fraction
of the database.

What is trained: the same EmbeddingNet both ends run (ees/fingerprint.py). The stem and
the stages before `train_from` stay frozen (generic edges and blobs), the later stages
are fine-tuned, and a 1x1-conv projection compresses the channels. No FC layer.

Training signal: InfoNCE over a batch of notes, two views per note. A view pair is
either two real captures of the note (other angle, lamp, exposure) or two augmentations
of one capture - residual misregistration, gamma/gain, blur, droplet bloom, sensor noise,
occlusion - so notes with a single photo still contribute. Every other note in the batch
is a negative.

Evaluation: note-level K-fold. Held-out notes are never seen in training; their extra
captures are queries against the first capture of every note. The trained model and the
frozen baseline are scored by the identical protocol and projected to larger databases
with ees/scaling.py.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from . import scaling
from .config import ROOT, Config
from .fingerprint import CNNEmbedder, EmbeddingNet, resolve_device, to_input
from .imaging import fluorescence_map, load_image
from .registration import Registrar, RegistrationError
from .template import NoteTemplate


@dataclass
class TrainConfig:
    # Defaults measured on the 2-Taka set (5-fold, unseen notes): top-1 96.4% -> 100%,
    # d' 3.1 -> 5.8, shortlist for 99% recall at 10k notes 1,439 -> 16, 10240-d -> 1280-d.
    proj_dim: int = 128               # 1x1-conv output channels (0 = keep the backbone's)
    train_from: str = "layer2"        # first fine-tuned stage; everything before stays frozen
    iters: int = 300
    batch: int = 24                   # notes per batch, two views each
    lr: float = 1e-4                  # fine-tuned stages; the new projection learns 10x faster
    weight_decay: float = 1e-4
    tau: float = 0.05                 # InfoNCE temperature
    pair_p: float = 0.75              # P(positive = two real captures) for notes with several
    pretrained: bool = True           # start from ImageNet weights
    seed: int = 0
    # augmentation: residual misregistration, lamp/exposure, focus, sensor noise, occlusion
    aug_rotate_deg: float = 4.0
    aug_scale: float = 0.04
    aug_shift: float = 0.025          # fraction of the input size
    aug_gamma: float = 1.6
    aug_gain: float = 0.3
    aug_blur_sigma: float = 1.8
    aug_bloom_p: float = 0.3
    aug_noise: float = 0.05
    aug_occlusions: int = 2


@dataclass
class Capture:
    note_id: str
    source: str
    fmap: np.ndarray   # canonical uint8 spray map
    mask: np.ndarray   # canonical uint8 window mask


# ─────────────────────────────────── data ───────────────────────────────────

def read_pairs(path: str | Path) -> dict[str, str]:
    """query stem -> gallery stem, from `evaluate`'s results.csv (GENUINE rows) or a plain
    query_file,gallery_file CSV (the `evaluate --pairs` format)."""
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    if rows and {"query", "note_id"} <= set(rows[0]):
        head = rows[0]
        q, n = head.index("query"), head.index("note_id")
        s = head.index("status") if "status" in head else None
        return {Path(r[q]).stem: Path(r[n]).stem for r in rows[1:]
                if r[n] and (s is None or r[s] == "GENUINE")}
    return {Path(r[0]).stem: Path(r[1]).stem for r in rows if len(r) >= 2}


def collect_notes(gallery: list[Path], queries: list[Path], pairs: dict[str, str]) -> dict[str, list[Path]]:
    """note_id -> photos, enrollment photo first. One note per gallery photo, ID = file stem."""
    notes = {p.stem: [p] for p in gallery}
    for q in queries:
        if pairs.get(q.stem) in notes:
            notes[pairs[q.stem]].append(q)
    return notes


class CaptureCache:
    """Registered spray maps on disk, so retraining does not re-register every photo."""

    def __init__(self, cfg: Config, directory: str | Path):
        self.cfg = cfg
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._registrar: Registrar | None = None
        h = hashlib.sha256()
        for f in sorted(Path(cfg.template_dir).iterdir()):
            h.update(f.read_bytes())
        keys = ["reg_max_side", "reg_sift_features", "reg_ratio", "reg_ransac_thresh", "reg_min_inliers",
                "reg_min_roi_coverage", "canonical_size", "roi_margin"]
        h.update(json.dumps({k: getattr(cfg, k) for k in keys}, sort_keys=True).encode())
        self.version = h.hexdigest()[:16]

    def get(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """(spray map, mask) in the canonical frame; raises RegistrationError."""
        st = path.stat()
        key = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{self.version}".encode()).hexdigest()
        f = self.dir / f"{key[:20]}.npz"
        if f.exists():
            z = np.load(f)
            return z["fmap"], z["mask"]
        if self._registrar is None:
            self._registrar = Registrar(NoteTemplate.load(self.cfg.template_dir), self.cfg)
        reg = self._registrar.register(load_image(path))
        fmap = fluorescence_map(reg.crop, reg.mask)
        np.savez_compressed(f, fmap=fmap, mask=reg.mask)
        return fmap, reg.mask


# ───────────────────────────────── training ─────────────────────────────────

def augment(x: torch.Tensor, m: torch.Tensor, tc: TrainConfig,
            gen: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture-to-capture variation, applied identically to B,1,S,S spray maps and masks."""
    B, dev, S = x.shape[0], x.device, x.shape[-1]

    def u(*shape):  # uniform in [-1, 1]
        return torch.rand(*shape, device=dev) * 2 - 1

    ang, sc = u(B) * math.radians(tc.aug_rotate_deg), 1 + u(B) * tc.aug_scale
    c, s = torch.cos(ang) / sc, torch.sin(ang) / sc
    t = u(2, B) * 2 * tc.aug_shift                     # affine_grid works in [-1, 1] coordinates
    theta = torch.stack([torch.stack([c, -s, t[0]], 1), torch.stack([s, c, t[1]], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x, m = F.grid_sample(x, grid, align_corners=False), F.grid_sample(m, grid, align_corners=False)

    x = x.clamp(0, 1).pow(torch.exp(u(B, 1, 1, 1) * math.log(tc.aug_gamma))) * (1 + u(B, 1, 1, 1) * tc.aug_gain)
    x = torch.cat([TF.gaussian_blur(x[i:i + 1], [2 * math.ceil(3 * sg) + 1] * 2, [float(sg)] * 2)
                   if sg > 0.3 else x[i:i + 1] for i, sg in enumerate(gen.uniform(0, tc.aug_blur_sigma, B))])
    bloom = torch.from_numpy(gen.random(B) < tc.aug_bloom_p).to(dev)
    if bloom.any():
        x[bloom] = F.max_pool2d(x[bloom], 3, 1, 1)
    x = x + torch.randn_like(x) * (torch.rand(B, 1, 1, 1, device=dev) * tc.aug_noise)
    for i in range(B):
        for _ in range(gen.integers(0, tc.aug_occlusions + 1)):
            h, w = gen.integers(S // 20, S // 7, 2)
            y0, x0 = gen.integers(0, S - h), gen.integers(0, S - w)
            x[i, :, y0:y0 + h, x0:x0 + w] = 0
    return x.clamp(0, 1), m


def info_nce(e: torch.Tensor, tau: float) -> torch.Tensor:
    """e = [view A of notes 0..n-1, view B of notes 0..n-1], L2-normalized."""
    n = e.shape[0] // 2
    sim = (e @ e.T) / tau
    sim.fill_diagonal_(-1e4)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(e.device)
    return F.cross_entropy(sim.float(), target)


def train_model(cfg: Config, tc: TrainConfig, caps: list[Capture], notes: list[list[int]],
                log=print) -> EmbeddingNet:
    """Fine-tune an EmbeddingNet. notes[i] = indices into caps of one training note."""
    device = resolve_device(cfg.device)
    torch.manual_seed(tc.seed)
    gen = np.random.default_rng(tc.seed)
    s = cfg.cnn_input
    resize = lambda a: cv2.resize(a, (s, s), interpolation=cv2.INTER_AREA)
    imgs = torch.from_numpy(np.stack([resize(c.fmap) for c in caps])).to(device)
    masks = torch.from_numpy(np.stack([resize(c.mask) for c in caps])).to(device)

    net = EmbeddingNet(cfg.backbone, cfg.cnn_layer, cfg.grid, cfg.gem_p, tc.proj_dim, tc.pretrained).to(device)
    n_frozen = 4 + net.stages.index(tc.train_from)    # conv1, bn1, relu, maxpool, then stages
    for p in net.body[:n_frozen].parameters():
        p.requires_grad = False
    groups = [{"params": list(net.body[n_frozen:].parameters()), "lr": tc.lr}]
    if net.proj is not None:
        groups.append({"params": list(net.proj.parameters()), "lr": tc.lr * 10})
    opt = torch.optim.AdamW(groups, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups],
                                                total_steps=tc.iters, pct_start=0.1)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    batch = min(tc.batch, len(notes))

    t0 = time.perf_counter()
    for it in range(tc.iters):
        net.train()
        for mod in net.modules():          # little data: keep the ImageNet BatchNorm statistics
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.eval()
        a, b = [], []
        for n in gen.choice(len(notes), batch, replace=False):
            ids = notes[n]
            if len(ids) > 1 and gen.random() < tc.pair_p:
                i, j = gen.choice(ids, 2, replace=False)
            else:
                i = j = ids[gen.integers(len(ids))]
            a.append(int(i))
            b.append(int(j))
        idx = torch.tensor(a + b, device=device)
        x, m = augment(imgs[idx].unsqueeze(1).float() / 255, masks[idx].unsqueeze(1).float() / 255, tc, gen)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            with torch.no_grad():
                f = net.body[:n_frozen](to_input(x))
            f = net.project(net.body[n_frozen:](f))
            loss = info_nce(net.pool(f, F.interpolate(m, size=f.shape[-2:], mode="area")), tc.tau)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if (it + 1) % 50 == 0 or it + 1 == tc.iters:
            log(f"    iter {it + 1}/{tc.iters}  loss {loss.item():.3f}  ({time.perf_counter() - t0:.0f}s)")
    return net.eval()


def embed_captures(embedder: CNNEmbedder, caps: list[Capture], batch: int = 16) -> np.ndarray:
    """The production embedding path, exactly as enrollment and verification run it."""
    return np.concatenate([embedder.embed_batch([c.fmap for c in caps[i:i + batch]],
                                                [c.mask for c in caps[i:i + batch]])
                           for i in range(0, len(caps), batch)])


# ──────────────────────────────── evaluation ────────────────────────────────

def _fold_scores(emb: np.ndarray, notes: list[list[int]], held: list[int]):
    enroll = np.array([ids[0] for ids in notes])       # gallery: first capture of every note
    ranks, gen, imp = [], [], []
    for col, n in enumerate(held):
        q = emb[notes[n][1:]]                           # the note's other captures
        full = q @ emb[enroll].T                        # against every enrolled note
        ranks += (1 + (full > full[:, [n]]).sum(1)).tolist()
        sub = q @ emb[enroll[held]].T                   # against unseen notes only
        gen += sub[:, col].tolist()
        imp += np.delete(sub, col, axis=1).ravel().tolist()
    return np.array(ranks), np.array(gen), np.array(imp)


def _summarize(folds) -> dict:
    ranks = np.concatenate([r for r, _, _ in folds])
    z = [scaling.standardize(g, i) for _, g, i in folds]     # each fold's own scale, then pooled
    zg, zi = np.concatenate([g for g, _ in z]), np.concatenate([i for _, i in z])
    return {"queries": int(len(ranks)), "impostor_scores": int(len(zi)),
            "top1": float(np.mean(ranks == 1)), "top5": float(np.mean(ranks <= 5)),
            "top20": float(np.mean(ranks <= 20)), "worst_rank": int(ranks.max()),
            **scaling.project(zg, zi)}


def cross_validate(cfg: Config, tc: TrainConfig, caps: list[Capture], notes: list[list[int]],
                   k: int = 5, log=print) -> dict:
    """Note-level K-fold: trained vs frozen embedding on notes the model never saw."""
    evaluable = [n for n, ids in enumerate(notes) if len(ids) > 1]
    if len(evaluable) < 2 * k:
        raise ValueError(f"{k}-fold cross-validation needs >= {2 * k} notes with two or more photos, "
                         f"have {len(evaluable)} (pass --queries and --pairs)")
    folds = np.array_split(np.random.default_rng(tc.seed).permutation(evaluable), k)
    frozen = embed_captures(CNNEmbedder(cfg, net=EmbeddingNet(cfg.backbone, cfg.cnn_layer, cfg.grid, cfg.gem_p)), caps)
    scores: dict[str, list] = {"frozen": [], "trained": []}
    for f, held in enumerate(folds, 1):
        held = sorted(int(n) for n in held)
        log(f"  fold {f}/{k}: training on {len(notes) - len(held)} notes, testing on {len(held)} unseen notes")
        net = train_model(cfg, tc, caps, [ids for n, ids in enumerate(notes) if n not in held], log)
        trained = embed_captures(CNNEmbedder(cfg, net=net), caps)
        scores["frozen"].append(_fold_scores(frozen, notes, held))
        scores["trained"].append(_fold_scores(trained, notes, held))
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out = {name: _summarize(s) for name, s in scores.items()}
    out["dim"] = {"frozen": int(frozen.shape[1]), "trained": int(trained.shape[1])}
    return out


def format_cv(cv: dict) -> str:
    fr, tr = cv["frozen"], cv["trained"]
    lines = [f"  unseen notes, {tr['queries']} queries     frozen ImageNet    trained",
             f"  embedding size                  {cv['dim']['frozen']:>12,}  {cv['dim']['trained']:>9,}"]
    for key, label in (("top1", "top-1 vs all enrolled"), ("top20", "top-20 vs all enrolled")):
        lines.append(f"  {label:<30}  {fr[key]:>12.1%}  {tr[key]:>9.1%}")
    lines.append(f"  {'worst rank of the right note':<30}  {fr['worst_rank']:>12}  {tr['worst_rank']:>9}")
    lines.append(f"  {'separation d-prime':<30}  {fr['d_prime']:>12.2f}  {tr['d_prime']:>9.2f}")
    lines.append(f"  shortlist for {tr['target_recall']:.0%} recall (projected):")
    for n in tr["shortlist_needed"]:
        lines.append(f"  {n:>14,} notes                {fr['shortlist_needed'][n]:>12,}  {tr['shortlist_needed'][n]:>9,}")
    return "\n".join(lines)


# ────────────────────────────────── driver ──────────────────────────────────

def run_training(cfg: Config, tc: TrainConfig, gallery: list[Path], queries: list[Path],
                 pairs_csv: Path | None, out_dir: Path, cv: int = 5, save: Path | None = None,
                 cache_dir: Path | None = None, log=None) -> dict:
    log = log or (lambda msg: print(msg, flush=True))
    notes_photos = collect_notes(gallery, queries, read_pairs(pairs_csv) if pairs_csv else {})
    cache = CaptureCache(cfg, cache_dir or ROOT / "data" / "train_cache")
    caps: list[Capture] = []
    notes: list[list[int]] = []
    unreadable: dict[str, str] = {}
    t0 = time.perf_counter()
    for i, (note_id, photos) in enumerate(notes_photos.items(), 1):
        ids = []
        for p in photos:
            try:
                fmap, mask = cache.get(p)
            except RegistrationError as e:
                unreadable[str(p)] = str(e)
                continue
            ids.append(len(caps))
            caps.append(Capture(note_id, str(p), fmap, mask))
        if ids:
            notes.append(ids)
        if i % 50 == 0 or i == len(notes_photos):
            log(f"  registered {i}/{len(notes_photos)} notes ({time.perf_counter() - t0:.0f}s)")
    several = sum(len(ids) > 1 for ids in notes)
    log(f"{len(notes)} notes, {len(caps)} photos ({several} notes with several photos), "
        f"{len(unreadable)} photos unreadable")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = {"created_at": now, "notes": len(notes), "photos": len(caps), "notes_with_several_photos": several,
              "unreadable": unreadable, "train_config": asdict(tc),
              "feature_settings": {k: getattr(cfg, k) for k in ("backbone", "cnn_layer", "cnn_input", "grid", "gem_p")}}
    if cv:
        log(f"cross-validation over notes, {cv} folds")
        report["cross_validation"] = cross_validate(cfg, tc, caps, notes, cv, log)
        log(format_cv(report["cross_validation"]))
    if save:
        log(f"training the final model on all {len(notes)} notes")
        net = train_model(cfg, tc, caps, notes, log)
        net.save(save, cfg.cnn_input, meta={"created_at": now, "train_config": asdict(tc), "notes": len(notes),
                                            "photos": len(caps), "cross_validation": report.get("cross_validation")})
        report["saved"] = str(save)
        log(f"model saved to {save}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2))
    return report
