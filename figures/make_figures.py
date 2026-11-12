"""Make the README figures from the example photos, the trained model and the published results.

    python figures/make_figures.py --gallery "photos/1st Angle"

--gallery points at the enrolment photos of the other banknotes (part of the full photo set,
available on request). It is used for the search panel and the closest-wrong-banknote panel.
Without it those two panels show the example banknote only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ees.config import Config  # noqa: E402
from ees.imaging import list_images, load_image  # noqa: E402
from ees.pipeline import Engine  # noqa: E402
from ees.train import TrainConfig, augment  # noqa: E402

ENROL, TEST = ROOT / "examples" / "IMG_3705.HEIC", ROOT / "examples" / "IMG_3850.HEIC"
OUT = ROOT / "figures"
W = 16.0  # figure width, inches
DPI = 120

THEMES = {
    "light": dict(surface="#fcfcfb", lane="#f3f2ee", border="#e1e0d9", ink="#0b0b0b", ink2="#52514e",
                  muted="#898781", grid="#e1e0d9", axis="#c3c2b7", blue="#2a78d6", orange="#eb6834",
                  good="#0ca30c", good_text="#006300", bad="#d03b3b", amber="#eda100",
                  ramp=["#cde2fb", "#6da7ec", "#2a78d6", "#104281"]),
    "dark": dict(surface="#1a1a19", lane="#232322", border="#2c2c2a", ink="#f0efec", ink2="#c3c2b7",
                 muted="#898781", grid="#2c2c2a", axis="#383835", blue="#3987e5", orange="#d95926",
                 good="#0ca30c", good_text="#0ca30c", bad="#e66767", amber="#c98500",
                 ramp=["#104281", "#256abf", "#5598e7", "#cde2fb"]),
}
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11})


# ------------------------------------------------------------------------------- data
def process_examples(gallery: list[Path]) -> dict:
    cfg = Config()
    cfg.embed_weights = ROOT / "models" / "ees_embed.pt"
    engine = Engine(cfg)
    enrol, test = engine.process(ENROL), engine.process(TEST)
    d = {"enrol": enrol, "test": test, "match": engine.matcher.match(test.fingerprint.local, enrol.fingerprint.local)}
    ranked = [("IMG_3705", float(np.dot(test.fingerprint.embedding, enrol.fingerprint.embedding)), enrol)]
    size = 1
    for i, p in enumerate(gallery, 1):
        if p.stem == ENROL.stem:
            continue
        try:
            cap = engine.process(p)
        except Exception:  # noqa: BLE001 - a photo that cannot be aligned is simply left out
            continue
        size += 1
        ranked.append((p.stem, float(np.dot(test.fingerprint.embedding, cap.fingerprint.embedding)), cap))
        ranked = sorted(ranked, key=lambda r: r[1], reverse=True)[:3]  # only the top three are drawn
        if i % 20 == 0:
            print(f"  gallery {i}/{len(gallery)}", flush=True)
    d["ranked"], d["gallery_size"] = ranked, size
    wrong = [r for r in ranked if r[0] != ENROL.stem]
    if wrong:
        nid, _, cap = wrong[0]
        d["wrong"] = (nid, cap, engine.matcher.match(test.fingerprint.local, cap.fingerprint.local))

    tc = TrainConfig()
    s = Config().cnn_input
    fm = cv2.resize(enrol.fingerprint.fmap, (s, s), interpolation=cv2.INTER_AREA)
    mk = cv2.resize(enrol.fingerprint.mask, (s, s), interpolation=cv2.INTER_NEAREST)
    x = torch.from_numpy(fm).float().div(255)[None, None].repeat(2, 1, 1, 1)
    m = torch.from_numpy(mk).float().div(255)[None, None].repeat(2, 1, 1, 1)
    torch.manual_seed(3)
    xa, _ = augment(x, m, tc, np.random.default_rng(3))
    d["altered"] = [(xa[i, 0].numpy() * 255).astype(np.uint8) for i in range(2)]
    return d


def window_radius(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    c = (mask.shape[0] - 1) / 2
    return float(np.percentile(np.hypot(ys - c, xs - c), 99.9))


def disc(img: np.ndarray, r: float, bg: str) -> np.ndarray:
    """RGB image with everything outside the window circle painted in the surface colour."""
    if img.ndim == 2:
        img = np.dstack([img] * 3)
    n = img.shape[0]
    yy, xx = np.mgrid[:n, :n]
    c = (n - 1) / 2
    out = img.copy()
    rgb = (np.array(matplotlib.colors.to_rgb(bg)) * 255).astype(np.uint8)
    out[np.hypot(yy - c, xx - c) > r] = rgb
    return out


def photo_with_window(cap, max_h: int = 900) -> tuple[np.ndarray, np.ndarray]:
    img = load_image(cap.source)
    n = cap.fingerprint.fmap.shape[0]
    r = window_radius(cap.registration.mask)
    t = np.linspace(0, 2 * np.pi, 200)
    ring = np.stack([n / 2 + r * np.cos(t), n / 2 + r * np.sin(t), np.ones_like(t)])
    pts = np.linalg.inv(cap.registration.homography) @ ring
    pts = (pts[:2] / pts[2]).T
    s = max_h / img.shape[0]
    small = cv2.resize(img, (int(img.shape[1] * s), max_h), interpolation=cv2.INTER_AREA)
    return small, pts * s


# ------------------------------------------------------------------------------- drawing helpers
class Canvas:
    def __init__(self, h: float, th: dict):
        self.h, self.th = h, th
        self.fig = plt.figure(figsize=(W, h), dpi=DPI, facecolor=th["surface"])

    def ax(self, x, top, w, h, **kw):
        a = self.fig.add_axes([x / W, (self.h - top - h) / self.h, w / W, h / self.h], **kw)
        a.set_facecolor("none")
        a.set_zorder(2)  # above the lane backgrounds
        return a

    def box(self, x, top, w, h, fill, edge, r=0.12, lw=0.8, z=-1):
        p = FancyBboxPatch((x / W, (self.h - top - h) / self.h), w / W, h / self.h,
                           boxstyle=f"round,pad=0,rounding_size={r / W}", transform=self.fig.transFigure,
                           facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z)
        self.fig.patches.append(p)

    def text(self, x, top, s, **kw):
        kw.setdefault("color", self.th["ink"])
        self.fig.text(x / W, (self.h - top) / self.h, s, **kw)

    def arrow(self, x0, x1, top):
        y = (self.h - top) / self.h
        self.fig.patches.append(FancyArrowPatch((x0 / W, y), (x1 / W, y), transform=self.fig.transFigure,
                                                arrowstyle="-|>", mutation_scale=16, lw=1.4,
                                                color=self.th["muted"]))

    def header(self, x, top, w, title, step=None):
        th = self.th
        self.box(x, top, w, 0.42, th["surface"], th["border"], r=0.21, lw=1.0, z=1)
        tx = x + 0.16
        if step is not None:
            cx, cy = x + 0.24, top + 0.21
            # placed in inches so the badge stays round whatever the figure's aspect ratio
            self.fig.patches.append(Circle((cx, self.h - cy), 0.15, transform=self.fig.dpi_scale_trans,
                                           facecolor=th["blue"], edgecolor="none", zorder=2))
            self.text(cx, cy, str(step), ha="center", va="center", fontsize=11, fontweight="bold",
                      color="#ffffff", zorder=3)
            tx = x + 0.48
        self.text(tx, top + 0.29, title, fontsize=12, fontweight="semibold", zorder=3)

    def caption(self, x, top, w, s):
        self.text(x + w / 2, top, s, ha="center", va="top", fontsize=10.5, color=self.th["ink2"], linespacing=1.35)

    def save(self, name):
        self.fig.savefig(OUT / name, dpi=DPI, facecolor=self.th["surface"])
        plt.close(self.fig)


def show(ax, img):
    ax.imshow(img, interpolation="lanczos")
    ax.set_axis_off()


def match_panel(ax, test, enrolled, match, color, th, r, max_lines=140):
    n = test.fingerprint.fmap.shape[0]
    gap = n * 0.08
    a = disc(test.fingerprint.fmap, r, th["lane"])
    b = disc(enrolled.fingerprint.fmap, r, th["lane"])
    ax.imshow(a, extent=(0, n, n, 0))
    ax.imshow(b, extent=(n + gap, 2 * n + gap, n, 0))
    if len(match.pairs):
        qp, ep = test.fingerprint.local.points, enrolled.fingerprint.local.points
        pick = match.pairs[np.linspace(0, len(match.pairs) - 1, min(max_lines, len(match.pairs))).astype(int)]
        for ia, ib in pick:
            (x0, y0), (x1, y1) = qp[ia], ep[ib]
            ax.plot([x0, x1 + n + gap], [y0, y1], color=color, lw=0.7, alpha=0.75)
            ax.plot([x0, x1 + n + gap], [y0, y1], "o", ms=2.2, mfc="none", mec=color, mew=0.6)
    ax.set_xlim(0, 2 * n + gap)
    ax.set_ylim(n, 0)
    ax.set_axis_off()


# ------------------------------------------------------------------------------- figure 1
def fig_how_it_works(d: dict, th: dict, name: str):
    S, LAST, GAP, X0 = 2.4, 3.4, 0.55, 0.4
    xs = [X0 + i * (S + GAP) for i in range(5)]
    lane_h, top0 = 4.45, 0.25
    c = Canvas(top0 * 2 + lane_h * 2 + 0.3, th)
    enrol, test = d["enrol"], d["test"]
    r = window_radius(enrol.registration.mask)
    lanes = [
        ("Storing end: the mint enrols each banknote once", enrol, "Enrolment photo", "iPhone 11, 365 nm UV lamp"),
        ("User end: anyone checks a banknote with a phone", test, "Test photo", "Another angle and lighting"),
    ]
    for li, (title, cap, photo_title, photo_cap) in enumerate(lanes):
        top = top0 + li * (lane_h + 0.3)
        c.box(0.2, top, W - 0.4, lane_h, th["lane"], th["border"], r=0.18)
        c.text(0.45, top + 0.42, f"({'ab'[li]})  {title}", fontsize=14, fontweight="bold")
        ht, it = top + 0.65, top + 1.2  # header top, image top
        cap_top = it + S + 0.15

        # input photo with the found window outlined
        c.header(xs[0], ht, S, photo_title)
        img, ring = photo_with_window(cap)
        a = c.ax(xs[0], it, S, S)
        a.imshow(img)
        a.plot(ring[:, 0], ring[:, 1], color=th["amber"], lw=2)
        a.set_axis_off()
        c.caption(xs[0], cap_top, S, photo_cap)

        c.header(xs[1], ht, S, "Find the window", 1)
        show(c.ax(xs[1], it, S, S), disc(cap.registration.crop, r, th["lane"]))
        c.caption(xs[1], cap_top, S, "Matched to the printed design,\nthen cut out as 768 × 768 pixels")

        c.header(xs[2], ht, S, "Make a spray map", 2)
        show(c.ax(xs[2], it, S, S), disc(cap.fingerprint.fmap, r, th["lane"]))
        c.caption(xs[2], cap_top, S, "Background removed;\nonly glowing droplets kept")

        for i in range(4):
            c.arrow(xs[i] + S + 0.08, xs[i + 1] - 0.08, it + S / 2)

        if li == 0:
            c.header(xs[3], ht, S, "CNN embedding", 3)
            a = c.ax(xs[3], it, S, S * 0.74)
            n = cap.fingerprint.fmap.shape[0]
            a.imshow(disc(cap.fingerprint.fmap, r, th["lane"]))
            clip = Circle(((n - 1) / 2, (n - 1) / 2), r, transform=a.transData)
            for k in (1, 2):
                for ln in (a.axvline(n * k / 3, color=th["blue"], lw=1.6), a.axhline(n * k / 3, color=th["blue"], lw=1.6)):
                    ln.set_clip_path(clip)
            a.set_axis_off()
            b = c.ax(xs[3] + 0.1, it + S * 0.80, S - 0.2, S * 0.2)
            e = cap.fingerprint.embedding.reshape(10, 128)
            b.imshow(np.clip(np.abs(e) / np.percentile(np.abs(e), 98), 0, 1), aspect="auto", interpolation="nearest",
                     cmap=LinearSegmentedColormap.from_list("ramp", th["ramp"]))
            b.set_xticks([]), b.set_yticks([])
            for s in b.spines.values():
                s.set_color(th["border"])
            c.caption(xs[3], cap_top, S, "1,280 numbers from the whole\nwindow and a 3 × 3 grid")

            c.header(xs[4], ht, S, "Enrol and sign", 5)
            a = c.ax(xs[4], it, S, S)
            a.imshow(disc(cap.fingerprint.fmap, r, th["lane"]))
            p = cap.fingerprint.local.points
            a.scatter(p[:, 0], p[:, 1], s=2.5, color=th["amber"], linewidths=0)
            a.set_axis_off()
            c.caption(xs[4], cap_top, S, f"{len(p):,} droplet key points, stored\nin a record signed with Ed25519")
        else:
            c.header(xs[3], ht, S, "Search the database", 3)
            top3 = d["ranked"][:3]
            rh = S / 3
            for k, (nid, cos, rc) in enumerate(top3):
                a = c.ax(xs[3], it + k * rh + 0.04, rh - 0.08, rh - 0.08)
                show(a, disc(rc.fingerprint.fmap, r, th["lane"]))
                right = nid == "IMG_3705"
                c.text(xs[3] + rh + 0.05, it + k * rh + rh / 2 - 0.02, f"#{k + 1}  {nid}", fontsize=11,
                       fontweight="semibold", color=th["ink"])
                c.text(xs[3] + rh + 0.05, it + k * rh + rh / 2 + 0.2,
                       f"similarity {cos:.2f}" + (", right banknote" if right else ""), fontsize=10, color=th["ink2"])
            n = d["gallery_size"]
            c.caption(xs[3], cap_top, S, (f"The 20 closest of {n} banknotes" if n > 1 else "The 20 closest banknotes")
                      + "\ngo on to the droplet check")

            x4 = xs[4]
            c.header(x4, ht, LAST, "Droplet check", 6)
            m = d["match"]
            c.text(x4 + 0.05, it + 0.05, f"Right banknote: {m.inliers} droplets agree", fontsize=10.5, color=th["ink"])
            c.text(x4 + LAST - 0.05, it + 0.05, "GENUINE", ha="right", fontsize=11, fontweight="bold", color=th["good_text"])
            match_panel(c.ax(x4, it + 0.12, LAST, S / 2 - 0.12), test, enrol, m, th["good"], th, r)
            if "wrong" in d:
                wid, wcap, wm = d["wrong"]
                c.text(x4 + 0.05, it + S / 2 + 0.17, f"Closest wrong banknote: {wm.inliers} agree", fontsize=10.5,
                       color=th["ink"])
                c.text(x4 + LAST - 0.05, it + S / 2 + 0.17, "REJECTED", ha="right", fontsize=11, fontweight="bold",
                       color=th["bad"])
                match_panel(c.ax(x4, it + S / 2 + 0.24, LAST, S / 2 - 0.24), test, wcap, wm, th["bad"], th, r)
            c.caption(x4, cap_top, LAST, "GENUINE at 20 or more agreeing droplets;\nNOT FOUND if no candidate reaches 20")
    c.save(name)


# ------------------------------------------------------------------------------- figure 2
def fig_training(d: dict, th: dict, name: str):
    S, X0 = 2.05, 0.4
    c = Canvas(4.25, th)
    top = 0.2
    c.box(0.2, top, W - 0.4, 3.85, th["lane"], th["border"], r=0.18)
    c.text(0.45, top + 0.42, "(c)  Training (Step 4): done once, before the network is used", fontsize=14, fontweight="bold")
    ht, it = top + 0.65, top + 1.2
    r = window_radius(d["enrol"].registration.mask)
    xs = [X0, X0 + S + 0.15, X0 + 2 * S + 0.85, X0 + 3 * S + 1.0]
    c.header(xs[0], ht, 2 * S + 0.15, "Two real photos of one banknote")
    show(c.ax(xs[0], it, S, S), disc(d["enrol"].fingerprint.fmap, r, th["lane"]))
    show(c.ax(xs[1], it, S, S), disc(d["test"].fingerprint.fmap, r, th["lane"]))
    c.text(xs[1] + S + 0.35, it + S / 2 + 0.08, "or", ha="center", fontsize=12, color=th["ink2"])
    c.header(xs[2], ht, 2 * S + 0.15, "Two altered copies of one photo")
    ra = window_radius(cv2.resize(d["enrol"].fingerprint.mask, (448, 448), interpolation=cv2.INTER_NEAREST))
    for k in range(2):
        show(c.ax(xs[2 + k], it, S, S), disc(d["altered"][k], ra, th["lane"]))
    c.caption(xs[0], it + S + 0.15, 4 * S + 1.0,
              "Each step uses 24 banknotes with two views each. The alterations imitate real differences between "
              "photos:\nsmall shifts, rotations and scale changes, brightness, blur, glow around bright droplets, "
              "noise and covered areas.")

    # network icon
    xn = xs[3] + S + 0.6
    c.arrow(xs[3] + S + 0.1, xn - 0.05, it + S / 2)
    c.header(xn, ht, 1.7, "CNN embedding")
    a = c.ax(xn, it + 0.25, 1.7, S - 0.5)
    for k, (h, col) in enumerate(zip([1.0, 0.85, 0.7, 0.55], th["ramp"])):
        x0 = k * 0.42
        a.add_patch(Polygon([[x0, 0.5 - h / 2], [x0 + 0.3, 0.5 - h / 2 + 0.06], [x0 + 0.3, 0.5 + h / 2 - 0.06],
                             [x0, 0.5 + h / 2]], closed=True, facecolor=col, edgecolor=th["blue"], lw=0.8))
    a.set_xlim(-0.05, 1.65), a.set_ylim(-0.05, 1.05), a.set_axis_off()
    c.caption(xn, it + S + 0.15, 1.7, "ResNet-50, first\nthree stages")

    # InfoNCE sketch
    xl = xn + 1.7 + 0.55
    c.arrow(xn + 1.7 + 0.08, xl - 0.05, it + S / 2)
    wl = W - 0.45 - xl
    c.header(xl, ht, wl, "InfoNCE loss")
    a = c.ax(xl, it, wl, S)
    a.set_xlim(-1.25, 1.25), a.set_ylim(-1.1, 1.1), a.set_aspect("equal"), a.set_axis_off()
    a.add_patch(Circle((0, 0), 1.05, facecolor=th["surface"], edgecolor=th["border"], lw=1))
    for sx in (-1, 1):  # two views of one banknote, pulled towards each other
        a.annotate("", xy=(sx * 0.1, 0.12), xytext=(sx * 0.3, 0.12),
                   arrowprops=dict(arrowstyle="-|>", color=th["blue"], lw=1.6, shrinkA=7, shrinkB=0))
        a.plot(sx * 0.3, 0.12, "o", ms=11, color=th["blue"], mec=th["surface"], mew=1.5, zorder=3)
    a.text(0, -0.12, "same banknote", ha="center", va="top", fontsize=10, color=th["ink2"])
    for ang in np.radians([30, 80, 135, 180, 225, 275, 325]):  # other banknotes, pushed away
        u = np.array([np.cos(ang), np.sin(ang)])
        a.annotate("", xy=u * 0.92, xytext=u * 0.62,
                   arrowprops=dict(arrowstyle="-|>", color=th["muted"], lw=1.3, shrinkA=6, shrinkB=0))
        a.plot(*(u * 0.62), "o", ms=9, color=th["muted"], mec=th["surface"], mew=1.2, zorder=3)
    c.caption(xl, it + S + 0.15, wl, "Two views of one banknote are pulled\ntogether; the other 23 are pushed apart")
    c.save(name)


# ------------------------------------------------------------------------------- figure 3
def style_axes(ax, th):
    ax.set_facecolor(th["surface"])
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    for k in ("left", "bottom"):
        ax.spines[k].set_color(th["axis"])
    ax.tick_params(colors=th["muted"], labelcolor=th["ink2"], labelsize=10.5)
    ax.grid(True, axis="y", color=th["grid"], lw=0.8)
    ax.set_axisbelow(True)


def fig_results(th: dict, name: str):
    rows = list(csv.DictReader(open(ROOT / "results" / "all_pairs_inliers.csv")))
    same = np.array([int(r["inliers"]) for r in rows if r["same_note"] == "1"])
    diff = np.array([int(r["inliers"]) for r in rows if r["same_note"] == "0"])
    rep = json.loads((ROOT / "results" / "training" / "train_report.json").read_text())["cross_validation"]

    c = Canvas(5.2, th)
    # (d) droplet check
    a = c.ax(0.95, 1.05, 6.6, 3.45)
    style_axes(a, th)
    edges_d = np.arange(0, diff.max() + 2)
    cnt_d = np.bincount(diff, minlength=len(edges_d) - 1)[: len(edges_d) - 1]
    edges_s = np.geomspace(40, 1000, 15)
    cnt_s, _ = np.histogram(same, edges_s)
    a.set_xscale("symlog", linthresh=1, linscale=0.2)
    a.set_yscale("log")
    a.bar(edges_d[:-1], np.where(cnt_d > 0, cnt_d, np.nan), width=np.diff(edges_d), align="edge",
          color=th["orange"], edgecolor=th["surface"], linewidth=1.5)
    a.bar(edges_s[:-1], np.where(cnt_s > 0, cnt_s, np.nan), width=np.diff(edges_s), align="edge",
          color=th["blue"], edgecolor=th["surface"], linewidth=1.5)
    a.axvline(20, color=th["ink2"], lw=1.2, ls=(0, (4, 3)))
    a.text(23, 150000, "accept at\n20 or more", color=th["ink2"], fontsize=10.5, va="top")
    a.set_xlim(0, 1200), a.set_ylim(0.7, 250000)
    a.set_xticks([0, 1, 10, 100, 1000]), a.set_xticklabels(["0", "1", "10", "100", "1000"])
    a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a.set_yticks([1, 10, 100, 1000, 10000]), a.set_yticklabels(["1", "10", "100", "1,000", "10,000"])
    a.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a.set_xlabel("Droplets that agree (inliers)", color=th["ink2"], fontsize=11)
    a.set_ylabel("Pairs", color=th["ink2"], fontsize=11)
    a.text(1.4, 150000, f"Different banknotes:\n{len(diff):,} pairs, best {diff.max()}\n({cnt_d[0]:,} with 0)",
           color=th["ink"], fontsize=10.5, va="top")
    a.text(48, 40, f"Same banknote: {len(same)} pairs\n{same.min()} to {same.max()}", color=th["ink"], fontsize=10.5)
    c.text(0.95, 0.42, "(d)  The droplet check separates right from wrong banknotes", fontsize=14, fontweight="bold")
    c.text(0.95, 0.72, "Every test photo against every enrolled banknote (138 × 138)", fontsize=11, color=th["ink2"])
    a.legend(handles=[matplotlib.patches.Patch(color=th["blue"], label="Same banknote"),
                      matplotlib.patches.Patch(color=th["orange"], label="Different banknotes")],
             loc="upper right", frameon=False, fontsize=10.5, labelcolor=th["ink2"])

    # (e) shortlist vs database size
    b = c.ax(9.1, 1.05, 6.4, 3.45)
    style_axes(b, th)
    b.grid(True, axis="x", color=th["grid"], lw=0.8)
    sizes = sorted(int(k) for k in rep["frozen"]["shortlist_needed"])
    for key, lab, col in (("frozen", "ImageNet network (not trained)", th["muted"]), ("trained", "Trained network", th["blue"])):
        y = [rep[key]["shortlist_needed"][str(s)] for s in sizes]
        b.plot(sizes, y, "-o", color=col, lw=2, ms=8, mec=th["surface"], mew=2, label=lab)
        v = rep[key]["shortlist_needed"]["1000000"]
        b.annotate(f"{v:,}", (1e6, v), xytext=(-12, 10), textcoords="offset points", ha="right",
                   color=th["ink"], fontsize=10.5, fontweight="semibold")
    b.set_xscale("log"), b.set_yscale("log")
    for axis in (b.xaxis, b.yaxis):
        axis.set_minor_locator(matplotlib.ticker.NullLocator())
    b.set_xticks(sizes), b.set_xticklabels(["1k", "10k", "100k", "1M", "10M"])
    b.set_yticks([1, 10, 100, 1e3, 1e4, 1e5, 1e6]), b.set_yticklabels(["1", "10", "100", "1k", "10k", "100k", "1M"])
    b.set_ylim(1, 5e6)
    b.set_xlabel("Banknotes in the database", color=th["ink2"], fontsize=11)
    b.set_ylabel("Candidates to check", color=th["ink2"], fontsize=11)
    b.legend(loc="upper left", frameon=False, fontsize=10.5, labelcolor=th["ink2"])
    c.text(9.1, 0.42, "(e)  Training shrinks the search", fontsize=14, fontweight="bold")
    c.text(9.1, 0.72, "Shortlist needed to keep 99% of genuine banknotes (projected)", fontsize=11, color=th["ink2"])
    c.save(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gallery", help="folder with the enrolment photos of the other banknotes")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    gallery = list_images(a.gallery) if a.gallery else []
    d = process_examples(gallery)
    for mode, th in THEMES.items():
        sfx = "" if mode == "light" else "_dark"
        fig_how_it_works(d, th, f"how_it_works{sfx}.png")
        fig_training(d, th, f"training{sfx}.png")
        fig_results(th, f"results{sfx}.png")
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
