"""Geometric verification of two fingerprints.

Both fingerprints live in the same canonical frame, so a genuine pair differs only by a
small residual misregistration (paper curl, homography error). We therefore:

1. compare RootSIFT descriptors only between keypoints that are spatially close
   (|shift| < match_max_shift), which cuts accidental matches by orders of magnitude;
2. keep mutual nearest neighbours that pass Lowe's ratio test;
3. fit a similarity transform with RANSAC and require it to be near identity.

The inlier count is the match score. It is interpretable ("97 droplets line up") and,
unlike a cosine similarity, its impostor distribution sits essentially at zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from .config import Config
from .fingerprint import LocalFeatures, resolve_device


@dataclass
class MatchResult:
    inliers: int
    putative: int
    transform: np.ndarray | None = None          # 2x3, a -> b
    pairs: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), int))  # inlier index pairs

    @property
    def residual_shift(self) -> float:
        return float(np.hypot(*self.transform[:, 2])) if self.transform is not None else float("nan")


class GeometricMatcher:
    def __init__(self, cfg: Config):
        self.ratio = cfg.match_ratio
        self.max_shift = cfg.match_max_shift * cfg.canonical_size
        self.thresh = cfg.match_ransac_thresh
        # ~3000 x 3000 similarity matrices per comparison: torch (GPU, or multi-threaded CPU)
        # is 20-50x faster than numpy here, mostly because of the column-wise argmax.
        self.device = resolve_device(cfg.device)

    def match(self, a: LocalFeatures, b: LocalFeatures) -> MatchResult:
        if len(a) < 4 or len(b) < 4:
            return MatchResult(0, 0)
        ia, ib = self.putative(a, b)
        return self.fit(a.points, b.points, ia, ib)

    @torch.no_grad()
    def putative(self, a: LocalFeatures, b: LocalFeatures) -> tuple[np.ndarray, np.ndarray]:
        """Mutual nearest neighbours passing the ratio test, among spatially close keypoints."""
        (da, pa), (db, pb) = a.tensors(self.device), b.tensors(self.device)
        sim = da @ db.T                                                      # cosine, RootSIFT is unit-L2
        sim.masked_fill_(torch.cdist(pa, pb) > self.max_shift, -1.0)        # spatially implausible

        rows = torch.arange(len(a), device=self.device)
        best_b = sim.argmax(1)
        best_a = sim.argmax(0)
        mutual = best_a[best_b] == rows
        s1 = sim[rows, best_b].clone()
        sim[rows, best_b] = -1.0
        s2 = sim.max(1).values
        # ratio test on descriptor distances: dist = sqrt(2 - 2 sim)
        d1, d2 = (2 - 2 * s1).clamp(min=0).sqrt(), (2 - 2 * s2).clamp(min=0).sqrt()
        ok = mutual & (s1 > 0) & (d1 < self.ratio * d2)
        return rows[ok].cpu().numpy(), best_b[ok].cpu().numpy()

    def fit(self, pa: np.ndarray, pb: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> MatchResult:
        """RANSAC similarity on putative matches; must be close to the identity."""
        if len(ia) < 4:
            return MatchResult(0, len(ia))
        M, inl = cv2.estimateAffinePartial2D(pa[ia], pb[ib], method=cv2.RANSAC,
                                             ransacReprojThreshold=self.thresh, maxIters=5000,
                                             confidence=0.999)
        if M is None or inl is None:
            return MatchResult(0, len(ia))
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        angle = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        if not (0.85 < scale < 1.18 and abs(angle) < 12):
            return MatchResult(0, len(ia), M)                                 # not a near-identity map
        keep = inl.ravel().astype(bool)
        return MatchResult(int(keep.sum()), len(ia), M, np.stack([ia[keep], ib[keep]], 1))
