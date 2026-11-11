"""Register an arbitrary photo of a note to the canonical frame.

photo --(SIFT on printed design + MAGSAC homography)--> template --(affine)--> canonical

This is what makes matching work "from a different angle": after this step the spray
window of every capture sits in the same N x N square, upright, with perspective
removed, regardless of how the phone was held.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config
from .imaging import gray_clahe, resize_max_side
from .template import NoteTemplate, circle_mask


class RegistrationError(RuntimeError):
    pass


@dataclass
class Registration:
    crop: np.ndarray          # RGB uint8, canonical N x N
    mask: np.ndarray          # uint8, 255 inside the spray window and inside the photo
    homography: np.ndarray    # 3x3, full-resolution photo pixels -> canonical pixels
    inliers: int
    matches: int
    roi_coverage: float

    @property
    def inlier_ratio(self) -> float:
        return self.inliers / max(self.matches, 1)


class Registrar:
    def __init__(self, template: NoteTemplate, cfg: Config):
        self.cfg = cfg
        self.template = template
        self.sift = cv2.SIFT_create(nfeatures=cfg.reg_sift_features)
        # Keypoints only from the printed design: mask out the (slightly grown) window.
        design_mask = cv2.bitwise_not(template.roi_mask(grow=1.06))
        kps, desc = self.sift.detectAndCompute(gray_clahe(template.image), design_mask)
        if desc is None or len(kps) < 100:
            raise RegistrationError("template has too few keypoints")
        self.t_pts = np.float32([k.pt for k in kps])
        self.t_desc = desc
        self.matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})
        self.matcher.add([self.t_desc])
        self.matcher.train()
        n = cfg.canonical_size
        self.to_canonical = template.canonical_affine(n, cfg.roi_margin)
        if template.canonical_mask is not None:
            if template.mask_roi_margin is not None and abs(template.mask_roi_margin - cfg.roi_margin) > 1e-9:
                raise RegistrationError(f"template mask was built for roi_margin={template.mask_roi_margin}, "
                                        f"config uses {cfg.roi_margin}; rebuild the template")
            self.roi_mask = cv2.resize(template.canonical_mask, (n, n), interpolation=cv2.INTER_NEAREST)
        else:
            self.roi_mask = circle_mask(n, cfg.roi_margin)

    def register(self, rgb: np.ndarray) -> Registration:
        cfg = self.cfg
        small, s = resize_max_side(rgb, cfg.reg_max_side)
        kps, desc = self.sift.detectAndCompute(gray_clahe(small), None)
        if desc is None or len(kps) < cfg.reg_min_inliers:
            raise RegistrationError("not enough texture in photo - is the note in frame and in focus?")

        knn = self.matcher.knnMatch(desc, k=2)
        good = [m for m, n2 in (p for p in knn if len(p) == 2) if m.distance < cfg.reg_ratio * n2.distance]
        if len(good) < cfg.reg_min_inliers:
            raise RegistrationError(f"note design not found ({len(good)} candidate matches)")
        src = np.float32([kps[m.queryIdx].pt for m in good])
        dst = self.t_pts[[m.trainIdx for m in good]]
        H, inl = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, cfg.reg_ransac_thresh,
                                    maxIters=20000, confidence=0.9999)
        n_inl = int(inl.sum()) if inl is not None else 0
        if H is None or n_inl < cfg.reg_min_inliers:
            raise RegistrationError(f"note design not found ({n_inl} geometric inliers)")

        # Full-resolution photo -> canonical.
        to_small = np.diag([s, s, 1.0])
        M = self.to_canonical @ H @ to_small
        crop, valid = self._warp(rgb, M)
        roi = self.roi_mask > 0
        coverage = float((valid[roi] > 0).mean())
        if coverage < cfg.reg_min_roi_coverage:
            raise RegistrationError(f"spray window only {coverage:.0%} inside the photo - reframe the shot")
        mask = cv2.bitwise_and(self.roi_mask, valid)
        return Registration(crop=crop, mask=mask, homography=M, inliers=n_inl,
                            matches=len(good), roi_coverage=coverage)

    def _warp(self, rgb: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Warp with proper anti-aliasing: the ROI is usually larger in the photo than
        in the canonical frame, so cut out its bounding box and area-downsample first."""
        n = self.cfg.canonical_size
        Minv = np.linalg.inv(M)
        corners = cv2.perspectiveTransform(np.float32([[[0, 0]], [[n, 0]], [[n, n]], [[0, n]]]), Minv)[:, 0]
        h, w = rgb.shape[:2]
        x0, y0 = np.floor(corners.min(0)).astype(int) - 4
        x1, y1 = np.ceil(corners.max(0)).astype(int) + 4
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 - x0 < 8 or y1 - y0 < 8:
            raise RegistrationError("spray window is outside the photo")
        patch = rgb[y0:y1, x0:x1]
        area_ratio = ((x1 - x0) * (y1 - y0)) / float(n * n)
        f = 1.0 / np.sqrt(area_ratio) if area_ratio > 1.5 else 1.0
        if f < 1.0:
            patch = cv2.resize(patch, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        back = np.array([[1 / f, 0, x0], [0, 1 / f, y0], [0, 0, 1]], float)
        M2 = M @ back
        crop = cv2.warpPerspective(patch, M2, (n, n), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        valid = cv2.warpPerspective(np.full(patch.shape[:2], 255, np.uint8), M2, (n, n),
                                    flags=cv2.INTER_NEAREST, borderValue=0)
        # The patch is clipped to the photo, so `valid` also marks what the photo actually saw.
        return crop, valid
