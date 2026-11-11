"""The note template: the geometric anchor every photo is registered against.

The printed banknote design (guilloche, numerals, text) is identical on every note,
so a single reference photo of the area around the spray window acts as a
high-precision "anchor mark". The spray window itself is described by an ellipse in
template pixel coordinates; it is masked out when extracting template keypoints so
the reference note's own spray never influences registration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


@dataclass
class NoteTemplate:
    image: np.ndarray                     # RGB uint8, the area of the note around the spray window
    roi_center: tuple[float, float]       # spray window ellipse, template pixels
    roi_axes: tuple[float, float]         # semi-axes (x, y)
    roi_angle: float = 0.0                # degrees
    name: str = "note"
    # Optional canonical-frame mask (255 = usable spray area) with printed design removed.
    # Built by `print_free_mask` from many registered notes, for a given roi_margin.
    canonical_mask: np.ndarray | None = None
    mask_roi_margin: float | None = None

    def roi_mask(self, grow: float = 1.0) -> np.ndarray:
        mask = np.zeros(self.image.shape[:2], np.uint8)
        axes = (round(self.roi_axes[0] * grow), round(self.roi_axes[1] * grow))
        center = (round(self.roi_center[0]), round(self.roi_center[1]))
        cv2.ellipse(mask, center, axes, self.roi_angle, 0, 360, 255, -1)
        return mask

    def canonical_affine(self, size: int, margin: float) -> np.ndarray:
        """3x3 map: template pixels -> canonical N x N frame.

        The window ellipse becomes a centred circle, which also removes most of the
        reference photo's own perspective tilt.
        """
        radius = size / 2 / (1.0 + margin)
        cx, cy = self.roi_center
        th = np.deg2rad(self.roi_angle)
        to_origin = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], float)
        unrotate = np.array([[np.cos(th), np.sin(th), 0], [-np.sin(th), np.cos(th), 0], [0, 0, 1]], float)
        circularize = np.diag([radius / self.roi_axes[0], radius / self.roi_axes[1], 1.0])
        to_center = np.array([[1, 0, size / 2], [0, 1, size / 2], [0, 0, 1]], float)
        return to_center @ circularize @ unrotate @ to_origin

    # --- persistence ---------------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(directory / "template.png"), cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
        meta = {
            "name": self.name,
            "roi_center": list(self.roi_center),
            "roi_axes": list(self.roi_axes),
            "roi_angle": self.roi_angle,
            "mask_roi_margin": self.mask_roi_margin,
        }
        (directory / "template.json").write_text(json.dumps(meta, indent=2))
        if self.canonical_mask is not None:
            cv2.imwrite(str(directory / "roi_mask.png"), self.canonical_mask)

    @classmethod
    def load(cls, directory: str | Path) -> "NoteTemplate":
        directory = Path(directory)
        if not (directory / "template.json").exists():
            raise FileNotFoundError(f"no template in {directory}; run `python -m pes build-template` first")
        meta = json.loads((directory / "template.json").read_text())
        image = cv2.cvtColor(cv2.imread(str(directory / "template.png")), cv2.COLOR_BGR2RGB)
        mask_path = directory / "roi_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        return cls(
            image=image,
            roi_center=tuple(meta["roi_center"]),
            roi_axes=tuple(meta["roi_axes"]),
            roi_angle=meta.get("roi_angle", 0.0),
            name=meta.get("name", "note"),
            canonical_mask=mask,
            mask_roi_margin=meta.get("mask_roi_margin"),
        )

    @classmethod
    def from_reference(
        cls,
        rgb: np.ndarray,
        crop_box: tuple[int, int, int, int],
        roi_center: tuple[float, float],
        roi_axes: tuple[float, float],
        roi_angle: float = 0.0,
        rotate: int = 0,
        scale: float = 0.5,
        name: str = "note",
    ) -> "NoteTemplate":
        """Build a template from a reference photo.

        Coordinates (crop_box = x0, y0, x1, y1 and the ROI ellipse) are given in the
        reference photo *after* rotation and scaling, i.e. in the frame you would see
        when viewing the rotated, `scale`-resized image.
        """
        if ROTATIONS[rotate] is not None:
            rgb = cv2.rotate(rgb, ROTATIONS[rotate])
        if scale != 1.0:
            rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        x0, y0, x1, y1 = crop_box
        image = np.ascontiguousarray(rgb[y0:y1, x0:x1])
        center = (roi_center[0] - x0, roi_center[1] - y0)
        return cls(image=image, roi_center=center, roi_axes=tuple(roi_axes), roi_angle=roi_angle, name=name)


def circle_mask(size: int, margin: float, shrink: float = 1.0) -> np.ndarray:
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (size // 2, size // 2), round(size / 2 / (1.0 + margin) * shrink), 255, -1)
    return mask


def print_free_mask(registered_grays: list[np.ndarray], margin: float, stroke_thresh: int = 30,
                    density_thresh: float = 0.15, shrink: float = 0.97) -> np.ndarray:
    """Canonical mask of the spray window with the printed design cut out.

    Takes CLAHE-gray canonical crops of many *different* notes. Their per-pixel median
    keeps what is identical on every note (the print) and washes out the random
    spray. Dense dark strokes in the median are print; the window minus those
    regions is where a fingerprint may come from. Matching print would make every
    note look like every other note.
    """
    med = np.median(np.stack(registered_grays).astype(np.float32), axis=0).astype(np.uint8)
    size = med.shape[0]
    blackhat = cv2.morphologyEx(med, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    density = cv2.GaussianBlur((blackhat > stroke_thresh).astype(np.float32), (0, 0), 10)
    printed = cv2.dilate((density > density_thresh).astype(np.uint8) * 255,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    roi = cv2.bitwise_and(circle_mask(size, margin, shrink), cv2.bitwise_not(printed))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(roi)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels == largest) * 255).astype(np.uint8)
