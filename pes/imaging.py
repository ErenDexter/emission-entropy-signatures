"""Image I/O and photometric normalization."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

try:  # iPhone photos are HEIC
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover
    pillow_heif = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}


def load_image(path: str | Path) -> np.ndarray:
    """Load any supported photo as an RGB uint8 array, honouring EXIF orientation."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return np.ascontiguousarray(np.asarray(im.convert("RGB")))


def list_images(folder: str | Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def resize_max_side(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Downscale so the longest side is at most max_side. Returns (image, scale)."""
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return img, s


def gray_clahe(img: np.ndarray, clip: float = 3.0, tiles: int = 8) -> np.ndarray:
    """Grayscale + local contrast equalization. Makes the printed note design look
    alike under the very different UV/ambient lighting of different captures."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(g)


def fluorescence_map(crop: np.ndarray, mask: np.ndarray, sigma: float = 24.0) -> np.ndarray:
    """Isolate the fluorescent spray from paper/print/illumination.

    The spray glows brighter than its local surroundings under UV in every capture,
    whereas absolute colour and brightness depend on the lamp, white balance and
    viewing angle. So: take luminance, remove the slowly varying background
    (lamp falloff, watermark shading) and keep only the *positive* residual - printed
    ink and paper texture are darker than or equal to the paper, never brighter.

    Returns a uint8 map where bright = spray.
    """
    lum = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    m = (mask > 0).astype(np.float32)
    # Normalized (mask-aware) Gaussian background estimate.
    num = cv2.GaussianBlur(lum * m, (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma) + 1e-6
    detail = lum - num / den
    inside = detail[mask > 0]
    lo, hi = np.median(inside), np.percentile(inside, 99.5)
    out = np.clip((detail - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    out[mask == 0] = 0.0
    return (out * 255).astype(np.uint8)
