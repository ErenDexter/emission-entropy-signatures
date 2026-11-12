"""Central configuration for the EES (Emission Entropy Signatures) pipeline.

Every tunable lives here so that the enrollment (mint) side and the verification
(user) side are guaranteed to run the *same* processing. A mismatch between the two
(e.g. a different canonical size or CNN layer) silently destroys matching accuracy,
so the config fingerprint is also stored with every enrolled record.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # --- locations -------------------------------------------------------------
    template_dir: Path = ROOT / "assets" / "template"
    db_dir: Path = ROOT / "data" / "db"
    keys_dir: Path = ROOT / "data" / "keys"
    collection: str = "ees_notes"
    # None -> embedded Qdrant stored under db_dir. Set e.g. "http://localhost:6333"
    # to use a Qdrant server instead (same code path, HNSW index, multi-client).
    qdrant_url: str | None = None

    # --- registration: photo -> canonical frame -------------------------------
    reg_max_side: int = 1600          # photo is downscaled to this for keypoint matching
    reg_sift_features: int = 8000
    reg_ratio: float = 0.8            # Lowe ratio test
    reg_ransac_thresh: float = 6.0    # pixels, in template coordinates
    reg_min_inliers: int = 40
    reg_min_roi_coverage: float = 0.97  # fraction of the spray ROI that must be inside the photo

    # --- canonical frame -------------------------------------------------------
    canonical_size: int = 768         # spray ROI is warped into an N x N square
    roi_margin: float = 0.12          # border around the ROI circle, as a fraction of its radius

    # --- global CNN embedding (vector DB key) ----------------------------------
    # Measured on the 2-Taka set (2nd-angle queries vs 1st-angle enrollments):
    # layer3 >> layer4 > layer2, and spatial grid pooling is essential - global pooling
    # alone reaches only 67% top-1 because all sprays share the same texture statistics.
    backbone: str = "resnet50"
    cnn_layer: str = "layer3"         # truncated before layer4/avgpool/fc
    cnn_input: int = 448
    gem_p: float = 3.0
    grid: int = 3                     # global + grid x grid regional GeM (0 = global only)
    device: str = "auto"
    # Checkpoint from `ees train` (fine-tuned body + 1x1-conv projection). None -> frozen
    # ImageNet weights. Its content hash becomes part of the feature fingerprint.
    embed_weights: Path | None = None

    # --- local features (geometric verification) -------------------------------
    local_max_features: int = 3000
    local_contrast_threshold: float = 0.01
    match_ratio: float = 0.8
    match_max_shift: float = 0.15     # max displacement of a match, fraction of canonical size
    match_ransac_thresh: float = 4.0  # canonical pixels

    # --- decision ----------------------------------------------------------------
    # Measured on 138 notes: genuine cross-angle pairs 42..990 inliers, worst of 18,906 impostor pairs 10.
    shortlist_k: int = 20             # candidates pulled from the vector DB for re-ranking
    accept_inliers: int = 20          # min geometric inliers to call a match genuine
    duplicate_inliers: int = 20       # enrollment refuses a pattern this close to an existing note

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        cfg = cls()
        if path:
            data = json.loads(Path(path).read_text())
            types = {f.name: str(f.type) for f in fields(cls)}
            for key, value in data.items():
                if key not in types:
                    raise KeyError(f"unknown config key: {key}")
                if value is not None and "Path" in types[key]:
                    value = Path(value)
                setattr(cfg, key, value)
        return cfg

    def feature_fingerprint(self) -> str:
        """Hash of every setting that changes the stored vectors.

        Records enrolled under a different fingerprint are not comparable and the
        verifier refuses to mix them.
        """
        keys = [
            "canonical_size", "roi_margin", "backbone", "cnn_layer", "cnn_input",
            "gem_p", "grid", "local_max_features", "local_contrast_threshold",
        ]
        values = {k: getattr(self, k) for k in keys}
        if self.embed_weights:  # absent for the frozen model, so its fingerprint is unchanged
            values["embed_weights"] = hashlib.sha256(Path(self.embed_weights).read_bytes()).hexdigest()
        payload = json.dumps(values, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}
