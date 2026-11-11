"""Turn a canonical spray crop into stored fingerprints.

Two complementary representations are extracted from the same canonical crop:

* a **global CNN embedding** - a CNN with its classifier/FC head removed, truncated at
  a mid-level stage, masked GeM pooling (+ a coarse spatial grid so *where* droplets
  are matters, not just what they look like). One vector per capture; this is the key
  in the vector database and gives fast 1:N shortlisting. Either the frozen ImageNet
  network, or one fine-tuned on spray data with `pes train` (pes/train.py), which adds
  a 1x1-conv projection and stays fully convolutional.

* **local features** - SIFT keypoints on the fluorescence map. Used to re-rank the
  shortlist by geometric consistency (how many droplets line up under one rigid
  transform). This is the high-confidence decision stage: random spray produces
  near-zero consistent matches between different notes, hundreds on the same note.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from .config import Config
from .imaging import fluorescence_map

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_BACKBONES = {
    # name: (constructor, default weights, ordered stage names)
    "resnet50": (torchvision.models.resnet50, "IMAGENET1K_V2", ["layer1", "layer2", "layer3", "layer4"]),
    "resnet18": (torchvision.models.resnet18, "IMAGENET1K_V1", ["layer1", "layer2", "layer3", "layer4"]),
}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


CHECKPOINT_FORMAT = "pes-embedding/1"


def to_input(x: torch.Tensor) -> torch.Tensor:
    """B,1,H,W spray maps in [0, 1] -> ImageNet-normalized B,3,H,W network input."""
    mean, std = _IMAGENET_MEAN.to(x.device), _IMAGENET_STD.to(x.device)
    return (x.repeat(1, 3, 1, 1) - mean) / std


class EmbeddingNet(torch.nn.Module):
    """CNN body (stem + stages up to `cnn_layer`), optional 1x1-conv projection, masked GeM
    over the whole window and over a grid of regions. Fully convolutional: no FC layer.

    The same module is trained by pes/train.py and run by both ends, so a trained model
    is exactly what enrollment and verification use.
    """

    def __init__(self, backbone: str = "resnet50", cnn_layer: str = "layer3", grid: int = 3,
                 gem_p: float = 3.0, proj_dim: int = 0, pretrained: bool = True):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"unsupported backbone {backbone!r}; choose from {list(_BACKBONES)}")
        ctor, weights, stages = _BACKBONES[backbone]
        net = ctor(weights=weights if pretrained else None)
        self.stages = stages[: stages.index(cnn_layer) + 1]
        # Stem + stages up to cnn_layer. avgpool and fc are dropped.
        self.body = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                        *[getattr(net, s) for s in self.stages])
        with torch.no_grad():
            channels = self.body.eval()(torch.zeros(1, 3, 64, 64)).shape[1]
        self.proj = torch.nn.Conv2d(channels, proj_dim, 1) if proj_dim else None
        self.channels = proj_dim or channels
        self.grid, self.gem_p = grid, gem_p
        self.dim = self.channels * (1 + grid * grid)
        self.arch = {"backbone": backbone, "cnn_layer": cnn_layer, "grid": grid, "gem_p": gem_p,
                     "proj_dim": proj_dim}

    def project(self, f: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.proj(f)) if self.proj is not None else f

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.body(x))

    def pool(self, feats: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """feats B,C,h,w; m B,1,h,w window mask in [0, 1] at feature resolution."""
        feats = feats.float()
        parts = [self._gem(feats, m)]
        g = self.grid
        if g > 0:
            h, w = feats.shape[-2:]
            ys = np.linspace(0, h, g + 1).round().astype(int)
            xs = np.linspace(0, w, g + 1).round().astype(int)
            for i in range(g):
                for j in range(g):
                    sl = (..., slice(ys[i], ys[i + 1]), slice(xs[j], xs[j + 1]))
                    parts.append(self._gem(feats[sl], m[sl]))
        out = torch.cat([F.normalize(p, dim=1) for p in parts], dim=1)
        return F.normalize(out, dim=1)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        return self.pool(f, F.interpolate(m, size=f.shape[-2:], mode="area"))

    def _gem(self, f: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        p = self.gem_p
        num = (f.clamp(min=1e-6).pow(p) * m).sum(dim=(2, 3))
        den = m.sum(dim=(2, 3)).clamp(min=1e-6)
        return (num / den).pow(1.0 / p)

    # --- persistence ---------------------------------------------------------------
    def save(self, path: str | Path, cnn_input: int, meta: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": CHECKPOINT_FORMAT, "arch": self.arch, "cnn_input": cnn_input,
                    "state_dict": self.state_dict(), "meta": meta or {}}, path)

    @classmethod
    def load(cls, path: str | Path) -> tuple["EmbeddingNet", dict]:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if ck.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{path} is not a PES embedding checkpoint")
        net = cls(**ck["arch"], pretrained=False)
        net.load_state_dict(ck["state_dict"])
        return net, {"cnn_input": ck["cnn_input"], **ck["meta"]}


class CNNEmbedder:
    """CNN without a classification head -> L2-normalized vector.

    Frozen ImageNet weights by default; `cfg.embed_weights` loads a model trained with
    `pes train`. `net` passes a model directly (used for cross-validation in training).
    """

    def __init__(self, cfg: Config, net: EmbeddingNet | None = None):
        self.cfg = cfg
        if net is None and cfg.embed_weights:
            net, meta = EmbeddingNet.load(cfg.embed_weights)
            trained = {**net.arch, "cnn_input": meta["cnn_input"]}
            for key in ("backbone", "cnn_layer", "grid", "gem_p", "cnn_input"):
                if trained[key] != getattr(cfg, key):
                    raise ValueError(f"{cfg.embed_weights} was trained with {key}={trained[key]!r}, "
                                     f"config has {getattr(cfg, key)!r}")
        self.net = net or EmbeddingNet(cfg.backbone, cfg.cnn_layer, cfg.grid, cfg.gem_p)
        self.device = resolve_device(cfg.device)
        self.net.eval().to(self.device)
        self.channels, self.dim = self.net.channels, self.net.dim

    @torch.no_grad()
    def embed(self, fmap: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.embed_batch([fmap], [mask])[0]

    @torch.no_grad()
    def embed_batch(self, fmaps: list[np.ndarray], masks: list[np.ndarray]) -> np.ndarray:
        s = self.cfg.cnn_input
        x = np.stack([cv2.resize(f, (s, s), interpolation=cv2.INTER_AREA) for f in fmaps])
        x = to_input(torch.from_numpy(x).float().div(255.0).unsqueeze(1)).to(self.device)
        feats = self.net.features(x)                                  # B, C, h, w
        h, w = feats.shape[-2:]
        m = np.stack([cv2.resize(mk, (w, h), interpolation=cv2.INTER_AREA) for mk in masks])
        m = torch.from_numpy(m).float().div(255.0).unsqueeze(1).to(self.device)  # B, 1, h, w
        return self.net.pool(feats, m).cpu().numpy().astype(np.float32)


@dataclass
class LocalFeatures:
    points: np.ndarray       # (N, 2) float32, canonical pixels
    descriptors: np.ndarray  # (N, 128) uint8 SIFT (OpenCV SIFT values are integers 0..255)

    def __len__(self) -> int:
        return len(self.points)

    @cached_property
    def root_sift(self) -> np.ndarray:
        """RootSIFT (Hellinger-normalized) descriptors; unit L2 norm, so dot = cosine."""
        d = self.descriptors.astype(np.float32)
        d /= d.sum(axis=1, keepdims=True) + 1e-7
        return np.sqrt(d)

    def tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """(RootSIFT, points) on `device`, cached so a query is uploaded once per verification."""
        cache = self.__dict__.setdefault("_tensors", {})
        if device not in cache:
            cache[device] = (torch.from_numpy(self.root_sift).to(device),
                             torch.from_numpy(np.ascontiguousarray(self.points)).to(device))
        return cache[device]

    def to_bytes(self) -> tuple[bytes, bytes]:
        return self.points.astype(np.float32).tobytes(), self.descriptors.astype(np.uint8).tobytes()

    @classmethod
    def from_bytes(cls, pts: bytes, desc: bytes) -> "LocalFeatures":
        return cls(np.frombuffer(pts, np.float32).reshape(-1, 2).copy(),
                   np.frombuffer(desc, np.uint8).reshape(-1, 128).copy())


class LocalExtractor:
    def __init__(self, cfg: Config):
        self.sift = cv2.SIFT_create(nfeatures=cfg.local_max_features,
                                    contrastThreshold=cfg.local_contrast_threshold)

    def extract(self, fmap: np.ndarray, mask: np.ndarray) -> LocalFeatures:
        # Erode so descriptors are not dominated by the mask edge.
        inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        kps, desc = self.sift.detectAndCompute(fmap, inner)
        if desc is None:
            return LocalFeatures(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8))
        return LocalFeatures(np.float32([k.pt for k in kps]), np.clip(desc, 0, 255).astype(np.uint8))


@dataclass
class Fingerprint:
    embedding: np.ndarray
    local: LocalFeatures
    fmap: np.ndarray         # uint8 fluorescence map (kept for visualization / audit)
    mask: np.ndarray


class FingerprintExtractor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cnn = CNNEmbedder(cfg)
        self.local = LocalExtractor(cfg)

    def extract(self, crop: np.ndarray, mask: np.ndarray) -> Fingerprint:
        fmap = fluorescence_map(crop, mask)
        return Fingerprint(self.cnn.embed(fmap, mask), self.local.extract(fmap, mask), fmap, mask)
