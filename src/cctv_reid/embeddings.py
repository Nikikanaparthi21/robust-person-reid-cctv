"""Frame embeddings and tracklet-level aggregation.

Pooling functions take a (T, D) array of frame embeddings for one tracklet and return one
L2-normalised (D,) vector. Encoders are loaded lazily so the rest of the pipeline runs
without torch.
"""

from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def mean_pool(frames: np.ndarray, quality: np.ndarray | None = None) -> np.ndarray:
    """Uniform average of normalised frame embeddings."""
    return l2_normalize(l2_normalize(frames).mean(axis=0))


def quality_filtered_pool(frames: np.ndarray, quality: np.ndarray, keep: float = 0.5) -> np.ndarray:
    """Average only the top `keep` fraction of frames ranked by a quality score (at least one)."""
    quality = np.asarray(quality, dtype=float)
    n_keep = max(1, int(np.ceil(keep * len(frames))))
    idx = np.argsort(-quality)[:n_keep]
    return mean_pool(frames[idx])


def quality_weighted_pool(frames: np.ndarray, quality: np.ndarray, temperature: float = 0.2) -> np.ndarray:
    """Softmax-weighted average using a per-frame quality score."""
    q = np.asarray(quality, dtype=float)
    w = np.exp((q - q.max()) / temperature)
    w /= w.sum()
    return l2_normalize((l2_normalize(frames) * w[:, None]).sum(axis=0))


def attention_pool(frames: np.ndarray, quality: np.ndarray | None = None, temperature: float = 0.1) -> np.ndarray:
    """Parameter-free temporal attention: frames closer to the tracklet consensus weigh more.

    Weights are a softmax over cosine similarity to the median-direction consensus, so frames
    corrupted by occlusion or blur (which drift away from the consensus) are down-weighted
    without needing an external quality score.
    """
    f = l2_normalize(frames)
    consensus = l2_normalize(np.median(f, axis=0))
    sim = f @ consensus
    w = np.exp((sim - sim.max()) / temperature)
    w /= w.sum()
    return l2_normalize((f * w[:, None]).sum(axis=0))


POOLERS = {
    "mean": mean_pool,
    "quality_filtered": quality_filtered_pool,
    "quality_weighted": quality_weighted_pool,
    "attention": attention_pool,
}


def pool_tracklets(frame_embeddings: np.ndarray, tracklet_index: dict[str, list[int]], method: str,
                   quality: np.ndarray | None = None) -> tuple[list[str], np.ndarray]:
    """Pool every tracklet; return (track keys, (n_tracklets, D) matrix)."""
    pool = POOLERS[method]
    keys, out = [], []
    for key, idx in tracklet_index.items():
        q = None if quality is None else quality[idx]
        if method in ("quality_filtered", "quality_weighted") and q is None:
            raise ValueError(f"{method} pooling needs per-frame quality scores")
        out.append(pool(frame_embeddings[idx], q) if q is not None else pool(frame_embeddings[idx]))
        keys.append(key)
    return keys, np.stack(out)


def frame_sharpness(gray: np.ndarray) -> float:
    """Cheap image-quality proxy: variance of a Laplacian-style second difference."""
    g = np.asarray(gray, dtype=float)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


class TorchreidEncoder:
    """OSNet encoder via torchreid. Requires `pip install torchreid torch`; not used in tests.

    Weights must be downloaded by the user and kept out of Git (see .gitignore).
    """

    def __init__(self, model_name: str = "osnet_x1_0", weights: str | None = None, device: str = "cpu"):
        import torch  # noqa: WPS433 - lazy import
        import torchreid

        self.torch = torch
        self.device = device
        self.model = torchreid.models.build_model(model_name, num_classes=1000, pretrained=weights is None)
        if weights:
            torchreid.utils.load_pretrained_weights(self.model, weights)
        self.model.eval().to(device)

    def encode(self, batch: np.ndarray) -> np.ndarray:
        """batch: (N, 256, 128, 3) uint8 RGB -> (N, D) float32 embeddings."""
        torch = self.torch
        x = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = ((x - mean) / std).to(self.device)
        with torch.no_grad():
            feats = self.model(x)
        return feats.cpu().numpy().astype("float32")


class ResNet50Encoder:
    """Frozen ImageNet ResNet-50 (torchvision) global-pooled 2048-d features.

    A simple fallback when torchreid/OSNet is hard to install. Not a ReID-trained model, so expect
    lower absolute accuracy than OSNet; use it to compare frame vs tracklet aggregation, not to claim SOTA.
    Requires torch + torchvision; not used in tests.
    """

    def __init__(self, device: str = "cpu"):
        import torch
        import torchvision

        self.torch = torch
        self.device = device
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Identity()
        self.model = model.eval().to(device)

    def encode(self, batch: np.ndarray) -> np.ndarray:
        """batch: (N, 256, 128, 3) uint8 RGB -> (N, 2048) float32 embeddings."""
        torch = self.torch
        x = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = ((x - mean) / std).to(self.device)
        with torch.no_grad():
            feats = self.model(x)
        return feats.cpu().numpy().astype("float32")
