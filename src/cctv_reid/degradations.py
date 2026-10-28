"""Controlled image degradations for robustness tests (blur, darkness, resolution, occlusion).

All functions take and return uint8 HxWx3 arrays and are deterministic given an rng.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img.copy()
    return _to_uint8(gaussian_filter(img.astype(float), sigma=(sigma, sigma, 0)))


def darken(img: np.ndarray, factor: float) -> np.ndarray:
    """Scale brightness by factor in (0, 1]; 1 leaves the image unchanged."""
    return _to_uint8(img.astype(float) * float(np.clip(factor, 0.0, 1.0)))


def low_resolution(img: np.ndarray, scale: float) -> np.ndarray:
    """Down-sample by block averaging then up-sample by repetition (nearest) to the original size."""
    if scale >= 1:
        return img.copy()
    h, w = img.shape[:2]
    step = max(1, int(round(1.0 / scale)))
    hh, ww = h - h % step, w - w % step
    small = img[:hh, :ww].astype(float).reshape(hh // step, step, ww // step, step, -1).mean(axis=(1, 3))
    up = np.repeat(np.repeat(small, step, axis=0), step, axis=1)
    out = img.astype(float).copy()
    out[:hh, :ww] = up
    return _to_uint8(out)


def occlude(img: np.ndarray, fraction: float, rng: np.random.Generator, fill: int = 0) -> np.ndarray:
    """Cover a horizontal band with a bar of the given fraction of image height, at a random vertical offset."""
    out = img.copy()
    h = img.shape[0]
    band = int(round(np.clip(fraction, 0, 1) * h))
    if band == 0:
        return out
    top = int(rng.integers(0, h - band + 1))
    out[top : top + band] = fill
    return out


def subsample_frames(indices: list[int], keep_every: int) -> list[int]:
    """Lower the sampling rate by keeping every n-th frame while preserving original frame indices."""
    return list(indices[:: max(1, keep_every)])
