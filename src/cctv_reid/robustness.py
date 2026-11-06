"""Torch-free pieces of the robustness experiments (issue #24): conditions, quality scores, sampling."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from . import degradations, embeddings, io_utils

# name -> (family, level, function(img, rng) -> img). Applied to the 128x256 encoder input.
CONDITIONS = {
    "clean": ("clean", 0, lambda im, rng: im),
    "blur_s2": ("blur", 2, lambda im, rng: degradations.blur(im, 2.0)),
    "blur_s4": ("blur", 4, lambda im, rng: degradations.blur(im, 4.0)),
    "dark_0.5": ("darkening", 0.5, lambda im, rng: degradations.darken(im, 0.5)),
    "dark_0.25": ("darkening", 0.25, lambda im, rng: degradations.darken(im, 0.25)),
    "lowres_0.5": ("low_resolution", 0.5, lambda im, rng: degradations.low_resolution(im, 0.5)),
    "lowres_0.25": ("low_resolution", 0.25, lambda im, rng: degradations.low_resolution(im, 0.25)),
    "occl_0.25": ("occlusion", 0.25, lambda im, rng: degradations.occlude(im, 0.25, rng)),
    "occl_0.5": ("occlusion", 0.5, lambda im, rng: degradations.occlude(im, 0.5, rng)),
}


def apply_condition(img: np.ndarray, name: str, rng: np.random.Generator) -> np.ndarray:
    return CONDITIONS[name][2](img, rng)


def sharpness_from_gray(gray: np.ndarray) -> np.ndarray:
    """(N, H, W) grayscale images -> 0-1 sharpness percentile rank, so softmax temperatures behave."""
    raw = np.array([embeddings.frame_sharpness(g) for g in gray])
    return (rankdata(raw) - 1) / max(1, len(raw) - 1)


def sharpness_quality(images: np.ndarray) -> np.ndarray:
    """(N, H, W, 3) RGB images -> sharpness percentile rank."""
    return sharpness_from_gray(images.mean(axis=3))


def subsample_indices(names: list[str], keep_every: int) -> np.ndarray:
    """Indices that keep every n-th frame of each tracklet (names must be in frame order per tracklet)."""
    seen: dict[tuple, int] = {}
    keep = []
    for i, n in enumerate(names):
        pid, cam, trk, _ = io_utils.parse_mars_name(n)
        c = seen.get((pid, cam, trk), 0)
        if c % max(1, keep_every) == 0:
            keep.append(i)
        seen[(pid, cam, trk)] = c + 1
    return np.array(keep, dtype=int)
