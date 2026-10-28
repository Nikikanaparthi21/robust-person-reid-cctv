"""Identity clustering on cosine distance with threshold selection on a validation split."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering

from .embeddings import l2_normalize
from .evaluation import pairwise_prf


def cluster(embeddings: np.ndarray, method: str, threshold: float, min_samples: int = 2) -> np.ndarray:
    """Cluster L2-normalised embeddings. Returns integer labels; DBSCAN noise gets unique labels.

    `threshold` is a cosine-distance cut: DBSCAN eps, or the agglomerative distance_threshold.
    """
    x = l2_normalize(np.asarray(embeddings, dtype=float))
    if method == "dbscan":
        labels = DBSCAN(eps=threshold, min_samples=min_samples, metric="cosine").fit_predict(x)
    elif method == "agglomerative":
        labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=threshold, metric="cosine", linkage="average"
        ).fit_predict(x)
    else:
        raise ValueError(f"unknown clustering method: {method}")
    return relabel_noise(labels)


def relabel_noise(labels: np.ndarray) -> np.ndarray:
    """Give each DBSCAN noise point (-1) its own cluster so singletons are counted honestly."""
    labels = np.asarray(labels).copy()
    nxt = labels.max() + 1 if labels.size else 0
    for i in np.where(labels == -1)[0]:
        labels[i] = nxt
        nxt += 1
    return labels


def tune_threshold(embeddings: np.ndarray, true_ids: np.ndarray, method: str,
                   grid: np.ndarray | None = None, min_samples: int = 2) -> tuple[float, float]:
    """Pick the cosine-distance threshold with the best pairwise F1 on a validation set."""
    if grid is None:
        grid = np.linspace(0.05, 0.9, 18)
    best_t, best_f1 = float(grid[0]), -1.0
    for t in grid:
        labels = cluster(embeddings, method, float(t), min_samples)
        f1 = pairwise_prf(true_ids, labels)["f1"]
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1
