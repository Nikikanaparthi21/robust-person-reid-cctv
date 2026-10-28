"""Retrieval, clustering and runtime metrics."""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
from sklearn.metrics import adjusted_rand_score

from .embeddings import l2_normalize


def retrieval_metrics(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, ks=(1, 5)) -> dict[str, float]:
    """Rank-k accuracy and mAP with the standard same-identity-same-camera exclusion."""
    q = l2_normalize(np.asarray(q_feats, dtype=float))
    g = l2_normalize(np.asarray(g_feats, dtype=float))
    q_ids, q_cams = np.asarray(q_ids), np.asarray(q_cams)
    g_ids, g_cams = np.asarray(g_ids), np.asarray(g_cams)
    dist = 1.0 - q @ g.T

    hits = {k: 0 for k in ks}
    aps, valid = [], 0
    for i in range(len(q)):
        keep = ~((g_ids == q_ids[i]) & (g_cams == q_cams[i]))
        order = np.argsort(dist[i][keep], kind="stable")
        match = (g_ids[keep][order] == q_ids[i])
        if not match.any():
            continue
        valid += 1
        for k in ks:
            hits[k] += bool(match[:k].any())
        precision = np.cumsum(match) / (np.arange(len(match)) + 1)
        aps.append(float((precision * match).sum() / match.sum()))
    if valid == 0:
        return {**{f"rank{k}": float("nan") for k in ks}, "mAP": float("nan"), "n_queries": 0}
    return {**{f"rank{k}": hits[k] / valid for k in ks}, "mAP": float(np.mean(aps)), "n_queries": valid}


def _contingency(true_ids, labels) -> np.ndarray:
    _, t = np.unique(true_ids, return_inverse=True)
    _, c = np.unique(labels, return_inverse=True)
    table = np.zeros((t.max() + 1, c.max() + 1), dtype=np.int64)
    np.add.at(table, (t, c), 1)
    return table


def _pairs(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    return x * (x - 1) // 2


def pairwise_prf(true_ids, labels) -> dict[str, float]:
    """Pairwise precision, recall and F1 over all item pairs, from the contingency table."""
    table = _contingency(true_ids, labels)
    tp = _pairs(table).sum()
    pred_pairs = _pairs(table.sum(axis=0)).sum()
    true_pairs = _pairs(table.sum(axis=1)).sum()
    precision = tp / pred_pairs if pred_pairs else 1.0
    recall = tp / true_pairs if true_pairs else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def cluster_metrics(true_ids, labels) -> dict[str, float]:
    """Pairwise P/R/F1, purity, ARI, plus false-merge and false-split summaries."""
    true_ids, labels = np.asarray(true_ids), np.asarray(labels)
    table = _contingency(true_ids, labels)
    out = pairwise_prf(true_ids, labels)
    out["purity"] = float(table.max(axis=0).sum() / table.sum())
    out["ari"] = float(adjusted_rand_score(true_ids, labels))
    out["n_clusters"] = int(table.shape[1])
    out["n_identities"] = int(table.shape[0])
    # Duplicate identities: average number of clusters each true identity is spread across.
    out["clusters_per_identity"] = float((table > 0).sum(axis=1).mean())
    # False merges: clusters holding more than one true identity.
    out["merged_cluster_fraction"] = float(((table > 0).sum(axis=0) > 1).mean())
    return out


@contextmanager
def timer(store: dict, key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = store.get(key, 0.0) + time.perf_counter() - t0
