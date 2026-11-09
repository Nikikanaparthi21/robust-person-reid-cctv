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


# ---------- tracking metrics (IDF1, ID switches) ----------
def tracking_metrics(gt, pred, iou_thresh: float = 0.5) -> dict[str, float]:
    """IDF1 and ID switches for one sequence.

    ``gt`` and ``pred`` map ``frame -> list of (track_id, [x1, y1, x2, y2])``.
    IDF1 uses one global one-to-one assignment between ground-truth and predicted
    identities (Ristani et al., 2016): IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN).
    An ID switch is counted when a ground-truth identity's matched predicted id changes
    between consecutive matched frames. Only meaningful on data with tracking labels
    (MARS has none; see docs/experiment-protocol.md).
    """
    from scipy.optimize import linear_sum_assignment

    from .tracking import iou_matrix

    frames = sorted(set(gt) | set(pred))
    gt_ids = sorted({i for f in gt.values() for i, _ in f})
    pr_ids = sorted({i for f in pred.values() for i, _ in f})
    n_gt = sum(len(v) for v in gt.values())
    n_pr = sum(len(v) for v in pred.values())
    if not gt_ids or not pr_ids:
        return {"idf1": 0.0, "idp": 0.0, "idr": 0.0, "id_switches": 0, "gt_detections": n_gt, "pred_detections": n_pr}
    gi = {i: k for k, i in enumerate(gt_ids)}
    pi = {i: k for k, i in enumerate(pr_ids)}

    # Co-occurrence counts (frames where a gt id and a predicted id overlap enough).
    overlap = np.zeros((len(gt_ids), len(pr_ids)))
    per_frame_pairs: dict[int, list[tuple[int, int]]] = {}
    for f in frames:
        g, p = gt.get(f, []), pred.get(f, [])
        if not g or not p:
            continue
        iou = iou_matrix(np.array([b for _, b in g]), np.array([b for _, b in p]))
        r, c = linear_sum_assignment(-iou)
        pairs = [(g[a][0], p[b][0]) for a, b in zip(r, c) if iou[a, b] >= iou_thresh]
        per_frame_pairs[f] = pairs
        for a, b in pairs:
            overlap[gi[a], pi[b]] += 1

    # Global identity assignment maximising IDTP.
    r, c = linear_sum_assignment(-overlap)
    idtp = float(overlap[r, c].sum())
    idfn, idfp = n_gt - idtp, n_pr - idtp
    idp = idtp / n_pr if n_pr else 0.0
    idr = idtp / n_gt if n_gt else 0.0
    idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) else 0.0

    # ID switches from frame-level matches.
    last: dict[int, int] = {}
    switches = 0
    for f in frames:
        for g_id, p_id in per_frame_pairs.get(f, []):
            if g_id in last and last[g_id] != p_id:
                switches += 1
            last[g_id] = p_id
    return {"idf1": float(idf1), "idp": float(idp), "idr": float(idr), "id_switches": int(switches),
            "gt_detections": n_gt, "pred_detections": n_pr}


# ---------- runtime / memory ----------
def peak_memory_mb() -> float:
    """Peak resident memory of this process in MB.

    Uses `resource` on Linux (reports KB) and macOS (reports bytes). `resource` does not exist on Windows,
    so there the peak working-set size comes from the Win32 API via ctypes. Returns NaN if neither is available.
    """
    import sys

    try:
        import resource
    except ImportError:
        return _windows_peak_memory_mb() if sys.platform == "win32" else float("nan")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024)


def _windows_peak_memory_mb() -> float:
    """Peak working set of this process in MB via GetProcessMemoryInfo (no extra dependencies)."""
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    get_info = getattr(kernel32, "K32GetProcessMemoryInfo", None) or ctypes.WinDLL("psapi").GetProcessMemoryInfo
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
    get_info.restype = wintypes.BOOL
    if not get_info(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return float("nan")
    return float(counters.PeakWorkingSetSize / (1024 * 1024))


def throughput(n_items: int, seconds: float) -> float:
    """Items per second; 0 when no time was recorded."""
    return float(n_items / seconds) if seconds > 0 else 0.0
