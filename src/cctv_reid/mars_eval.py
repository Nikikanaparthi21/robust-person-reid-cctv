"""Evaluate precomputed frame embeddings on MARS-formatted names.

Simplified protocol (NOT the official MARS query/gallery lists): per identity, camera-1
tracklets are queries and tracklets from the other cameras are the gallery, with pid 0
(distractors) kept in the gallery and pid -1 (junk) dropped. Clustering uses a deterministic
identity-level validation/test split to tune thresholds. Report it as "MARS-style protocol".
"""

from __future__ import annotations

import numpy as np

from . import clustering, embeddings, evaluation, io_utils


def evaluate(features: np.ndarray, names: list[str], seed: int = 255, max_ids: int = 100,
             max_frames: int = 16, quality: np.ndarray | None = None) -> list[dict]:
    rng = np.random.default_rng(seed)
    parsed = [io_utils.parse_mars_name(n) for n in names]
    pid = np.array([p[0] for p in parsed])
    cam = np.array([p[1] for p in parsed])
    keys = [f"{p[0]}_{p[1]}_{p[2]}" for p in parsed]
    keep = pid != -1
    features, pid, cam = features[keep], pid[keep], cam[keep]
    keys = [k for k, kp in zip(keys, keep) if kp]
    quality = None if quality is None else quality[keep]

    index: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        index.setdefault(k, []).append(i)
    for k, idx in index.items():  # cap frames per tracklet by even subsampling (keeps temporal spread)
        if len(idx) > max_frames:
            index[k] = [idx[j] for j in np.linspace(0, len(idx) - 1, max_frames).astype(int)]

    methods = ["frame", "mean", "attention"] + (["quality_filtered", "quality_weighted"] if quality is not None else [])
    real_ids = sorted({p for p in pid if p > 0})
    rng.shuffle(real_ids)
    ids = real_ids[:max_ids]
    half = len(ids) // 2
    val_ids, test_ids = set(ids[:half]), set(ids[half:])

    rows = []
    for method in methods:
        row = {"method": method}
        if method == "frame":
            q_idx = [int(rng.choice(v)) for k, v in index.items()
                     if int(k.split("_")[0]) in test_ids and cam[v[0]] == 1]
            g_idx = np.array([i for v in index.values() for i in v
                              if cam[i] != 1 and (pid[i] in test_ids or pid[i] == 0)])
            r = evaluation.retrieval_metrics(features[q_idx], pid[q_idx], cam[q_idx],
                                             features[g_idx], pid[g_idx], cam[g_idx])
            row.update({k: v for k, v in r.items() if k != "n_queries"})
            row["n_queries"] = r["n_queries"]
            for split_name, split in (("val", val_ids), ("test", test_ids)):
                sel = np.array([i for v in index.values() for i in v if pid[i] in split])
                row[f"_{split_name}"] = (features[sel], pid[sel])
            min_samples = 5
        else:
            tkeys, feats = embeddings.pool_tracklets(features, index, method, quality)
            tpid = np.array([int(k.split("_")[0]) for k in tkeys])
            tcam = np.array([int(k.split("_")[1]) for k in tkeys])
            q = np.isin(tpid, list(test_ids)) & (tcam == 1)
            g = (np.isin(tpid, list(test_ids)) | (tpid == 0)) & (tcam != 1)
            r = evaluation.retrieval_metrics(feats[q], tpid[q], tcam[q], feats[g], tpid[g], tcam[g])
            row.update({k: v for k, v in r.items() if k != "n_queries"})
            row["n_queries"] = r["n_queries"]
            for split_name, split in (("val", val_ids), ("test", test_ids)):
                sel = np.isin(tpid, list(split))
                row[f"_{split_name}"] = (feats[sel], tpid[sel])
            min_samples = 2
        (xv, yv), (xt, yt) = row.pop("_val"), row.pop("_test")
        for cm in ("dbscan", "agglomerative"):
            thr, _ = clustering.tune_threshold(xv, yv, cm, grid=np.linspace(0.1, 0.9, 9), min_samples=min_samples)
            labels = clustering.cluster(xt, cm, thr, min_samples)
            m = evaluation.cluster_metrics(yt, labels)
            for k in ("precision", "recall", "f1", "purity", "ari", "clusters_per_identity"):
                row[f"{cm}_{k}"] = m[k]
            row[f"{cm}_threshold"] = thr
        rows.append(row)
    return rows
