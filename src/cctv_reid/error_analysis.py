"""Error taxonomy for tracklet clustering and retrieval (issue #25).

Uses the same identity split, tracklet capping and threshold tuning as ``mars_eval`` so the errors
describe exactly the configuration reported in the results table. Public-benchmark names only.

Categories
- false_merge: a cluster contains tracklets of more than one identity. Sub-typed as
  ``distractor_merge`` (a pid-0 distractor tracklet joined a real identity) or ``identity_merge``.
- false_split: one identity is spread across several clusters; sub-typed ``cross_camera_split``
  when the pieces are separated by camera, otherwise ``within_camera_split``.
- retrieval_miss: rank-1 result is wrong; ``distractor_top1`` or ``wrong_identity_top1``.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from . import clustering, embeddings, evaluation, io_utils


def analyze(features: np.ndarray, names: list[str], method: str = "mean", cluster_method: str = "agglomerative",
            seed: int = 255, max_ids: int = 100, max_frames: int = 16, quality=None) -> dict:
    rng = np.random.default_rng(seed)
    parsed = [io_utils.parse_mars_name(n) for n in names]
    pid = np.array([p[0] for p in parsed])
    keep = pid != -1
    features, pid = features[keep], pid[keep]
    quality = None if quality is None else quality[keep]
    keys = [f"{p[0]}_{p[1]}_{p[2]}" for p, k in zip(parsed, keep) if k]
    index: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        index.setdefault(k, []).append(i)
    for k, idx in index.items():
        if len(idx) > max_frames:
            index[k] = [idx[j] for j in np.linspace(0, len(idx) - 1, max_frames).astype(int)]

    real_ids = sorted({p for p in pid if p > 0})
    rng.shuffle(real_ids)
    ids = real_ids[:max_ids]
    val_ids, test_ids = set(ids[: len(ids) // 2]), set(ids[len(ids) // 2:])

    tkeys, feats = embeddings.pool_tracklets(features, index, method, quality)
    tpid = np.array([int(k.split("_")[0]) for k in tkeys])
    tcam = np.array([int(k.split("_")[1]) for k in tkeys])
    n_frames = np.array([len(index[k]) for k in tkeys])
    ms = 2

    v = np.isin(tpid, list(val_ids))
    thr, _ = clustering.tune_threshold(feats[v], tpid[v], cluster_method, grid=np.linspace(0.1, 0.9, 9),
                                       min_samples=ms, counts=n_frames[v])
    t = np.isin(tpid, list(test_ids) + [0])  # distractors present so false merges with them are visible
    labels = clustering.cluster(feats[t], cluster_method, thr, ms)
    t_keys, t_pid, t_cam = [k for k, m in zip(tkeys, t) if m], tpid[t], tcam[t]

    merges, splits = [], []
    for c in sorted(set(labels)):
        members = np.where(labels == c)[0]
        pids = set(t_pid[members])
        if len(pids) > 1:
            kind = "distractor_merge" if 0 in pids else "identity_merge"
            merges.append({"cluster": int(c), "type": kind, "identities": sorted(int(p) for p in pids),
                           "tracklets": [t_keys[i] for i in members][:8]})
    for p in sorted(set(t_pid) - {0}):
        members = np.where(t_pid == p)[0]
        cl = set(labels[members])
        if len(cl) > 1:
            by_cam = {}
            for i in members:
                by_cam.setdefault(int(labels[i]), set()).add(int(t_cam[i]))
            cross = all(len(cams) == 1 for cams in by_cam.values()) and len({next(iter(c)) for c in by_cam.values()}) > 1
            splits.append({"identity": int(p), "type": "cross_camera_split" if cross else "within_camera_split",
                           "n_clusters": len(cl), "tracklets": [t_keys[i] for i in members][:8]})

    # Retrieval: camera-1 queries against other cameras (test ids + distractors).
    q = np.isin(tpid, list(test_ids)) & (tcam == 1)
    g = (np.isin(tpid, list(test_ids)) | (tpid == 0)) & (tcam != 1)
    qn, gn = embeddings.l2_normalize(feats[q]), embeddings.l2_normalize(feats[g])
    qk, gk = [k for k, m in zip(tkeys, q) if m], [k for k, m in zip(tkeys, g) if m]
    sims = qn @ gn.T
    misses = []
    for i in range(len(qk)):
        same_cam_same_id = (tpid[g] == tpid[q][i]) & (tcam[g] == tcam[q][i])
        s = np.where(same_cam_same_id, -np.inf, sims[i])
        top = int(np.argmax(s))
        if tpid[g][top] != tpid[q][i]:
            misses.append({"query": qk[i], "top1": gk[top], "type": "distractor_top1" if tpid[g][top] == 0
                           else "wrong_identity_top1", "similarity": float(s[top])})

    counts = Counter([m["type"] for m in merges] + [s["type"] for s in splits] + [m["type"] for m in misses])
    return {"pooling": method, "cluster_method": cluster_method, "threshold": float(thr),
            "n_test_identities": len(test_ids), "n_queries": len(qk), "summary": dict(counts),
            "false_merges": merges, "false_splits": splits, "retrieval_misses": misses}


def to_markdown(res: dict, limit: int = 10) -> str:
    s = res["summary"]
    lines = [f"# Error analysis: {res['pooling']} pooling + {res['cluster_method']} (threshold {res['threshold']:.2f})", "",
             f"Test identities: {res['n_test_identities']}; retrieval queries: {res['n_queries']}.", "",
             "| Error type | Count |", "|---|---|"]
    for k in ("identity_merge", "distractor_merge", "cross_camera_split", "within_camera_split",
              "wrong_identity_top1", "distractor_top1"):
        lines.append(f"| {k} | {s.get(k, 0)} |")
    for title, key, fmt in (("False merges", "false_merges", lambda e: f"cluster {e['cluster']} ({e['type']}): ids {e['identities']}; e.g. {e['tracklets'][:3]}"),
                            ("False splits", "false_splits", lambda e: f"id {e['identity']} ({e['type']}): {e['n_clusters']} clusters; e.g. {e['tracklets'][:3]}"),
                            ("Retrieval misses", "retrieval_misses", lambda e: f"{e['query']} -> {e['top1']} ({e['type']}, sim {e['similarity']:.2f})")):
        lines += ["", f"## {title} (first {limit})"] + [f"- {fmt(e)}" for e in res[key][:limit]]
    return "\n".join(lines) + "\n"
