"""Data for the review viewer (demo/index.html): per-query ranked matches and identity clusters.

Torch-free. Input is frame embeddings + MARS-style file names. Output is a JSON-ready dict.
Matches are *leads for a human reviewer*, never confirmed identities. MARS has no wall-clock time,
so `time_s` is None unless a mapping from tracklet key to seconds is supplied (video manifests have it).
"""

from __future__ import annotations

import numpy as np

from . import clustering, embeddings, io_utils

DISCLAIMER = ("Similarity scores are leads for human review. They do not identify or confirm anyone. "
              "Use only public benchmark data or footage you are authorized to review.")


def _index_tracklets(names, max_frames):
    parsed = [io_utils.parse_mars_name(n) for n in names]
    index: dict[str, list[int]] = {}
    for i, (pid, cam, trk, _) in enumerate(parsed):
        if pid != -1:
            index.setdefault(f"{pid}_{cam}_{trk}", []).append(i)
    for k, idx in index.items():
        if len(idx) > max_frames:
            index[k] = [idx[j] for j in np.linspace(0, len(idx) - 1, max_frames).astype(int)]
    return parsed, index


def _thumb_frames(names, idx, n=3):
    pick = np.linspace(0, len(idx) - 1, min(n, len(idx))).round().astype(int)
    return [names[idx[j]] for j in pick]


def build(features: np.ndarray, names: list[str], n_queries: int = 12, top_k: int = 12, threshold: float = 0.3,
          seed: int = 255, max_frames: int = 16, times: dict[str, float] | None = None,
          model_note: str = "") -> dict:
    parsed, index = _index_tracklets(names, max_frames)
    keys, feats = embeddings.pool_tracklets(features, index, "mean")
    feats = embeddings.l2_normalize(feats.astype(np.float64))
    pid = np.array([int(k.split("_")[0]) for k in keys])
    cam = np.array([int(k.split("_")[1]) for k in keys])
    trk = np.array([int(k.split("_")[2]) for k in keys])
    n_frames = np.array([len(index[k]) for k in keys])
    labels = clustering.cluster(feats, "agglomerative", threshold)

    def card(i):
        return {"key": keys[i], "camera": int(cam[i]), "tracklet": int(trk[i]), "n_frames": int(n_frames[i]),
                "time_s": None if times is None else times.get(keys[i]),
                "thumbs": _thumb_frames(names, index[keys[i]])}

    # Queries: camera-1 tracklets of real identities that also appear on another camera.
    cams_of = {}
    for p, c in zip(pid, cam):
        cams_of.setdefault(int(p), set()).add(int(c))
    cand = [i for i in range(len(keys)) if pid[i] > 0 and cam[i] == 1 and len(cams_of[int(pid[i])]) > 1]
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(cand, size=min(n_queries, len(cand)), replace=False)) if cand else []

    sims_all = np.einsum("ik,jk->ij", feats[chosen], feats) if len(chosen) else np.zeros((0, len(keys)))
    queries = []
    for row, qi in enumerate(chosen):
        sims = sims_all[row].copy()
        sims[qi] = -np.inf
        sims[(pid == pid[qi]) & (cam == cam[qi])] = -np.inf  # same-camera tracklets of the same person are trivial
        order = np.argsort(-sims)[:top_k]
        results = []
        for rank, j in enumerate(order, 1):
            if not np.isfinite(sims[j]):
                break
            results.append({**card(j), "rank": rank, "similarity": round(float(sims[j]), 4),
                            "cluster": int(labels[j]), "same_cluster_as_query": bool(labels[j] == labels[qi]),
                            "truth": "same" if pid[j] == pid[qi] else ("distractor" if pid[j] == 0 else "different")})
        queries.append({**card(qi), "id": f"q{row + 1}", "cluster": int(labels[qi]), "results": results,
                        "n_true_matches": int(((pid == pid[qi]) & (cam != cam[qi])).sum())})

    clusters = []
    for c in np.unique(labels):
        members = np.where(labels == c)[0]
        if len(members) < 2 or len(set(cam[members])) < 2:
            continue  # only show clusters that link at least two cameras
        centroid = embeddings.l2_normalize(feats[members].mean(axis=0))
        clusters.append({"id": int(c), "size": int(len(members)), "cameras": sorted({int(x) for x in cam[members]}),
                         "purity_hint": None,
                         "members": [{**card(j), "similarity_to_centroid": round(float(feats[j] @ centroid), 4),
                                      "truth_pid": int(pid[j])}
                                     for j in members[np.argsort(-(feats[members] @ centroid))]]})
    clusters.sort(key=lambda c: -c["size"])
    return {"meta": {"disclaimer": DISCLAIMER, "model": model_note, "threshold": threshold,
                     "n_tracklets": len(keys), "has_time": times is not None,
                     "time_note": "" if times is not None else "MARS has no wall-clock time; showing camera and tracklet."},
            "queries": queries, "clusters": clusters[:40]}
