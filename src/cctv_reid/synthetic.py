"""Synthetic frame-embedding benchmark used ONLY to validate pipeline logic.

Nothing here measures real CCTV behaviour. The generator encodes explicit assumptions:
every frame is a noisy view of a per-identity prototype, a fraction of frames is corrupted
(occlusion/blur) by mixing in another identity, and each camera adds a shared offset. Results
show how the pooling and clustering code behave under those assumptions, and must not be
reported as evidence about real data. Real evidence comes from MARS features (see
scripts/evaluate_features.py).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import clustering, embeddings, evaluation


@dataclass(frozen=True)
class SynthConfig:
    n_ids: int = 30
    dim: int = 128
    n_cams: int = 3
    tracklets_per_cam: int = 2
    min_frames: int = 8
    max_frames: int = 14
    frame_noise: float = 0.9      # per-frame noise norm relative to a unit prototype
    p_bad: float = 0.1            # fraction of corrupted frames (occlusion / blur)
    bad_mix: float = 0.55         # weight of the other identity inside a corrupted frame
    cam_shift: float = 0.5
    track_shift: float = 0.4      # per-tracklet viewpoint offset shared by its frames
    quality_noise: float = 0.25   # noise on the observed quality score
    keep_every: int = 1           # sampling-rate reduction (keep every n-th frame)


CONDITIONS: dict[str, dict] = {
    "clean": {},
    "blur": {"frame_noise": 1.3},
    "low_resolution": {"frame_noise": 1.6},
    "occlusion": {"p_bad": 0.4},
    "dark_and_occluded": {"frame_noise": 1.3, "p_bad": 0.3},
    "low_sampling_rate": {"keep_every": 4},
}

# Stress sweep, fixed before looking at any results: corrupted-frame fraction 0.2-0.8 with a larger
# shared per-tracklet viewpoint offset and heavier corruption. Used to see where pooling methods separate.
STRESS_CONDITIONS: dict[str, dict] = {
    f"stress_p_bad_{int(p * 100)}": {"p_bad": p, "track_shift": 1.0, "bad_mix": 0.7, "frame_noise": 1.3}
    for p in (0.2, 0.4, 0.6, 0.8)
}

METHODS = ["frame", "mean", "quality_filtered", "quality_weighted", "attention"]


def make_dataset(cfg: SynthConfig, rng: np.random.Generator) -> dict:
    d = cfg.dim
    protos = embeddings.l2_normalize(rng.normal(size=(cfg.n_ids, d)))
    cam_vecs = embeddings.l2_normalize(rng.normal(size=(cfg.n_cams, d)))
    frames, ids, cams, trk_keys, qual_true = [], [], [], [], []
    for pid in range(cfg.n_ids):
        for cam in range(cfg.n_cams):
            for t in range(cfg.tracklets_per_cam):
                n = int(rng.integers(cfg.min_frames, cfg.max_frames + 1))
                n = len(range(0, n, max(1, cfg.keep_every)))
                trk_vec = embeddings.l2_normalize(rng.normal(size=d))
                base = protos[pid] + cfg.cam_shift * cam_vecs[cam] + cfg.track_shift * trk_vec
                for _ in range(n):
                    noise = rng.normal(scale=cfg.frame_noise / np.sqrt(d), size=d)
                    bad = rng.random() < cfg.p_bad
                    if bad:
                        other = protos[int(rng.choice([i for i in range(cfg.n_ids) if i != pid]))]
                        f = (1 - cfg.bad_mix) * base + cfg.bad_mix * other + 1.5 * noise
                    else:
                        f = base + noise
                    frames.append(f)
                    ids.append(pid)
                    cams.append(cam)
                    trk_keys.append(f"{pid}_{cam}_{t}")
                    qual_true.append(0.2 if bad else 1.0)
    frames = np.stack(frames)
    quality = np.asarray(qual_true) + rng.normal(scale=cfg.quality_noise, size=len(qual_true))
    ids, cams = np.asarray(ids), np.asarray(cams)
    index: dict[str, list[int]] = {}
    for i, k in enumerate(trk_keys):
        index.setdefault(k, []).append(i)
    return {"frames": frames, "ids": ids, "cams": cams, "quality": quality, "index": index}


def _tracklet_arrays(ds: dict, method: str):
    keys, feats = embeddings.pool_tracklets(ds["frames"], ds["index"], method, ds["quality"])
    first = [ds["index"][k][0] for k in keys]
    return keys, feats, ds["ids"][first], ds["cams"][first]


def _frame_labels_from_tracklets(ds: dict, keys: list[str], track_labels: np.ndarray) -> np.ndarray:
    labels = np.empty(len(ds["ids"]), dtype=int)
    for k, lab in zip(keys, track_labels):
        labels[ds["index"][k]] = lab
    return labels


def evaluate_method(test: dict, val: dict, method: str, rng: np.random.Generator,
                    cluster_methods=("dbscan", "agglomerative")) -> dict:
    """Retrieval + clustering metrics for one representation on one test set."""
    row: dict = {"method": method}
    times: dict = {}
    if method == "frame":
        # Single random frame per cam-0 tracklet as query, all frames of other cameras as gallery.
        q_idx = [int(rng.choice(v)) for k, v in test["index"].items() if test["cams"][v[0]] == 0]
        g_mask = test["cams"] != 0
        with evaluation.timer(times, "retrieval_s"):
            r = evaluation.retrieval_metrics(
                test["frames"][q_idx], test["ids"][q_idx], test["cams"][q_idx],
                test["frames"][g_mask], test["ids"][g_mask], test["cams"][g_mask])
        min_samples = 5
        items_test, ids_test = test["frames"], test["ids"]
        items_val, ids_val = val["frames"], val["ids"]
    else:
        keys, feats, ids, cams = _tracklet_arrays(test, method)
        vkeys, vfeats, vids, _ = _tracklet_arrays(val, method)
        q, g = cams == 0, cams != 0
        with evaluation.timer(times, "retrieval_s"):
            r = evaluation.retrieval_metrics(feats[q], ids[q], cams[q], feats[g], ids[g], cams[g])
        min_samples = 2
        items_test, ids_test = feats, ids
        items_val, ids_val = vfeats, vids
    row.update({k: v for k, v in r.items() if k != "n_queries"})

    for cm in cluster_methods:
        thr, _ = clustering.tune_threshold(items_val, ids_val, cm, grid=np.linspace(0.1, 0.9, 9),
                                           min_samples=min_samples)
        with evaluation.timer(times, f"{cm}_s"):
            labels = clustering.cluster(items_test, cm, thr, min_samples)
        if method != "frame":
            labels = _frame_labels_from_tracklets(test, keys, labels)
        m = evaluation.cluster_metrics(test["ids"], labels)
        for k in ("precision", "recall", "f1", "purity", "ari", "clusters_per_identity", "merged_cluster_fraction"):
            row[f"{cm}_{k}"] = m[k]
        row[f"{cm}_threshold"] = thr
    row.update({k: round(v, 4) for k, v in times.items()})
    return row


def run_all(seeds: list[int], base: SynthConfig | None = None, conditions: dict | None = None) -> list[dict]:
    base = base or SynthConfig()
    conditions = conditions or CONDITIONS
    rows = []
    for seed in seeds:
        for cname, override in conditions.items():
            cfg = replace(base, **override)
            test = make_dataset(cfg, np.random.default_rng(seed))
            val = make_dataset(cfg, np.random.default_rng(10_000 + seed))  # separate identities
            rng = np.random.default_rng(20_000 + seed)
            for method in METHODS:
                row = evaluate_method(test, val, method, rng)
                row.update({"seed": seed, "condition": cname})
                rows.append(row)
    return rows
