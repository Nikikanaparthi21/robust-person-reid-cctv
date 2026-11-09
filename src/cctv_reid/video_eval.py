"""Score the video pipeline against a hand-made ground-truth log (who was in which camera, when).

Ground truth CSV columns: person, camera, start, end. `camera` must match the --video camera names; `start`/`end` are either
clock times ("2026-11-06 09:01:15" / ISO) or HH:MM:SS from the start of that camera's video, the same convention as
tracklets.csv. A tracklet takes the label of the person whose interval overlaps at least `min_overlap` of the tracklet;
tracklets with no such person are "unlabeled" and act as distractors in the gallery.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from . import clustering, evaluation


def parse_time(s: str) -> float:
    """ISO / 'YYYY-MM-DD HH:MM:SS' -> epoch seconds (naive local); 'HH:MM:SS' or plain seconds -> seconds."""
    s = str(s).strip()
    try:
        return float(s)
    except ValueError:
        pass
    if ":" in s and "-" not in s[:5]:
        parts = [float(x) for x in s.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return datetime.fromisoformat(s).timestamp()


AMBIGUOUS = "?"


def assign_labels(tracklets: list[dict], truth: list[dict], min_overlap: float = 0.5) -> list[str | None]:
    """tracklets: dicts with camera, start, end (seconds, same convention as truth).

    Returns the person, None (nobody in the log: a distractor), or AMBIGUOUS when two or more different people overlap
    the tracklet by at least `min_overlap` (time alone cannot say which one it is; such tracklets are left out of scoring).
    """
    out = []
    for t in tracklets:
        dur = max(1e-6, t["end"] - t["start"])
        ov: dict[str, float] = {}
        for g in truth:
            if g["camera"] == t["camera"]:
                o = max(0.0, min(t["end"], g["end"]) - max(t["start"], g["start"])) / dur
                ov[g["person"]] = max(ov.get(g["person"], 0.0), o)
        good = [p for p, o in ov.items() if o >= min_overlap]
        out.append(good[0] if len(good) == 1 else (AMBIGUOUS if len(good) > 1 else None))
    return out


def evaluate_video(tracklets: list[dict], feats: np.ndarray, truth: list[dict], threshold: float = 0.3,
                   min_overlap: float = 0.5, layout: dict | None = None) -> dict:
    all_labels = assign_labels(tracklets, truth, min_overlap)
    keep = [i for i, lab in enumerate(all_labels) if lab != AMBIGUOUS]
    n_ambiguous = len(all_labels) - len(keep)
    tracklets = [tracklets[i] for i in keep]
    feats = np.asarray(feats)[keep]
    labels = [all_labels[i] for i in keep]
    cams = np.array([t["camera_index"] for t in tracklets])
    people = sorted({p for p in labels if p})
    pid = np.array([people.index(p) + 1 if p else -(i + 1) for i, p in enumerate(labels)])  # unlabeled: unique negatives
    cams_per = {}
    for p, c in zip(pid, cams):
        if p > 0:
            cams_per.setdefault(int(p), set()).add(int(c))
    q = np.array([i for i in range(len(pid)) if pid[i] > 0 and len(cams_per[int(pid[i])]) > 1])
    out = {"n_tracklets": len(tracklets), "n_labeled": int((pid > 0).sum()), "n_people": len(people),
           "n_unlabeled": int((pid < 0).sum()), "n_ambiguous": n_ambiguous, "n_queries": int(len(q))}
    if len(q):
        r = evaluation.retrieval_metrics(feats[q], pid[q], cams[q], feats, pid, cams)
        out.update({"rank1": r["rank1"], "rank5": r["rank5"], "mAP": r["mAP"]})
    lab = pid > 0
    if lab.sum() > 1:
        pred = clustering.cluster(np.asarray(feats)[lab], "agglomerative", threshold)
        m = evaluation.cluster_metrics(pid[lab], pred)
        out.update({"cluster_precision": m["precision"], "cluster_recall": m["recall"], "cluster_f1": m["f1"],
                    "cluster_purity": m["purity"], "clusters_per_person": m["clusters_per_identity"]})
    out["labels"] = all_labels
    if len(q) and layout:
        out["time_prior"] = time_prior_comparison(tracklets, feats, pid, cams, q, layout)
    if len(q):
        out["details"] = query_details(tracklets, feats, pid, cams, q, labels)
    return out


def _sim(feats):
    f = np.asarray(feats, np.float64)
    f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
    return np.einsum("id,jd->ij", f, f)


def rank_metrics(sim, pid, cams, q):
    """Cross-camera rank-1/5 and mAP from a similarity matrix (same-camera candidates skipped)."""
    hits1 = hits5 = 0
    aps = []
    for i in q:
        order = [j for j in np.argsort(-sim[i], kind="stable") if j != i and cams[j] != cams[i]]
        match = np.array([pid[j] == pid[i] for j in order])
        if not match.any():
            continue
        hits1 += bool(match[:1].any())
        hits5 += bool(match[:5].any())
        prec = np.cumsum(match) / (np.arange(len(match)) + 1)
        aps.append(float((prec * match).sum() / match.sum()))
    n = max(len(aps), 1)
    return {"rank1": hits1 / n, "rank5": hits5 / n, "mAP": float(np.mean(aps)) if aps else float("nan")}


def time_prior_comparison(tracklets, feats, pid, cams, q, layout, bonuses=(0.0, 0.05, 0.1, 0.2)):
    """Same queries, ranked by similarity + bonus for plausible walking times (0.0 = plain appearance)."""
    from . import time_prior

    sim = _sim(feats)
    return {str(b): rank_metrics(sim + time_prior.bonus_matrix(tracklets, layout, b), pid, cams, q) for b in bonuses}


def query_details(tracklets, feats, pid, cams, q, labels):
    """For each query (a labeled person seen on 2+ cameras): where did the first correct match rank, and what was top-1?

    Same-camera candidates are skipped, as in the metrics, so this is purely cross-camera retrieval.
    """
    feats = np.asarray(feats, float)
    rows = []
    for i in q:
        sim = _sim(feats)[i]
        order = [j for j in np.argsort(-sim) if j != i and cams[j] != cams[i]]
        correct = [r for r, j in enumerate(order, 1) if pid[j] == pid[i]]
        top = order[0] if order else None
        rows.append({"query": tracklets[i].get("id", i), "person": labels[i], "n_correct_other_cams": len(correct),
                     "first_correct_rank": correct[0] if correct else None,
                     "top1": tracklets[top].get("id", int(top)) if top is not None else None,
                     "top1_label": (labels[top] or "(unlabeled)") if top is not None else None,
                     "top1_sim": round(float(sim[top]), 3) if top is not None else None})
    return rows
