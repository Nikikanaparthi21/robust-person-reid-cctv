"""Soft spatio-temporal prior: a match between two cameras is more plausible when the time gap fits the walking time.

Layout = {(from_camera, to_camera): (min_seconds, max_seconds)}.  It is a *bonus* added to the appearance similarity,
never a filter: the same person can come back to a camera hours later, and the layout cannot say so.
"""

from __future__ import annotations

import csv

import numpy as np

# Rough values from the user's camera notes (CAM 2 -> CAM 1 ~13 s, CAM 1 -> CAM 3 ~6 s, CAM 2 -> CAM 3 ~19 s via CAM 1),
# widened to generous ranges. An ASSUMPTION used until the user supplies a layout file; walking is assumed symmetric.
DEFAULT_LAYOUT = {("CAM 2", "CAM 1"): (3, 40), ("CAM 1", "CAM 3"): (2, 30), ("CAM 2", "CAM 3"): (5, 60)}


def load_layout(path: str) -> dict:
    """Read data/examples/camera_layout_template.csv style file; rows with empty min/max are skipped."""
    out = {}
    for r in csv.DictReader(open(path)):
        if str(r.get("min_seconds", "")).strip() and str(r.get("max_seconds", "")).strip():
            out[(r["from_camera"].strip(), r["to_camera"].strip())] = (float(r["min_seconds"]), float(r["max_seconds"]))
    return out


def _range(layout, a, b):
    return layout.get((a, b)) or layout.get((b, a))        # symmetric unless both directions are given


def bonus_matrix(tracklets: list[dict], layout: dict, bonus: float = 0.1, slack: float = 5.0) -> np.ndarray:
    """(N, N) matrix: +bonus where two tracklets on different cameras are separated by a plausible walking time."""
    n = len(tracklets)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = tracklets[i], tracklets[j]
            if a["camera"] == b["camera"]:
                continue
            first, second = (a, b) if a["start"] <= b["start"] else (b, a)
            rng = _range(layout, first["camera"], second["camera"])
            if rng is None:
                continue
            gap = second["start"] - first["end"]
            if rng[0] - slack <= gap <= rng[1] + slack:
                out[i, j] = out[j, i] = bonus
    return out
