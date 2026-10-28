"""Simplified ByteTrack-style tracker: two-stage IoU association with low-score recovery.

This is a dependency-light re-implementation of the ByteTrack idea (Zhang et al., 2022):
high-score detections are matched to tracks first, then remaining tracks are matched to
low-score detections so briefly occluded people are not lost. It uses constant-velocity
box prediction instead of a Kalman filter. It is a baseline for tracklet generation, not a
replica of the official implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detection import Detection


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) xyxy boxes."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    hits: list[Detection] = field(default_factory=list)
    missed: int = 0

    def predict(self) -> np.ndarray:
        return self.box + self.velocity

    def update(self, det: Detection) -> None:
        new_box = det.box
        self.velocity = 0.5 * self.velocity + 0.5 * (new_box - self.box)
        self.box = new_box
        self.hits.append(det)
        self.missed = 0


def _match(tracks: list[Track], dets: list[Detection], iou_thresh: float):
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))
    iou = iou_matrix(np.stack([t.predict() for t in tracks]), np.stack([d.box for d in dets]))
    rows, cols = linear_sum_assignment(-iou)
    matches, used_t, used_d = [], set(), set()
    for r, c in zip(rows, cols):
        if iou[r, c] >= iou_thresh:
            matches.append((r, c))
            used_t.add(r)
            used_d.add(c)
    return (
        matches,
        [i for i in range(len(tracks)) if i not in used_t],
        [j for j in range(len(dets)) if j not in used_d],
    )


class ByteTrackLite:
    def __init__(self, high_thresh: float = 0.5, low_thresh: float = 0.1, iou_thresh: float = 0.3,
                 max_missed: int = 30, new_track_thresh: float = 0.6, min_hits: int = 3):
        self.high, self.low = high_thresh, low_thresh
        self.iou_thresh, self.max_missed = iou_thresh, max_missed
        self.new_thresh, self.min_hits = new_track_thresh, min_hits
        self._next_id = 1
        self.active: list[Track] = []
        self.finished: list[Track] = []

    def step(self, detections: list[Detection]) -> None:
        high = [d for d in detections if d.score >= self.high]
        low = [d for d in detections if self.low <= d.score < self.high]

        m1, un_t, un_h = _match(self.active, high, self.iou_thresh)
        for ti, di in m1:
            self.active[ti].update(high[di])

        rem_tracks = [self.active[i] for i in un_t]
        m2, un_t2, _ = _match(rem_tracks, low, self.iou_thresh)
        for ti, di in m2:
            rem_tracks[ti].update(low[di])
        lost = {id(rem_tracks[i]) for i in un_t2}

        for t in self.active:
            if id(t) in lost:
                t.missed += 1
        survivors = []
        for t in self.active:
            (self.finished if t.missed > self.max_missed else survivors).append(t)
        self.active = survivors

        for j in un_h:
            d = high[j]
            if d.score >= self.new_thresh:
                t = Track(self._next_id, d.box.copy())
                t.hits.append(d)
                self._next_id += 1
                self.active.append(t)

    def run(self, frames: list[list[Detection]]) -> dict[int, list[Detection]]:
        for dets in frames:
            self.step(dets)
        tracks = self.finished + self.active
        return {t.track_id: t.hits for t in tracks if len(t.hits) >= self.min_hits}
