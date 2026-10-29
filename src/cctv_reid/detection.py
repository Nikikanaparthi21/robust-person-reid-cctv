"""Normalized person detections. Detector wrappers feed this format into tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    frame: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    @property
    def box(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=float)


def from_array(frame: int, boxes: np.ndarray, scores: np.ndarray, min_score: float = 0.0) -> list[Detection]:
    """Build detections from an (N, 4) xyxy array and (N,) scores, dropping scores below min_score."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(boxes) != len(scores):
        raise ValueError("boxes and scores must have the same length")
    return [Detection(frame, *map(float, b), float(s)) for b, s in zip(boxes, scores) if s >= min_score]
