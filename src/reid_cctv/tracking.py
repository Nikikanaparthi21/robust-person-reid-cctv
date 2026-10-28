"""Deterministic IoU-based multi-object tracking baseline."""

from dataclasses import dataclass
from typing import Any

from .detection import Detection


def intersection_over_union(
    first: list[float],
    second: list[float],
) -> float:
    """Calculate IoU for boxes formatted as [x1, y1, x2, y2]."""
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)
    intersection = intersection_width * intersection_height

    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection

    if union == 0:
        return 0.0

    return intersection / union


@dataclass
class Track:
    """Internal state for one active track."""

    track_id: int
    detection: Detection
    missed_frames: int = 0


class IoUTracker:
    """Simple deterministic tracker suitable as a ByteTrack-equivalent fallback."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missing_frames: int = 15,
    ) -> None:
        if not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold must be between 0 and 1")
        if max_missing_frames < 0:
            raise ValueError("max_missing_frames must be non-negative")

        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        self._next_track_id = 1
        self._tracks: dict[int, Track] = {}

    def update(self, detections: list[Detection]) -> list[dict[str, Any]]:
        """Match detections to tracks and return manifest-ready records."""
        ordered_detections = sorted(
            detections,
            key=lambda detection: (-detection.confidence, detection.frame_number),
        )

        unmatched_track_ids = set(self._tracks)
        records: list[dict[str, Any]] = []

        for detection in ordered_detections:
            best_track_id: int | None = None
            best_iou = self.iou_threshold

            for track_id in sorted(unmatched_track_ids):
                candidate = self._tracks[track_id]
                score = intersection_over_union(
                    candidate.detection.bbox,
                    detection.bbox,
                )

                if score > best_iou:
                    best_iou = score
                    best_track_id = track_id

            if best_track_id is None:
                best_track_id = self._next_track_id
                self._next_track_id += 1
                self._tracks[best_track_id] = Track(
                    track_id=best_track_id,
                    detection=detection,
                )
            else:
                self._tracks[best_track_id].detection = detection
                self._tracks[best_track_id].missed_frames = 0
                unmatched_track_ids.remove(best_track_id)

            record = detection.to_dict()
            record["track_id"] = best_track_id
            record["iou_match_threshold"] = self.iou_threshold
            records.append(record)

        for track_id in unmatched_track_ids:
            self._tracks[track_id].missed_frames += 1

        expired_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if track.missed_frames > self.max_missing_frames
        ]

        for track_id in expired_ids:
            del self._tracks[track_id]

        return sorted(
            records,
            key=lambda record: (record["frame_number"], record["track_id"]),
        )