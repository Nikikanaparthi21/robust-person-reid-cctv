import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from reid_cctv.detection import Detection
from reid_cctv.tracking import IoUTracker, intersection_over_union


def test_intersection_over_union():
    first = [0, 0, 10, 10]
    second = [5, 5, 15, 15]

    assert round(intersection_over_union(first, second), 4) == 0.1429


def test_tracker_preserves_track_id_for_overlapping_boxes():
    tracker = IoUTracker(iou_threshold=0.3)

    first_frame = tracker.update(
        [
            Detection(
                frame_number=0,
                timestamp_seconds=0.0,
                x1=10,
                y1=10,
                x2=50,
                y2=100,
                confidence=0.95,
            )
        ]
    )

    second_frame = tracker.update(
        [
            Detection(
                frame_number=1,
                timestamp_seconds=1 / 3,
                x1=12,
                y1=11,
                x2=52,
                y2=101,
                confidence=0.94,
            )
        ]
    )

    assert first_frame[0]["track_id"] == 1
    assert second_frame[0]["track_id"] == 1