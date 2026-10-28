"""Detection data structures and validation helpers."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Detection:
    """One person detection in one video frame."""

    frame_number: int
    timestamp_seconds: float
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str = "person"

    def __post_init__(self) -> None:
        if self.frame_number < 0:
            raise ValueError("frame_number must be non-negative")

        if self.timestamp_seconds < 0:
            raise ValueError("timestamp_seconds must be non-negative")

        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")

        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bounding box must have positive width and height")

        if self.class_name != "person":
            raise ValueError("only person detections are supported")

    @property
    def bbox(self) -> list[float]:
        """Return the box as [x1, y1, x2, y2]."""
        return [self.x1, self.y1, self.x2, self.y2]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self) | {"bbox": self.bbox}