"""Privacy-safe manifest writing and validation utilities."""

import csv
import json
from pathlib import Path
from typing import Any, Iterable


MANIFEST_FIELDS = [
    "track_id",
    "frame_number",
    "timestamp_seconds",
    "x1",
    "y1",
    "x2",
    "y2",
    "confidence",
    "class_name",
]


def validate_manifest_record(record: dict[str, Any]) -> None:
    """Validate one tracklet manifest record."""
    missing_fields = [
        field for field in MANIFEST_FIELDS if field not in record
    ]

    if missing_fields:
        raise ValueError(f"Manifest record is missing: {missing_fields}")

    if record["track_id"] < 1:
        raise ValueError("track_id must be positive")

    if record["frame_number"] < 0:
        raise ValueError("frame_number must be non-negative")

    if record["timestamp_seconds"] < 0:
        raise ValueError("timestamp_seconds must be non-negative")

    if not 0 <= record["confidence"] <= 1:
        raise ValueError("confidence must be between 0 and 1")

    if record["x2"] <= record["x1"] or record["y2"] <= record["y1"]:
        raise ValueError("bounding box must have positive dimensions")


def write_manifest_jsonl(
    records: Iterable[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Write one manifest record per JSON line."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            validate_manifest_record(record)
            output_file.write(json.dumps(record, sort_keys=True) + "\n")


def write_manifest_csv(
    records: Iterable[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Write manifest records to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=MANIFEST_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()

        for record in records:
            validate_manifest_record(record)
            writer.writerow(record)