# Detection and Tracklet Manifest

## Purpose

This module provides a deterministic baseline for converting person detections into tracklet records for the MARS person re-identification dataset.

The repository does not contain raw CCTV footage, private videos, or model weights.

## Pipeline

1. Read person detections from the selected MARS subset.
2. Store each detection with its frame number, timestamp, bounding box, and confidence.
3. Match detections between consecutive frames using bounding-box intersection over union (IoU).
4. Assign a persistent `track_id` to overlapping detections.
5. Export the resulting records as JSONL and CSV manifests.

The current tracker is an IoU-based equivalent baseline. It is deterministic and provides the same required tracklet fields while avoiding an untracked external tracker implementation.

## Manifest schema

Each output record contains:

| Field | Type | Description |
|---|---|---|
| `track_id` | integer | Persistent identity assigned by the tracker |
| `frame_number` | integer | Zero-based frame number |
| `timestamp_seconds` | float | Timestamp derived from the sampling rate |
| `x1` | float | Left bounding-box coordinate |
| `y1` | float | Top bounding-box coordinate |
| `x2` | float | Right bounding-box coordinate |
| `y2` | float | Bottom bounding-box coordinate |
| `confidence` | float | Detection confidence from 0 to 1 |
| `class_name` | string | Detection class; currently `person` |
| `bbox` | list | Bounding box in `[x1, y1, x2, y2]` format |
| `iou_match_threshold` | float | IoU threshold used for matching |

## Configuration

The baseline values are:

- Dataset: MARS
- Split: test
- Sampling rate: 3 frames per second
- Image size: 256 x 128
- IoU matching threshold: 0.3
- Maximum missing frames: 15
- Random seed: 255

## Sample run

The privacy-safe sample output was generated with three synthetic detections:

```bash
PYTHONPATH=src python -m py_compile \
  src/reid_cctv/detection.py \
  src/reid_cctv/tracking.py \
  src/reid_cctv/io_utils.py