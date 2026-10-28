"""Manifests, seeds, hashes and MARS file-name parsing. No raw media is stored."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# MARS image names look like 0001C1T0001F001.jpg: person id, camera, tracklet, frame.
_MARS_RE = re.compile(r"^(-?\d{1,4})C(\d)T(\d{4})F(\d{3,4})")


@dataclass
class FrameRecord:
    path: str
    pid: int
    camera: int
    tracklet: int
    frame: int


@dataclass
class TrackletRecord:
    """One tracklet: the unit that temporal aggregation pools over."""

    track_id: str
    pid: int
    camera: int
    frame_indices: list[int] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)


def set_seed(seed: int) -> np.random.Generator:
    """Seed python and numpy; return a numpy Generator for explicit use."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def parse_mars_name(name: str) -> tuple[int, int, int, int]:
    """Parse a MARS file name into (pid, camera, tracklet, frame)."""
    m = _MARS_RE.match(Path(name).name)
    if m is None:
        raise ValueError(f"Not a MARS file name: {name!r}")
    pid, cam, trk, frm = (int(g) for g in m.groups())
    return pid, cam, trk, frm


def scan_mars_split(root: str | Path, split_dir: str) -> list[FrameRecord]:
    """List frame records under <root>/<split_dir>/<pid>/*.jpg without reading images."""
    base = Path(root) / split_dir
    records: list[FrameRecord] = []
    for img in sorted(base.glob("*/*.jpg")):
        pid, cam, trk, frm = parse_mars_name(img.name)
        records.append(FrameRecord(str(img), pid, cam, trk, frm))
    return records


def group_into_tracklets(frames: list[FrameRecord]) -> dict[str, list[int]]:
    """Map 'pid_camera_tracklet' to the indices of its frames, in frame order."""
    groups: dict[str, list[int]] = {}
    for i, f in enumerate(frames):
        groups.setdefault(f"{f.pid}_{f.camera}_{f.tracklet}", []).append(i)
    for idx in groups.values():
        idx.sort(key=lambda j: frames[j].frame)
    return groups


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if hasattr(o, "__dataclass_fields__"):
            return asdict(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=default)


def load_json(path: str | Path):
    with open(path) as fh:
        return json.load(fh)
