"""Stdlib-only MARS file-name parsing and deterministic subset selection."""

from __future__ import annotations

import random
import re
from pathlib import Path

# MARS image names look like 0001C1T0001F001.jpg: person id, camera, tracklet, frame.
# Junk images live in folder "00-1" with names like 00-1C1T0001F001.jpg (pid -1); distractors use 0000.
_MARS_RE = re.compile(r"^(00-1|-?\d{1,4})C(\d)T(\d{4})F(\d{3,4})")


def parse_mars_name(name: str) -> tuple[int, int, int, int]:
    """Parse a MARS file name into (pid, camera, tracklet, frame)."""
    m = _MARS_RE.match(Path(name).name)
    if m is None:
        raise ValueError(f"Not a MARS file name: {name!r}")
    pid_s, cam, trk, frm = m.groups()
    pid = -1 if pid_s == "00-1" else int(pid_s)
    return pid, int(cam), int(trk), int(frm)


def select_frames(names: list[str], seed: int = 255, max_ids: int = 100, max_frames: int = 16,
                  max_distractor_tracklets: int = 150) -> list[str]:
    """Pick a reproducible subset: `max_ids` identities, up to `max_frames` evenly spaced frames per
    tracklet, plus up to `max_distractor_tracklets` distractor (pid 0) tracklets. Junk (pid -1) is dropped.
    Returns the chosen entries of `names`, sorted.
    """
    tracklets: dict[tuple[int, int, int], list[tuple[int, str]]] = {}
    for n in names:
        pid, cam, trk, frm = parse_mars_name(n)
        if pid == -1:
            continue
        tracklets.setdefault((pid, cam, trk), []).append((frm, n))

    rng = random.Random(seed)
    real_ids = sorted({k[0] for k in tracklets if k[0] > 0})
    rng.shuffle(real_ids)
    keep_ids = set(real_ids[:max_ids])
    distractors = sorted(k for k in tracklets if k[0] == 0)
    rng.shuffle(distractors)
    keep_keys = {k for k in tracklets if k[0] in keep_ids} | set(distractors[:max_distractor_tracklets])

    chosen: list[str] = []
    for key in sorted(keep_keys):
        frames = [n for _, n in sorted(tracklets[key])]
        if len(frames) > max_frames:
            step = (len(frames) - 1) / (max_frames - 1)
            frames = [frames[round(i * step)] for i in range(max_frames)]
        chosen += frames
    return sorted(chosen)
