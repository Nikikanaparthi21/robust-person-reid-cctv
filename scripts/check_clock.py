"""Check the burned-in clock reader on a real video (and show where the clock jumps).

usage: python scripts/check_clock.py --video path/to/cam1.asf --every 5 --limit 600
Prints how many sampled frames were readable and every place the clock jumps
(clip boundaries in a motion-triggered recording).
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cctv_reid.clock_reader import read_clock  # noqa: E402


def hms(s):
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--every", type=float, default=5.0, help="sample every N video seconds")
ap.add_argument("--limit", type=float, default=None, help="only first N video seconds")
a = ap.parse_args()

cap = cv2.VideoCapture(a.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
total = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
end = min(total, a.limit) if a.limit else total
print(f"{a.video}: {total / 60:.1f} min of video, {fps:.1f} fps, frame {int(cap.get(3))}x{int(cap.get(4))}")
t, ok, bad, prev = 0.0, 0, 0, None
jumps = []
while t < end:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    good, frame = cap.read()
    if not good:
        break
    c = read_clock(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if c is None:
        bad += 1
    else:
        ok += 1
        if prev is not None and not (0 <= c - prev[1] - (t - prev[0]) <= 3):
            jumps.append((t, prev[1], c))
        prev = (t, c)
    t += a.every
print(f"readable {ok}/{ok + bad} samples")
print(f"{len(jumps)} clock jumps (video time: clock before -> after):")
for vt, b, c in jumps[:40]:
    print(f"  at video {hms(int(vt))}: {hms(b)} -> {hms(c)}")
