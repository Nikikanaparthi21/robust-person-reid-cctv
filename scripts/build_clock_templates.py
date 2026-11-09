"""Rebuild src/cctv_reid/clock_templates.npz from labelled screenshots.

usage: python scripts/build_clock_templates.py shot1.png=060005 shot2.png=073448 ...
Each screenshot must show the date 2026-10-29 and the given HH MM SS.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cctv_reid import clock_reader as C  # noqa: E402

acc = {d: [] for d in range(10)}
for arg in sys.argv[1:]:
    path, label = arg.rsplit("=", 1)
    fr = np.array(Image.open(path).convert("RGB"))
    for d, f in zip(label, C.cell_features(fr, C.TIME_CELLS, C.TIME_X0)):
        if f.any():
            acc[int(d)].append(f)
    for d, f in zip("20261029", C.cell_features(fr, C.DATE_CELLS, C.DATE_X0, C.DATE_PITCH)):
        if f.any():
            acc[int(d)].append(f)
missing = [d for d in acc if not acc[d]]
assert not missing, f"no sample for digits {missing}"
np.savez(C.TEMPLATES, digits=np.arange(10),
         feats=np.stack([np.mean(acc[d], 0) for d in range(10)]))
print({d: len(v) for d, v in acc.items()})
