"""Read the burned-in CCTV clock (``YYYY-MM-DD HH:MM:SS AM``) from a video frame.

The recorder draws the clock in a fixed font at a fixed place (top right), so each
digit is matched against stored digit templates (``clock_templates.npz``) instead
of using general OCR, which fails on this font.  Positions are fractions of the
frame, so they hold at any resolution.  Calibrated on 4 screenshots of the
user's three home cameras (all use the same layout).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

REF_W, REF_H = 1917.0, 1036.0           # screenshot the cells were measured on
TIME_PITCH = 32.0 / REF_W                # distance between characters (clock)
DATE_PITCH = 32.25 / REF_W               # ... (date; slightly different)
GLYPH_W = 28.0 / REF_W
TIME_X0 = 1421.0 / REF_W                 # first digit of HH
DATE_X0 = 1066.0 / REF_W                 # first digit of the year
Y0, Y1 = 0.012, 0.115                   # generous band: window crop differs between players
TIME_CELLS = [0, 1, 3, 4, 6, 7]          # char slots of H H : M M : S S
DATE_CELLS = [0, 1, 2, 3, 5, 6, 8, 9]    # slots of 2026-10-29 digits
FEAT = (12, 20)                          # w, h of the feature image
TEMPLATES = Path(__file__).with_name("clock_templates.npz")


def _feature(gray: np.ndarray, x0: float, x1: float) -> np.ndarray:
    """One character cell -> FEAT-sized image of the glyph's white strokes.

    Works at the frame's native resolution (no resizing, which blurs thin
    strokes).  Each glyph is white with a solid black outline, so its white
    strokes form a blob separate from bright background.  Blobs touching the
    crop edge (background) are dropped and the largest remaining blob is kept.
    Geometry is corrected for squeezed frames (e.g. 960x1080 shown at 1.85:1).
    """
    h, w = gray.shape
    sx, sy = w / REF_W, h / REF_H
    pad = max(2, int(round(5 * sx)))
    xa, xb = int(round(x0 * w)) - pad, int(round(x1 * w)) + pad
    ya, yb = int(Y0 * h), int(Y1 * h)
    cell = gray[max(ya, 0):yb, max(xa, 0):xb] > 200
    fw, fh = FEAT
    out = np.zeros((fh, fw), np.float32)
    lab, n = ndimage.label(cell)
    best, best_area = 0, 0
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if ys.min() == 0 or xs.min() == 0 or ys.max() == cell.shape[0] - 1 \
                or xs.max() == cell.shape[1] - 1:
            continue
        if len(ys) > best_area:
            best, best_area = i, len(ys)
    if best_area < 25 * sx * sy:
        return out
    ys, xs = np.where(lab == best)
    glyph = (lab[ys.min():ys.max() + 1, xs.min():xs.max() + 1] == best).astype(np.float32)
    ref_h, ref_w = glyph.shape[0] / sy, glyph.shape[1] / sx   # size in calibration pixels
    scale = fh / ref_h
    gw = max(1, min(fw, int(round(ref_w * scale))))
    yi = np.minimum((np.arange(fh) * glyph.shape[0] / fh).astype(int), glyph.shape[0] - 1)
    xi = np.minimum((np.arange(gw) * glyph.shape[1] / gw).astype(int), glyph.shape[1] - 1)
    out[:, (fw - gw) // 2:(fw - gw) // 2 + gw] = glyph[np.ix_(yi, xi)]
    return out


def _gray(frame: np.ndarray) -> np.ndarray:
    f = frame.astype(np.float32)
    return f.mean(2) if f.ndim == 3 else f


def cell_features(frame: np.ndarray, slots, x0: float, pitch: float = TIME_PITCH) -> list[np.ndarray]:
    g = _gray(frame)
    return [_feature(g, x0 + k * pitch, x0 + k * pitch + GLYPH_W) for k in slots]


_T = None


def _templates():
    global _T
    if _T is None:
        z = np.load(TEMPLATES)
        _T = (z["digits"], z["feats"])
    return _T


def read_clock(frame: np.ndarray, max_dist: float = 0.40, margin: float = 1.25):
    """Return seconds since midnight (AM assumed), or None if unreadable.

    The recording window is morning (06:00-10:00); 12-hour 'PM' is not handled.
    """
    digits, feats = _templates()
    out = []
    for f in cell_features(frame, TIME_CELLS, TIME_X0):
        if not f.any():
            return None
        d = np.sqrt(((feats - f) ** 2).mean((1, 2)))
        k = int(d.argmin())
        second = float(np.sort(d)[1])
        if d[k] > max_dist or second < margin * d[k]:   # poor or ambiguous match
            return None
        out.append(int(digits[k]))
    hh, mm, ss = out[0] * 10 + out[1], out[2] * 10 + out[3], out[4] * 10 + out[5]
    if hh > 12 or mm > 59 or ss > 59:
        return None
    return hh * 3600 + mm * 60 + ss
