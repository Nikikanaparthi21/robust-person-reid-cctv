"""Video -> tracklets -> embeddings -> viewer data. Detector and encoder are plugged in as callables, so everything
here except the two model wrappers runs (and is tested) without torch.

detector(list_of_RGB_images) -> list of (boxes (N,4) xyxy, scores (N,)) per image
encoder.encode((N, 256, 128, 3) uint8 RGB) -> (N, D) float32
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import embeddings
from .detection import from_array
from .tracking import ByteTrackLite

CROP_W, CROP_H = 128, 256


@dataclass
class VideoTracklet:
    tid: str
    camera: str
    track_id: int
    frame_idx: list[int]              # original video frame numbers
    times_s: list[float]              # seconds from the start of the video
    boxes: list[np.ndarray] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    crops: list[np.ndarray] = field(default_factory=list)   # RGB uint8 (256, 128, 3), evenly spaced subset

    @property
    def start_s(self) -> float:
        return self.times_s[0]

    @property
    def end_s(self) -> float:
        return self.times_s[-1]


def read_frames(path: str, every_n: int | None = None, target_fps: float = 5.0):
    """Yield (frame_idx, time_s, RGB image) for sampled frames. Returns fps via the first yielded item's attribute."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = every_n if every_n else max(1, int(round(fps / target_fps)))
    i = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if i % step == 0:
                yield i, i / fps, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            i += 1
    finally:
        cap.release()


def video_info(path: str) -> tuple[float, int]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 25.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return float(fps), n


def crop_person(img: np.ndarray, box, out_w: int = CROP_W, out_h: int = CROP_H) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, x2 = max(0, min(w - 1, x1)), max(1, min(w, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(1, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((out_h, out_w, 3), np.uint8)
    return cv2.resize(img[y1:y2, x1:x2], (out_w, out_h), interpolation=cv2.INTER_AREA)


def process_video(path: str, camera: str, detector, every_n: int | None = None, target_fps: float = 5.0,
                  det_score: float = 0.5, min_box_height: int = 60, min_hits: int = 5, max_missed: int = 15,
                  max_crops: int = 16, batch: int = 8) -> list[VideoTracklet]:
    """Detect persons in sampled frames, track them, and keep up to `max_crops` evenly spaced crops per tracklet."""
    meta, frames_dets = [], []
    pending: list[tuple[int, float, np.ndarray]] = []

    def flush():
        if not pending:
            return
        outs = detector([p[2] for p in pending])
        for (fi, t, img), (boxes, scores) in zip(pending, outs):
            boxes = np.asarray(boxes, float).reshape(-1, 4)
            scores = np.asarray(scores, float).reshape(-1)
            ok = (scores >= det_score) & ((boxes[:, 3] - boxes[:, 1]) >= min_box_height) if len(boxes) else []
            dets = from_array(fi, boxes[ok], scores[ok]) if len(boxes) else []
            frames_dets.append(dets)
            meta.append((fi, t))
        pending.clear()

    # Pass 1: detect + track (no frames are kept in memory). Pass 2 re-reads the video for the few crops needed.
    for fi, t, img in read_frames(path, every_n, target_fps):
        pending.append((fi, t, img))
        if len(pending) >= batch:
            flush()
    flush()

    time_of = dict(meta)
    tracks = ByteTrackLite(iou_thresh=0.3, max_missed=max_missed, min_hits=min_hits).run(frames_dets)
    out, wanted = [], {}
    for track_id, dets in sorted(tracks.items()):
        pick = np.linspace(0, len(dets) - 1, min(max_crops, len(dets))).round().astype(int)
        tl = VideoTracklet(tid=f"{camera}_{track_id}", camera=camera, track_id=int(track_id),
                           frame_idx=[d.frame for d in dets], times_s=[time_of[d.frame] for d in dets],
                           boxes=[d.box for d in dets], scores=[d.score for d in dets])
        out.append(tl)
        for rank, j in enumerate(pick):
            wanted.setdefault(dets[j].frame, []).append((tl, rank, dets[j].box))
    for tl in out:
        tl.crops = [None] * len([1 for v in wanted.values() for w in v if w[0] is tl])
    for fi, t, img in read_frames(path, every_n, target_fps):
        for tl, rank, box in wanted.get(fi, []):
            tl.crops[rank] = crop_person(img, box)
    return out


def embed_tracklets(tracklets: list[VideoTracklet], encoder, batch: int = 64, pooling: str = "mean"):
    """Embed each tracklet's crops and pool to one L2-normalised vector per tracklet."""
    flat = [c for t in tracklets for c in t.crops]
    frame_feats = np.concatenate([encoder.encode(np.stack(flat[s:s + batch])) for s in range(0, len(flat), batch)]) \
        if flat else np.zeros((0, 1), np.float32)
    index, pos = {}, 0
    for t in tracklets:
        index[t.tid] = list(range(pos, pos + len(t.crops)))
        pos += len(t.crops)
    keys, pooled = embeddings.pool_tracklets(frame_feats, index, pooling)
    return keys, pooled


def format_clock(seconds: float, start_iso: str | None = None) -> str:
    """HH:MM:SS from video start, or a wall-clock time when the video's real start time is known."""
    from datetime import datetime, timedelta

    if start_iso:
        return (datetime.fromisoformat(start_iso) + timedelta(seconds=float(seconds))).strftime("%Y-%m-%d %H:%M:%S")
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


class TorchvisionPersonDetector:
    """COCO-pretrained Faster R-CNN, person class only. Needs torch + torchvision (downloads weights on first use)."""

    def __init__(self, device: str = "cpu", model: str = "resnet50"):
        import torch
        import torchvision
        from torchvision.models import detection as D

        self.torch, self.device = torch, torch.device(device)
        if model == "mobilenet":
            net = D.fasterrcnn_mobilenet_v3_large_fpn(weights=D.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
        else:
            net = D.fasterrcnn_resnet50_fpn(weights=D.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.net = net.eval().to(self.device)

    def __call__(self, images):
        torch = self.torch
        batch = [torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float().div(255).to(self.device) for im in images]
        with torch.no_grad():
            outs = self.net(batch)
        res = []
        for o in outs:
            m = o["labels"] == 1  # COCO person
            res.append((o["boxes"][m].cpu().numpy(), o["scores"][m].cpu().numpy()))
        return res
