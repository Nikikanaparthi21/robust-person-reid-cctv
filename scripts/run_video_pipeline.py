"""Real-video pipeline: videos -> person detection -> tracking -> OSNet embeddings -> review viewer data.

UNTESTED with the real detector/OSNet in the cloud sandbox (needs torch + torchvision + torchreid). The video,
tracking, clock and viewer-data logic is tested with a fake detector and encoder on synthetic video.

  python scripts/run_video_pipeline.py \
      --video "Entrance=/path/entrance.mp4@2026-11-06T09:00:00" "Aisle 2=/path/aisle2.mp4@2026-11-06T09:00:07" \
      --weights models/osnet_ft/best.pth --device mps --out demo/data
  open demo/index.html
Long footage: add one or more  --segment HH:MM:SS HH:MM:SS  (video time) to process only those parts of every video,
e.g. --segment 00:12:00 00:22:00 --segment 01:30:00 01:40:00 . Times shown in the viewer stay absolute.

Motion-triggered recordings (clips joined together, so video time is NOT wall time): add --read-clock. The burned-in
clock is read from the frames, timestamps come from it, tracking restarts at every clip boundary, and --video needs only the
date, e.g. "CAM 1=/path/cam1.asf@2026-10-29". Then pick parts by clock time:  --clock-segment 08:18:00 08:21:00 .

Each --video is  CameraName=path[@start-time]. The optional ISO start time is the real clock time of the first frame,
so results show wall-clock times (otherwise HH:MM:SS from the start of that video). Videos are symlinked into the
output folder so the viewer can play the matching clip. Use only footage you are authorized to process; outputs stay local.
Tip for Apple Silicon: the detector runs on CPU by default; PYTORCH_ENABLE_MPS_FALLBACK=1 lets --det-device mps work.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from cctv_reid import demo_data, video_eval, video_pipeline as vp  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--video", nargs="+", required=True, help='CameraName=path[@ISO start time], one per camera')
p.add_argument("--weights", default=None, help="fine-tuned OSNet checkpoint (models/osnet_ft/best.pth)")
p.add_argument("--model", default="osnet_x1_0")
p.add_argument("--device", default="cpu", help="encoder device: cpu | mps | cuda")
p.add_argument("--det-device", default="cpu")
p.add_argument("--detector", default="resnet50", choices=["resnet50", "mobilenet"])
p.add_argument("--segment", nargs=2, action="append", metavar=("START", "END"),
               help="process only this part of every video (HH:MM:SS from the video start); repeat for several segments")
p.add_argument("--read-clock", action="store_true", help="read the burned-in clock for timestamps and clip boundaries")
p.add_argument("--clock-segment", nargs=2, action="append", metavar=("START", "END"),
               help="with --read-clock: process this clock-time window (HH:MM:SS) in every video; repeatable")
p.add_argument("--fps", type=float, default=5.0, help="frames per second to process")
p.add_argument("--det-score", type=float, default=0.6)
p.add_argument("--min-box-height", type=int, default=80)
p.add_argument("--min-hits", type=int, default=5, help="minimum detections for a tracklet")
p.add_argument("--max-missed", type=int, default=15, help="sampled frames a track may be lost before it ends")
p.add_argument("--threshold", type=float, default=0.3, help="clustering cosine-distance threshold")
p.add_argument("--top-k", type=int, default=12)
p.add_argument("--max-queries", type=int, default=300)
p.add_argument("--out", default="demo/data")
args = p.parse_args()

out = Path(args.out)
(out / "thumbs").mkdir(parents=True, exist_ok=True)
(out / "videos").mkdir(parents=True, exist_ok=True)
detector = vp.TorchvisionPersonDetector(args.det_device, args.detector)
from cctv_reid.embeddings import TorchreidEncoder  # noqa: E402

encoder = TorchreidEncoder(args.model, args.weights, args.device)

def video_clock_segments(path, windows, name):
    """Convert clock-time windows (HH:MM:SS) to video-time (start, end) segments for this file.

    The file's clock is indexed once in a single sequential pass (random seeks are very slow in these files) and
    cached next to the results, so later runs on the same file are instant.
    """
    import json

    pth = Path(path)
    cache = out / f"clock_index_{pth.stem}.json"
    key = [pth.stat().st_size, int(pth.stat().st_mtime)]
    index = None
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved.get("key") == key:
            index = saved["index"]
    if index is None:
        print(f"[{name}] indexing the burned-in clock (one pass over the file, cached afterwards) ...", flush=True)
        index = vp.scan_clock_index(path)
        cache.write_text(json.dumps({"key": key, "index": index}))
    hms = lambda v: f"{int(v) // 3600:02d}:{int(v) % 3600 // 60:02d}:{int(v) % 60:02d}"  # noqa: E731
    print(f"[{name}] clock index: {len(index)} samples, {hms(index[0][1])} .. {hms(index[-1][1])}", flush=True)
    segs = []
    for a, b in windows:
        w = vp.window_from_index(index, video_eval.parse_time(a), video_eval.parse_time(b))
        if w is None:
            print(f"[{name}] clock window {a}-{b}: nothing recorded in it (gap between clips), skipped")
        else:
            segs.append(w)
    return segs


all_tracklets, starts, links, dates = [], {}, {}, {}
for cam_idx, spec in enumerate(args.video, 1):
    name, rest = spec.split("=", 1)
    path, _, start = rest.partition("@")
    starts[name] = start or None
    date = start[:10] if start else None
    if args.read_clock and not date:
        sys.exit(f'--read-clock needs the date in --video, e.g. "{name}={path}@2026-10-29"')
    dates[name] = date
    link = out / "videos" / f"cam{cam_idx}{Path(path).suffix}"
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(Path(path).resolve(), link)
    links[name] = f"videos/{link.name}"
    print(f"[{name}] detecting and tracking {path} ...", flush=True)
    if args.clock_segment:
        if not args.read_clock:
            sys.exit("--clock-segment needs --read-clock")
        segments = video_clock_segments(path, args.clock_segment, name)
    else:
        segments = [(video_eval.parse_time(a), None if b is None else video_eval.parse_time(b))   # no --segment = whole file
                    for a, b in (args.segment or [(0, None)])]
    segments = [(a, None if b in (None, 0) else b) for a, b in segments]
    # Save point per camera: a long run that stops partway resumes from the finished cameras.
    import hashlib
    import pickle
    key = hashlib.sha1(repr((Path(path).stat().st_size, int(Path(path).stat().st_mtime), segments, args.fps, args.det_score,
                             args.min_box_height, args.min_hits, args.max_missed, args.detector, args.read_clock)).encode()).hexdigest()[:12]
    cache = out / f"tracklets_cache_{cam_idx}_{key}.pkl"
    if cache.exists():
        cam_tracklets = pickle.loads(cache.read_bytes())
        print(f"[{name}] reusing saved result: {len(cam_tracklets)} tracklets ({cache.name})", flush=True)
        all_tracklets += cam_tracklets
        continue
    cam_tracklets = []
    for k, (a, b) in enumerate(segments, 1):
        if args.segment or args.clock_segment:
            print(f"[{name}] segment {k}: {vp.format_clock(a)} - {vp.format_clock(b) if b else 'end'}", flush=True)
        tls = vp.process_video(path, name, detector, target_fps=args.fps, det_score=args.det_score,
                               min_box_height=args.min_box_height, min_hits=args.min_hits, max_missed=args.max_missed,
                               start_s=a, end_s=b, read_clock=args.read_clock,
                               id_prefix=f"s{k}_" if (args.segment or args.clock_segment) else "")
        print(f"[{name}] {len(tls)} tracklets", flush=True)
        cam_tracklets += tls
    cache.write_bytes(pickle.dumps(cam_tracklets))
    all_tracklets += cam_tracklets
if not all_tracklets:
    sys.exit("no tracklets found: lower --det-score / --min-hits, or check the video")

print("embedding tracklets ...", flush=True)
keys, feats = vp.embed_tracklets(all_tracklets, encoder)

items, manifest = [], []
cam_index = {name: i for i, name in enumerate(starts, 1)}
for t in all_tracklets:
    thumbs = []
    from PIL import Image  # noqa: E402

    for k in sorted({0, len(t.crops) // 2, len(t.crops) - 1}):
        fn = f"{t.tid.replace(' ', '_')}_{k}.jpg"
        Image.fromarray(t.crops[k]).save(out / "thumbs" / fn, quality=88)
        thumbs.append(fn)
    if args.read_clock and t.start_clock_s is not None:
        a, b = vp.format_day_clock(dates[t.camera], t.start_clock_s), vp.format_day_clock(dates[t.camera], t.end_clock_s)
        has_wall = True
    else:
        a, b = vp.format_clock(t.start_s, starts[t.camera]), vp.format_clock(t.end_s, starts[t.camera])
        has_wall = bool(starts[t.camera])
    items.append({"key": t.tid, "camera": cam_index[t.camera], "camera_name": t.camera, "tracklet": t.track_id,
                  "n_frames": len(t.frame_idx), "start_s": round(t.start_s, 2), "end_s": round(t.end_s, 2),
                  "time_label": f"{a} - {b[-8:] if has_wall else b}", "thumbs": thumbs, "video": links[t.camera]})
    manifest.append({"tracklet_id": t.tid, "camera": t.camera, "track_id": t.track_id, "start_s": round(t.start_s, 2),
                     "end_s": round(t.end_s, 2), "start_clock": a, "end_clock": b, "n_detections": len(t.frame_idx),
                     "mean_score": round(float(np.mean(t.scores)), 3)})

order = {k: i for i, k in enumerate(keys)}
feats = feats[[order[it["key"]] for it in items]]
data = demo_data.build_from_tracklets(items, feats, args.top_k, args.threshold, args.max_queries,
                                      model_note=f"{args.model} {'fine-tuned' if args.weights else 'ImageNet'}; {args.detector} person detector")
(out / "data.js").write_text("window.DEMO_DATA = " + json.dumps(data) + ";\n")
np.save(out / "tracklet_embeddings.npy", feats)
with open(out / "tracklets.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(manifest[0]))
    w.writeheader()
    w.writerows(manifest)
print(f"done: {len(items)} tracklets -> {out}/data.js (open demo/index.html)")
