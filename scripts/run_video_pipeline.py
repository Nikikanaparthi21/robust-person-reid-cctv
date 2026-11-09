"""Real-video pipeline: videos -> person detection -> tracking -> OSNet embeddings -> review viewer data.

UNTESTED with the real detector/OSNet in the cloud sandbox (needs torch + torchvision + torchreid). The video,
tracking, clock and viewer-data logic is tested with a fake detector and encoder on synthetic video.

  python scripts/run_video_pipeline.py \
      --video "Entrance=/path/entrance.mp4@2026-11-06T09:00:00" "Aisle 2=/path/aisle2.mp4@2026-11-06T09:00:07" \
      --weights models/osnet_ft/best.pth --device mps --out demo/data
  open demo/index.html
Long footage: add one or more  --segment HH:MM:SS HH:MM:SS  (video time) to process only those parts of every video,
e.g. --segment 00:12:00 00:22:00 --segment 01:30:00 01:40:00 . Times shown in the viewer stay absolute.

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

all_tracklets, starts, links = [], {}, {}
for cam_idx, spec in enumerate(args.video, 1):
    name, rest = spec.split("=", 1)
    path, _, start = rest.partition("@")
    starts[name] = start or None
    link = out / "videos" / f"cam{cam_idx}{Path(path).suffix}"
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(Path(path).resolve(), link)
    links[name] = f"videos/{link.name}"
    print(f"[{name}] detecting and tracking {path} ...", flush=True)
    segments = [(video_eval.parse_time(a), video_eval.parse_time(b)) for a, b in (args.segment or [(0, None)])]
    segments = [(a, None if b in (None, 0) else b) for a, b in segments]
    for k, (a, b) in enumerate(segments, 1):
        if args.segment:
            print(f"[{name}] segment {k}: {vp.format_clock(a)} - {vp.format_clock(b) if b else 'end'}", flush=True)
        tls = vp.process_video(path, name, detector, target_fps=args.fps, det_score=args.det_score,
                               min_box_height=args.min_box_height, min_hits=args.min_hits, max_missed=args.max_missed,
                               start_s=a, end_s=b, id_prefix=f"s{k}_" if args.segment else "")
        print(f"[{name}] {len(tls)} tracklets", flush=True)
        all_tracklets += tls
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
    a, b = vp.format_clock(t.start_s, starts[t.camera]), vp.format_clock(t.end_s, starts[t.camera])
    items.append({"key": t.tid, "camera": cam_index[t.camera], "camera_name": t.camera, "tracklet": t.track_id,
                  "n_frames": len(t.frame_idx), "start_s": round(t.start_s, 2), "end_s": round(t.end_s, 2),
                  "time_label": f"{a} - {b[-8:] if starts[t.camera] else b}", "thumbs": thumbs, "video": links[t.camera]})
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
