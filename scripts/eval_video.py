"""Score a run of scripts/run_video_pipeline.py against your ground-truth log.

  python scripts/eval_video.py --run demo/data --truth data/ground_truth.csv [--threshold 0.3]

ground_truth.csv (one row per appearance; see data/examples/ground_truth_template.csv):
  person,camera,start,end
  Alex,Entrance,2026-11-06 09:00:02,2026-11-06 09:00:20
Use the same time convention as run_video_pipeline.py (clock times if you gave @start times, else HH:MM:SS from video start).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from cctv_reid import io_utils, video_eval  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--run", required=True, help="output folder of run_video_pipeline.py")
p.add_argument("--truth", required=True)
p.add_argument("--threshold", type=float, default=0.3)
p.add_argument("--min-overlap", type=float, default=0.5)
args = p.parse_args()

run = Path(args.run)
rows = list(csv.DictReader(open(run / "tracklets.csv")))
cam_index = {}
tracklets = []
for r in rows:
    cam_index.setdefault(r["camera"], len(cam_index) + 1)
    tracklets.append({"id": r["tracklet_id"], "camera": r["camera"], "camera_index": cam_index[r["camera"]],
                      "start": video_eval.parse_time(r["start_clock"]), "end": video_eval.parse_time(r["end_clock"])})
feats = np.load(run / "tracklet_embeddings.npy")
truth = [{"person": r["person"], "camera": r["camera"], "start": video_eval.parse_time(r["start"]),
          "end": video_eval.parse_time(r["end"])} for r in csv.DictReader(open(args.truth))
         if video_eval.parse_time(r["end"]) > video_eval.parse_time(r["start"])]   # zero-length rows cannot be scored
res = video_eval.evaluate_video(tracklets, feats, truth, args.threshold, args.min_overlap)
labels = res.pop("labels")
for t, lab in zip(tracklets, labels):
    shown = "(ambiguous: excluded)" if lab == "?" else (lab or "(unlabeled)")
    print(f"{t['id']:<22} -> {shown}")
print()
for k, v in res.items():
    print(f"{k:<20} {v:.3f}" if isinstance(v, float) else f"{k:<20} {v}")
io_utils.save_json({**res, "labels": dict(zip([t['id'] for t in tracklets], labels))}, run / "video_eval.json")
print("saved", run / "video_eval.json")
