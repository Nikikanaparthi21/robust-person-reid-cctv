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
p.add_argument("--layout", default=None,
               help='camera layout CSV, or "default" for walking times taken from your notes; adds a plain-vs-time-prior comparison')
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
layout = None
if args.layout:
    from cctv_reid import time_prior
    layout = time_prior.DEFAULT_LAYOUT if args.layout == "default" else time_prior.load_layout(args.layout)
res = video_eval.evaluate_video(tracklets, feats, truth, args.threshold, args.min_overlap, layout)
tp = res.pop("time_prior", None)
labels = res.pop("labels")
details = res.pop("details", [])
for t, lab in zip(tracklets, labels):
    shown = "(ambiguous: excluded)" if lab == "?" else (lab or "(unlabeled)")
    print(f"{t['id']:<22} -> {shown}")
print()
for k, v in res.items():
    print(f"{k:<20} {v:.3f}" if isinstance(v, float) else f"{k:<20} {v}")
if tp:
    print("\nwalking-time prior (bonus added to similarity when the gap fits the layout; 0.0 = appearance only):")
    for b, m in tp.items():
        print(f"  bonus {b:<5} rank1 {m['rank1']:.3f}  rank5 {m['rank5']:.3f}  mAP {m['mAP']:.3f}")
if details:
    print("\nper-query cross-camera results (query -> first correct rank; wrong top-1 shown):")
    for d in details:
        ok = d["first_correct_rank"] == 1
        print(f"  {d['query']:<14} {d['person']:<8} first correct at rank {d['first_correct_rank']}"
              + ("" if ok else f"   top-1 was {d['top1']} ({d['top1_label']}, sim {d['top1_sim']})"))
from datetime import datetime

unl = [(t, lab) for t, lab in zip(tracklets, labels) if lab is None]
if unl:
    print(f"\n{len(unl)} unlabeled tracklets (no logged person overlaps them). If one is really a person who is missing from your log,")
    print("it counts as a wrong match above. Worth a look in the viewer:")
    for t, _ in unl:
        print(f"  {t['id']:<12} {t['camera']}  {datetime.fromtimestamp(t['start']):%H:%M:%S} - {datetime.fromtimestamp(t['end']):%H:%M:%S}")
rows_cov, summ = video_eval.coverage(tracklets, truth, args.min_overlap)
print("\nlogged appearances found by the pipeline (a tracklet on that camera lying inside the logged interval):")
for name, (f, n) in summ.items():
    print(f"  {name:<8} {f}/{n}" + (f"  ({100 * f / n:.0f}%)" if n else ""))
missed = [o for o in rows_cov if not o["found"] and o["dur"] >= 10]
if missed:
    print(f"  missed appearances of 10 s or longer ({len(missed)}):")
    for o in sorted(missed, key=lambda o: o["start"])[:25]:
        print(f"    {o['person']:<10} {o['camera']}  {datetime.fromtimestamp(o['start']):%H:%M:%S} - {datetime.fromtimestamp(o['end']):%H:%M:%S}  ({o['dur']:.0f} s)")
io_utils.save_json({**res, "time_prior": tp, "details": details, "labels": dict(zip([t['id'] for t in tracklets], labels))}, run / "video_eval.json")
print("saved", run / "video_eval.json")
