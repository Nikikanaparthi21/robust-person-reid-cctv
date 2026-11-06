"""Evaluate precomputed embeddings on MARS (run this where MARS and the encoder weights live).

Usage:
  python scripts/evaluate_features.py --features data/embeddings/mars_test.npy \
      --names data/embeddings/mars_test_names.txt --out results/mars_resnet50_frozen
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from cctv_reid import io_utils, mars_eval  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--features", required=True, help="(N, D) .npy of frame embeddings")
p.add_argument("--names", required=True, help="text file: one MARS image file name per row, same order")
p.add_argument("--quality", help="optional (N,) .npy of per-frame quality scores")
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=255)
p.add_argument("--max-ids", type=int, default=100)
p.add_argument("--max-frames", type=int, default=16)
args = p.parse_args()

feats = np.load(args.features)
names = [ln.strip() for ln in open(args.names) if ln.strip()]
assert len(names) == len(feats), "features and names must align"
quality = np.load(args.quality) if args.quality else None
rows = mars_eval.evaluate(feats, names, args.seed, args.max_ids, args.max_frames, quality)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
io_utils.save_json({"features": args.features, "seed": args.seed, "max_ids": args.max_ids,
                    "max_frames": args.max_frames, "protocol": "MARS-style (camera-1 queries), see mars_eval.py"},
                   out / "run_config.json")
print(pd.DataFrame(rows).round(3).to_string())
