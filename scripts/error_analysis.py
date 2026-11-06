"""Error taxonomy on precomputed MARS embeddings (same inputs as evaluate_features.py).

Usage:
  python scripts/error_analysis.py --features data/embeddings/mars_test.npy \
      --names data/embeddings/mars_test_names.txt --pooling mean --out results/mars_resnet50_frozen/errors
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from cctv_reid import error_analysis, io_utils  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--features", required=True)
p.add_argument("--names", required=True)
p.add_argument("--quality")
p.add_argument("--pooling", default="mean", choices=["mean", "attention", "quality_filtered", "quality_weighted"])
p.add_argument("--cluster-method", default="agglomerative", choices=["agglomerative", "dbscan"])
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=255)
p.add_argument("--max-ids", type=int, default=100)
p.add_argument("--max-frames", type=int, default=16)
a = p.parse_args()

feats = np.load(a.features)
names = [ln.strip() for ln in open(a.names) if ln.strip()]
q = np.load(a.quality) if a.quality else None
res = error_analysis.analyze(feats, names, a.pooling, a.cluster_method, a.seed, a.max_ids, a.max_frames, q)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
io_utils.save_json(res, out / f"errors_{a.pooling}_{a.cluster_method}.json")
(out / f"errors_{a.pooling}_{a.cluster_method}.md").write_text(error_analysis.to_markdown(res))
print(error_analysis.to_markdown(res))
