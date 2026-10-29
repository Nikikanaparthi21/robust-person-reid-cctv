"""Run the synthetic sanity benchmark and write results/synthetic/{results.csv,summary.md}.

Synthetic data only validates pipeline logic; it is NOT evidence about real CCTV footage.
Usage: python scripts/run_synthetic.py --seeds 255 256 257 258 259
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from cctv_reid import synthetic  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, nargs="+", default=[255, 256, 257, 258, 259])
p.add_argument("--out", default="results/synthetic")
p.add_argument("--suite", choices=["standard", "stress"], default="standard")
args = p.parse_args()

t0 = time.time()
conds = synthetic.CONDITIONS if args.suite == "standard" else synthetic.STRESS_CONDITIONS
rows = synthetic.run_all(args.seeds, conditions=conds)
df = pd.DataFrame(rows)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "results.csv", index=False)

metrics = ["rank1", "mAP", "agglomerative_f1", "agglomerative_ari", "agglomerative_clusters_per_identity",
           "dbscan_f1", "dbscan_clusters_per_identity"]
order = {m: i for i, m in enumerate(synthetic.METHODS)}
g = df.groupby(["condition", "method"])[metrics].agg(["mean", "std"])
lines = [f"# Synthetic sanity benchmark (seeds {args.seeds}, mean ± std)\n",
         "Synthetic embeddings only: validates pipeline logic, not real CCTV behaviour.\n"]
for cond in conds:
    lines.append(f"\n## {cond}\n")
    lines.append("| method | " + " | ".join(metrics) + " |")
    lines.append("|---|" + "---|" * len(metrics))
    for m in sorted(synthetic.METHODS, key=order.get):
        r = g.loc[(cond, m)]
        cells = [f"{r[(k, 'mean')]:.3f} ± {r[(k, 'std')]:.3f}" for k in metrics]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
tm = df.groupby("method")[["retrieval_s", "dbscan_s", "agglomerative_s"]].mean().round(3)
lines.append("\n## Mean runtime per run (seconds, 2 CPU cores)\n")
lines.append(tm.to_markdown())
(out / "summary.md").write_text("\n".join(lines) + "\n")
print(f"done in {time.time() - t0:.0f}s -> {out}")
