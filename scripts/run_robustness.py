"""Robustness experiments (issue #24): degrade the MARS test images, re-embed, re-evaluate.  UNTESTED here
(needs torch, torchreid, MARS); the condition, quality and sampling helpers are unit-tested.

Conditions: blur, darkening, low resolution, occlusion (each at 2 levels) plus a frame-sampling sweep that
keeps every 2nd/4th/8th frame of each tracklet. Each condition re-embeds the same 19,776-image subset
(about 8 minutes on an M-series Mac), so the full run is roughly 70 minutes. The run is resumable.

  python scripts/run_robustness.py \
      --zip "<MARS>/data/bbox_test.zip" --weights models/osnet_ft/best.pth --device mps \
      --out results/mars_osnet_robustness
  # quick check first (about 1 minute): add  --conditions clean blur_s2 --max-ids 20

Per-frame quality = sharpness percentile rank, which also enables quality-filtered / quality-weighted pooling.
"""

import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from cctv_reid import io_utils, mars_eval, robustness  # noqa: E402
from cctv_reid.mars_names import select_frames  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--zip", required=True, help="path to bbox_test.zip")
p.add_argument("--out", required=True)
p.add_argument("--model", default="osnet_x1_0")
p.add_argument("--weights", default=None, help="fine-tuned checkpoint (omit for ImageNet-only OSNet)")
p.add_argument("--device", default="cpu")
p.add_argument("--conditions", nargs="*", default=list(robustness.CONDITIONS))
p.add_argument("--sampling", nargs="*", type=int, default=[2, 4, 8], help="keep-every-n frame sweeps on clean features")
p.add_argument("--batch", type=int, default=128)
p.add_argument("--seed", type=int, default=255)
p.add_argument("--max-ids", type=int, default=100)
p.add_argument("--max-frames", type=int, default=16)
p.add_argument("--max-distractor-tracklets", type=int, default=150)
args = p.parse_args()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from cctv_reid.embeddings import TorchreidEncoder  # noqa: E402

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
csv_path = out / "robustness.csv"
done = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
finished = set(done["condition"]) if len(done) else set()

zf = zipfile.ZipFile(args.zip)
members = {Path(n).name: n for n in zf.namelist() if n.lower().endswith(".jpg")}
names = select_frames(list(members), args.seed, args.max_ids, args.max_frames, args.max_distractor_tracklets)
print(f"{len(names)} images selected", flush=True)
enc = TorchreidEncoder(args.model, args.weights, args.device)
KEEP = ["method", "rank1", "rank5", "mAP", "dbscan_f1", "agglomerative_f1", "agglomerative_clusters_per_identity",
        "agglomerative_precision", "agglomerative_recall"]
rows = done.to_dict("records") if len(done) else []
clean_feats = clean_quality = None


def record(condition, family, level, results):
    for r in results:
        rows.append({"condition": condition, "family": family, "level": level, **{k: r[k] for k in KEEP}})
    pd.DataFrame(rows).to_csv(csv_path, index=False)


for cond in args.conditions:
    family, level, _ = robustness.CONDITIONS[cond]
    if cond in finished and cond != "clean":
        print("skip (already done):", cond)
        continue
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    feats, quals = [], []
    for s in range(0, len(names), args.batch):
        imgs = np.stack([
            robustness.apply_condition(
                np.asarray(Image.open(io.BytesIO(zf.read(members[n]))).convert("RGB").resize((128, 256))), cond, rng)
            for n in names[s : s + args.batch]
        ])
        feats.append(enc.encode(imgs))
        quals.append(imgs.mean(axis=3))  # grayscale copies only, to keep memory small
    feats = np.concatenate(feats)
    quality = robustness.sharpness_from_gray(np.concatenate(quals))
    if cond == "clean":
        clean_feats, clean_quality = feats, quality
    if cond not in finished:
        res = mars_eval.evaluate(feats, names, args.seed, args.max_ids, args.max_frames, quality)
        record(cond, family, level, res)
        best = {r["method"]: round(r["mAP"], 3) for r in res}
        print(f"{cond}: mAP {best}  ({time.time() - t0:.0f}s)", flush=True)

if clean_feats is None and args.sampling:
    print("sampling sweep needs the clean condition in the same run (include 'clean' in --conditions)")
elif args.sampling:
    for k in args.sampling:
        cond = f"sample_every_{k}"
        if cond in finished:
            continue
        keep = robustness.subsample_indices(names, k)
        res = mars_eval.evaluate(clean_feats[keep], [names[i] for i in keep], args.seed, args.max_ids,
                                 args.max_frames, clean_quality[keep])
        record(cond, "sampling", k, res)
        print(f"{cond}: mAP { {r['method']: round(r['mAP'], 3) for r in res} }", flush=True)

io_utils.save_json({**vars(args), "quality": "sharpness percentile rank",
                    "note": "degradations applied to the 128x256 encoder input; MARS-style protocol"},
                   out / "run_config.json")

# Plot: mAP vs severity per family, one line per pooling method.
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

df = pd.DataFrame(rows)
fams = [f for f in ["blur", "darkening", "low_resolution", "occlusion", "sampling"] if f in set(df["family"])]
if fams and "clean" in set(df["condition"]):
    fig, axes = plt.subplots(1, len(fams), figsize=(3.2 * len(fams), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    clean = df[df["condition"] == "clean"].set_index("method")["mAP"]
    for ax, fam in zip(axes, fams):
        for m in ("frame", "mean", "attention"):
            d = df[(df["family"] == fam) & (df["method"] == m)].sort_values("level")
            xs = [0] + list(d["level"]) if fam != "sampling" else [1] + list(d["level"])
            ys = [clean[m]] + list(d["mAP"])
            ax.plot(range(len(xs)), ys, marker="o", label=m)
            ax.set_xticks(range(len(xs)), [str(x) for x in xs])
        ax.set_title(fam.replace("_", " "))
        ax.set_xlabel("keep every n-th frame" if fam == "sampling" else "severity (0 = clean)")
    axes[0].set_ylabel("mAP")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "robustness_map.png", dpi=160)
    print("saved", out / "robustness_map.png")
print("done ->", csv_path)
