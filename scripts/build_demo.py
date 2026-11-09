"""Build the data for demo/index.html (the review viewer).

Real data (run where MARS and the embeddings live; thumbnails come from the zip and stay on your machine):
  python scripts/build_demo.py --features data/embeddings/mars_test_osnet_ft.npy \
      --names data/embeddings/mars_test_osnet_ft_names.txt --zip "<MARS>/data/bbox_test.zip" --out demo/data
Sample (no data needed; placeholder silhouettes, synthetic embeddings, NOT a result):
  python scripts/build_demo.py --synthetic --out demo/sample
Then open demo/index.html in a browser. demo/data/ is git-ignored: MARS images must not be redistributed.
"""

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from cctv_reid import demo_data, synthetic  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--features")
p.add_argument("--names")
p.add_argument("--zip", help="MARS bbox_test.zip, only used to write thumbnails")
p.add_argument("--out", required=True)
p.add_argument("--synthetic", action="store_true")
p.add_argument("--queries", type=int, default=12)
p.add_argument("--top-k", type=int, default=12)
p.add_argument("--threshold", type=float, default=0.3, help="agglomerative cosine-distance threshold")
p.add_argument("--seed", type=int, default=255)
p.add_argument("--model-note", default="OSNet x1_0 fine-tuned on MARS train (frame embeddings, mean-pooled per tracklet)")
args = p.parse_args()

if args.synthetic:
    cfg = synthetic.SynthConfig(n_ids=40, n_cams=6, tracklets_per_cam=2, min_frames=6, max_frames=12)
    ds = synthetic.make_dataset(cfg, np.random.default_rng(7))
    names = [None] * len(ds["frames"])
    for key, idxs in ds["index"].items():
        pid, cam, trk = (int(x) for x in key.split("_"))
        for k, i in enumerate(idxs, 1):
            names[i] = f"{pid + 1:04d}C{cam + 1}T{trk + 1:04d}F{k:03d}.jpg"
    feats, note = ds["frames"], "SYNTHETIC sample data: random embeddings, not a research result"
else:
    if not (args.features and args.names):
        p.error("--features and --names are required unless --synthetic")
    feats = np.load(args.features)
    names = [ln.strip() for ln in open(args.names) if ln.strip()]
    note = args.model_note

data = demo_data.build(feats, names, args.queries, args.top_k, args.threshold, args.seed, model_note=note)
data["meta"]["synthetic"] = bool(args.synthetic)

out = Path(args.out)
(out / "thumbs").mkdir(parents=True, exist_ok=True)
needed = set()
for q in data["queries"]:
    needed.update(q["thumbs"])
    for r in q["results"]:
        needed.update(r["thumbs"])
for c in data["clusters"]:
    for m in c["members"][:12]:
        needed.update(m["thumbs"])
if args.zip:
    from PIL import Image

    zf = zipfile.ZipFile(args.zip)
    members = {Path(n).name: n for n in zf.namelist() if n.lower().endswith(".jpg")}
    for n in sorted(needed):
        if n in members:
            Image.open(io.BytesIO(zf.read(members[n]))).convert("RGB").resize((96, 192)).save(out / "thumbs" / n, quality=88)
    print(f"wrote {len(needed)} thumbnails to {out / 'thumbs'}")
(out / "data.js").write_text("window.DEMO_DATA = " + json.dumps(data) + ";\n")
print(f"wrote {out / 'data.js'}: {len(data['queries'])} queries, {len(data['clusters'])} cross-camera clusters")
