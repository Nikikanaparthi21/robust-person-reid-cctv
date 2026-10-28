"""Extract frozen OSNet embeddings for a MARS split (UNTESTED here: needs torch, torchreid, MARS, weights).

Setup on a laptop/GPU box:
  pip install torch torchvision torchreid pillow
Usage:
  python scripts/extract_embeddings.py --mars-root /path/to/MARS --split bbox_test \
      --out data/embeddings/mars_test --device mps      # or cuda / cpu
Outputs <out>.npy (N, D) and <out>_names.txt. Keep both out of Git (data/ is local).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from cctv_reid import io_utils  # noqa: E402
from cctv_reid.embeddings import TorchreidEncoder  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--mars-root", required=True)
p.add_argument("--split", default="bbox_test")
p.add_argument("--out", required=True)
p.add_argument("--model", default="osnet_x1_0")
p.add_argument("--weights", default=None, help="optional local weights file; default downloads ImageNet-pretrained")
p.add_argument("--device", default="cpu")
p.add_argument("--batch", type=int, default=128)
p.add_argument("--every", type=int, default=1, help="keep every n-th frame per tracklet to save time")
args = p.parse_args()

records = io_utils.scan_mars_split(args.mars_root, args.split)
groups = io_utils.group_into_tracklets(records)
chosen = sorted(i for idx in groups.values() for i in idx[:: args.every])
enc = TorchreidEncoder(args.model, args.weights, args.device)

feats, names = [], []
for s in range(0, len(chosen), args.batch):
    batch_idx = chosen[s : s + args.batch]
    imgs = np.stack([np.asarray(Image.open(records[i].path).convert("RGB").resize((128, 256))) for i in batch_idx])
    feats.append(enc.encode(imgs))
    names += [Path(records[i].path).name for i in batch_idx]
    if (s // args.batch) % 20 == 0:
        print(f"{s + len(batch_idx)}/{len(chosen)}", flush=True)

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
np.save(f"{out}.npy", np.concatenate(feats))
Path(f"{out}_names.txt").write_text("\n".join(names) + "\n")
print("saved", f"{out}.npy")
