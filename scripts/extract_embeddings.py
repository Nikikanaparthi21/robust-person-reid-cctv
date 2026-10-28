"""Extract frozen embeddings for a reproducible MARS subset, reading directly from the zip.

UNTESTED here (needs torch, MARS and weights); subset selection IS tested. Step 1: dry run (no torch needed):
  python scripts/extract_embeddings.py --zip "<MARS>/data/bbox_test.zip" --out data/embeddings/mars_test --dry-run
Step 2: extract (pip install torch torchvision pillow; add torchreid for OSNet):
  python scripts/extract_embeddings.py --zip "<MARS>/data/bbox_test.zip" --out data/embeddings/mars_test \
      --model resnet50 --device mps          # or --model osnet_x1_0, --device cuda / cpu
Outputs <out>.npy (N, D) and <out>_names.txt. Keep both out of Git. Default subset: 100 identities,
<= 16 evenly spaced frames per tracklet, 150 distractor tracklets, seed 255 (about 20k images).
"""

import argparse
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from cctv_reid.mars_names import select_frames  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--zip", required=True, help="path to bbox_test.zip (or bbox_train.zip)")
p.add_argument("--out", required=True)
p.add_argument("--model", default="resnet50", choices=["resnet50", "osnet_x1_0"])
p.add_argument("--weights", default=None, help="optional local OSNet weights file")
p.add_argument("--device", default="cpu")
p.add_argument("--batch", type=int, default=128)
p.add_argument("--seed", type=int, default=255)
p.add_argument("--max-ids", type=int, default=100)
p.add_argument("--max-frames", type=int, default=16)
p.add_argument("--max-distractor-tracklets", type=int, default=150)
p.add_argument("--dry-run", action="store_true", help="only report the selected subset")
args = p.parse_args()

zf = zipfile.ZipFile(args.zip)
members = {Path(n).name: n for n in zf.namelist() if n.lower().endswith(".jpg")}
chosen = select_frames(list(members), args.seed, args.max_ids, args.max_frames, args.max_distractor_tracklets)
print(f"{len(members)} images in zip -> {len(chosen)} selected ({args.max_ids} identities, seed {args.seed})")
if args.dry_run:
    sys.exit(0)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

if args.model == "resnet50":
    from cctv_reid.embeddings import ResNet50Encoder  # noqa: E402

    enc = ResNet50Encoder(args.device)
else:
    from cctv_reid.embeddings import TorchreidEncoder  # noqa: E402

    enc = TorchreidEncoder(args.model, args.weights, args.device)

feats = []
for s in range(0, len(chosen), args.batch):
    batch_names = chosen[s : s + args.batch]
    imgs = np.stack([
        np.asarray(Image.open(io.BytesIO(zf.read(members[n]))).convert("RGB").resize((128, 256)))
        for n in batch_names
    ])
    feats.append(enc.encode(imgs))
    if (s // args.batch) % 10 == 0:
        print(f"{s + len(batch_names)}/{len(chosen)}", flush=True)

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
np.save(f"{out}.npy", np.concatenate(feats))
Path(f"{out}_names.txt").write_text("\n".join(chosen) + "\n")
print("saved", f"{out}.npy", f"{out}_names.txt")
