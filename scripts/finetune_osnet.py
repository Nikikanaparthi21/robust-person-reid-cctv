"""Fine-tune OSNet (ID cross-entropy + batch-hard triplet loss) on MARS bbox_train.  (issue #21)

UNTESTED in the cloud sandbox (needs torch, torchreid and MARS); the helpers it uses are unit-tested.
Train identities (MARS train, 625 ids) are disjoint from the test identities used for evaluation.
A slice of train identities is held out for validation, so test data is never used for model selection.

  pip install torch torchvision torchreid pillow
  python scripts/finetune_osnet.py --zip "<MARS>/data/bbox_train.zip" --out models/osnet_ft --device mps
  # smoke test first (a minute or two):
  python scripts/finetune_osnet.py --zip ... --out models/osnet_smoke --max-ids 40 --epochs 1 --iters 5 --device mps

Then extract with the fine-tuned weights and evaluate exactly like the frozen baseline:
  python scripts/extract_embeddings.py --zip "<MARS>/data/bbox_test.zip" --out data/embeddings/mars_test_osnet_ft \
      --model osnet_x1_0 --weights models/osnet_ft/best.pth --device mps
  python scripts/evaluate_features.py --features data/embeddings/mars_test_osnet_ft.npy \
      --names data/embeddings/mars_test_osnet_ft_names.txt --out results/mars_osnet_finetuned

Keep models/ out of Git (weights are gitignored). Record the commit, seed and run_config.json with the results.
"""

import argparse
import csv
import io
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from cctv_reid import evaluation, io_utils  # noqa: E402
from cctv_reid.finetune_utils import pk_batches, split_identities  # noqa: E402
from cctv_reid.mars_names import parse_mars_name, select_frames  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--zip", required=True, help="path to MARS bbox_train.zip")
p.add_argument("--out", required=True, help="output folder for checkpoints and logs")
p.add_argument("--device", default="cpu", help="cpu | mps | cuda")
p.add_argument("--model", default="osnet_x1_0")
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--iters", type=int, default=150, help="batches per epoch")
p.add_argument("--p", type=int, default=16, help="identities per batch")
p.add_argument("--k", type=int, default=4, help="images per identity per batch")
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--weight-decay", type=float, default=5e-4)
p.add_argument("--margin", type=float, default=0.3)
p.add_argument("--label-smooth", type=float, default=0.1)
p.add_argument("--max-ids", type=int, default=625, help="train identities to use (625 = all)")
p.add_argument("--max-frames", type=int, default=8, help="frames kept per tracklet (evenly spaced)")
p.add_argument("--val-fraction", type=float, default=0.1)
p.add_argument("--seed", type=int, default=255)
args = p.parse_args()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchreid  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms as T  # noqa: E402

io_utils.set_seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device(args.device)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

# ---- data: lazy reads from the zip, subset chosen deterministically ----
zf = zipfile.ZipFile(args.zip)
members = {Path(n).name: n for n in zf.namelist() if n.lower().endswith(".jpg")}
names = select_frames(list(members), args.seed, args.max_ids, args.max_frames, max_distractor_tracklets=0)
parsed = [parse_mars_name(n) for n in names]
pids = [x[0] for x in parsed]
cams = [x[1] for x in parsed]
train_ids, val_ids = split_identities(pids, args.val_fraction, args.seed)
label_of = {pid: i for i, pid in enumerate(sorted(train_ids))}
train_idx = [i for i, pid in enumerate(pids) if pid in train_ids]
val_idx = [i for i, pid in enumerate(pids) if pid in val_ids]
train_labels = [label_of[pids[i]] for i in train_idx]
print(f"{len(names)} images; train ids {len(train_ids)} ({len(train_idx)} imgs), val ids {len(val_ids)} ({len(val_idx)} imgs)")

mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
augment = T.Compose([T.RandomHorizontalFlip(), T.Pad(10), T.RandomCrop((256, 128)),
                     T.RandomErasing(p=0.5, value=0)])


def load(i: int) -> "torch.Tensor":
    img = Image.open(io.BytesIO(zf.read(members[names[i]]))).convert("RGB").resize((128, 256))
    return torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0


def normalise(x):
    return (x - mean) / std


# ---- model: loss='triplet' makes train-mode forward return (logits, features) ----
model = torchreid.models.build_model(args.model, num_classes=len(train_ids), loss="triplet", pretrained=True)
model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * args.iters)


def batch_hard_triplet(feats, y, margin):
    d = torch.cdist(feats, feats)
    same = y[:, None] == y[None, :]
    hardest_pos = (d * same.float()).max(dim=1).values
    hardest_neg = (d + 1e5 * same.float()).min(dim=1).values
    return F.relu(hardest_pos - hardest_neg + margin).mean()


@torch.no_grad()
def embed(indices):
    model.eval()
    feats = []
    for s in range(0, len(indices), 128):
        x = torch.stack([normalise(load(i)) for i in indices[s:s + 128]]).to(device)
        feats.append(model(x).cpu().numpy())
    return np.concatenate(feats)


def validate():
    """Frame-level rank-1/mAP on held-out TRAIN identities (camera 1 queries vs other cameras)."""
    f = embed(val_idx)
    vp = np.array([pids[i] for i in val_idx])
    vc = np.array([cams[i] for i in val_idx])
    q, g = vc == 1, vc != 1
    r = evaluation.retrieval_metrics(f[q], vp[q], vc[q], f[g], vp[g], vc[g])
    return r["rank1"], r["mAP"]


log_path = out / "train_log.csv"
with open(log_path, "w", newline="") as fh:
    csv.writer(fh).writerow(["epoch", "loss", "ce", "triplet", "val_rank1", "val_mAP", "seconds"])
best = -1.0
t_start = time.time()
for epoch in range(1, args.epochs + 1):
    model.train()
    losses = []
    batches = pk_batches(train_labels, args.p, args.k, args.iters, args.seed + epoch)
    for batch in batches:
        x = torch.stack([augment(load(train_idx[j])) for j in batch])
        x = normalise(x).to(device)
        y = torch.tensor([train_labels[j] for j in batch], device=device)
        logits, feats = model(x)
        ce = F.cross_entropy(logits, y, label_smoothing=args.label_smooth)
        tri = batch_hard_triplet(feats, y, args.margin)
        loss = ce + tri
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append((loss.item(), ce.item(), tri.item()))
    mean_l = np.mean(losses, axis=0)
    r1, mAP = validate()
    secs = time.time() - t_start
    with open(log_path, "a", newline="") as fh:
        csv.writer(fh).writerow([epoch, *np.round(mean_l, 4), round(r1, 4), round(mAP, 4), round(secs)])
    print(f"epoch {epoch}/{args.epochs} loss {mean_l[0]:.3f} (ce {mean_l[1]:.3f}, tri {mean_l[2]:.3f}) "
          f"val rank-1 {r1:.3f} mAP {mAP:.3f}  {secs:.0f}s", flush=True)
    torch.save({"state_dict": model.state_dict(), "epoch": epoch}, out / "last.pth")
    if mAP > best:
        best = mAP
        torch.save({"state_dict": model.state_dict(), "epoch": epoch}, out / "best.pth")

io_utils.save_json({**vars(args), "train_identities": len(train_ids), "val_identities": len(val_ids),
                    "train_images": len(train_idx), "best_val_mAP": best,
                    "protocol": "val = held-out MARS-train identities; test identities never used"},
                   out / "run_config.json")
print("best val mAP", round(best, 4), "->", out / "best.pth")
