# Reproducibility Guide

Everything in the report regenerates from the commands below. All runs are seeded (`seed = 255`).

## 1. Environment

```bash
make setup                      # venv + requirements.txt
pip install torch torchvision   # only needed for real embedding extraction
```

Exact package versions used are frozen in `docs/environment-freeze.txt`.

## 2. Tests (no data needed)

```bash
make test        # unit tests: parsing, pooling, clustering, metrics, tracking, error analysis
make synthetic   # pipeline sanity benchmark; checks the code, not a research claim
```

## 3. Real data: MARS

Raw MARS files are not in the repository (size and licence). Download `bbox_test.zip` from the approved source, then:

```bash
make mars ZIP="/path/to/MARS/bbox_test.zip" DEVICE=mps    # mps | cpu | cuda
make errors                                              # error taxonomy
```

The subset is deterministic: 100 identities, up to 16 frames per tracklet, 150 distractor tracklets, seed 255
(`src/cctv_reid/mars_names.py::select_frames`). Dry run should report 681089 images in the zip and 19776 selected.

Outputs: `results/<run>/results.csv`, `run_config.json`, and `errors/*.md|json`.

## 4. Protocol (state this in every table caption)

- "MARS-style" protocol, **not** the official MARS query/gallery lists: camera-1 tracklets are queries, other cameras are the gallery, same-identity-same-camera matches excluded, pid 0 distractors kept in the gallery, pid -1 junk dropped.
- Clustering thresholds are tuned on a validation half of identities and reported on the other half.
- All clustering metrics are frame-level so methods are comparable.
- MARS has no tracking labels, so IDF1 / ID switches are not reported on it (see issue #20).

## 5. Experiment log (fill one row per run)

| Run | Date | Encoder | Pooling | Command | Commit | Notes |
|---|---|---|---|---|---|---|
| mars_resnet50_frozen | Nov 3 | ImageNet ResNet-50 (frozen) | frame / mean / attention / quality | `make mars` | main | Baseline: pooling raised mAP 0.157 to 0.272 |

## 6. Responsible use

Public benchmark data only in the repository. Any authorized CCTV clip stays local, is never uploaded, and is used for qualitative demonstration only. Matches are leads for human review, not verified identities.
