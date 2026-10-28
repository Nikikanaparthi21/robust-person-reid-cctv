# Source layout (`src/cctv_reid`)

| Module | Purpose | Tested |
|---|---|---|
| `io_utils.py` | seeds, hashes, JSON manifests, MARS file-name parsing | yes |
| `detection.py` | normalized person-detection record | yes |
| `tracking.py` | ByteTrack-style two-stage IoU tracker (simplified, no Kalman filter) | yes, on synthetic boxes |
| `embeddings.py` | pooling (mean, quality-filtered, quality-weighted, attention); OSNet wrapper | pooling yes; OSNet wrapper no (needs torch + weights) |
| `clustering.py` | DBSCAN / agglomerative on cosine distance, threshold tuning on validation | yes |
| `evaluation.py` | rank-1/5, mAP, pairwise P/R/F1, purity, ARI, timing | yes |
| `degradations.py` | blur, darkening, low resolution, occlusion, frame subsampling | yes |
| `synthetic.py` | synthetic embedding benchmark for pipeline sanity only | yes |
| `mars_eval.py` | MARS-style evaluation of precomputed embeddings | yes, on MARS-named fake features |

Scripts: `scripts/extract_embeddings.py` (untested, needs MARS + torchreid), `scripts/evaluate_features.py`, `scripts/run_synthetic.py`.

Run tests: `python -m pytest -q`. Synthetic results live in `results/synthetic*/` and are NOT evidence about real CCTV or MARS.
Keep modules deterministic where possible and expose configuration through YAML rather than hard-coded paths.
