# Robust Person Re-Identification and Automatic Identity Clustering in CCTV Footage

DATA255 deep learning course project.

## Research question

Can tracklet-level deep appearance embeddings and clustering identify and merge repeated appearances of the same person more accurately than frame-level embeddings in low-quality CCTV footage?

## Pipeline

Person detection -> multi-object tracking -> appearance embeddings -> frame/tracklet aggregation -> clustering -> retrieval timestamps.

## Status (Nov 4, 2026)

Working: MARS subset selection, embedding extraction, frame vs tracklet pooling (mean, quality-filtered, quality-weighted, attention), DBSCAN and agglomerative clustering, retrieval and clustering metrics, IoU tracker with tracklet manifests, robustness degradations, error analysis. Frozen ResNet-50 baseline on MARS: tracklet pooling raises mAP from 0.157 to 0.272.
In progress: OSNet fine-tuning, robustness runs on real data, retrieval viewer.

## Quick start

```bash
make setup && make test        # see docs/reproducibility.md for the full guide
make mars ZIP=/path/to/bbox_test.zip DEVICE=mps
```

## Scope

This is a research prototype comparing frame-level and tracklet-level representations under blur, lighting changes, occlusion, pose changes, and different sampling rates. It is not intended for unrestricted surveillance or automated decisions about people.

## Repository workflow

Use one GitHub Issue per task, assign an owner, create a feature branch, and open a pull request before merging into main. Record the dataset, split, configuration, seed, metrics, and runtime for every experiment.

## Privacy

Use public benchmark data or authorized footage. Do not commit raw CCTV video, face crops, credentials, or large model weights.
