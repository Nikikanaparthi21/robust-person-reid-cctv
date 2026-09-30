# Robust Person Re-Identification and Automatic Identity Clustering in CCTV Footage

DATA255 deep learning course project.

## Research question

Can tracklet-level deep appearance embeddings and clustering identify and merge repeated appearances of the same person more accurately than frame-level embeddings in low-quality CCTV footage?

## Pipeline

Person detection -> multi-object tracking -> appearance embeddings -> frame/tracklet aggregation -> clustering -> retrieval timestamps.

## Scope

This is a research prototype comparing frame-level and tracklet-level representations under blur, lighting changes, occlusion, pose changes, and different sampling rates. It is not intended for unrestricted surveillance or automated decisions about people.

## Repository workflow

Use one GitHub Issue per task, assign an owner, create a feature branch, and open a pull request before merging into main. Record the dataset, split, configuration, seed, metrics, and runtime for every experiment.

## Privacy

Use public benchmark data or authorized footage. Do not commit raw CCTV video, face crops, credentials, or large model weights.
