# Robust Person Re-Identification and Automatic Identity Clustering in CCTV Footage

## Abstract

Long-duration CCTV footage produces many repeated detections of the same person. This project studies whether temporal tracklets and deep appearance embeddings can automatically group repeated appearances more accurately than independent frame embeddings. The output will be identity clusters, representative samples, similarity scores, and source timestamps for human review.

## Research question

Can tracklet-level deep appearance embeddings and clustering identify and merge repeated appearances of the same person more accurately than frame-level embeddings in low-quality CCTV footage?

## Methodology

The pipeline is:

1. Person detection.
2. Multi-object tracking with ByteTrack or an equivalent tracker.
3. Appearance embeddings using a documented pretrained person-ReID encoder such as OSNet.
4. Frame-level baseline and tracklet aggregation using mean or quality-filtered pooling.
5. Cosine similarity with DBSCAN and hierarchical clustering.
6. Retrieval results, representative galleries, and timestamps.

Quantitative evaluation will use a public video person-ReID benchmark such as MARS or DukeMTMC-VideoReID, subject to access and licensing. An authorized CCTV clip will be used only for qualitative demonstration and will not be uploaded.

## Experiments and evaluation

We will compare frame-level versus tracklet-level embeddings, mean versus quality-filtered pooling, clean versus blurred/dark/low-resolution/partially occluded footage, and higher versus lower sampling rates. Metrics will include rank-1/rank-5 accuracy, mean average precision where supported, pairwise precision/recall/F1, cluster purity, adjusted Rand index, ID switches, IDF1 where annotations are available, processing time, throughput, and memory use. Error analysis will document false merges, false splits, missed detections, and occlusion failures.

## Deliverables and milestones

- Week 1: literature review, dataset decision, repository, and baseline specification.
- Week 2: detector, tracker, tracklet extraction, and data manifest.
- Week 3: frame-level and tracklet-level embedding baseline.
- Week 4: clustering, retrieval outputs, and evaluation scripts.
- Week 5: robustness ablations and error analysis.
- Week 6: figures, report, presentation, and optional paper-style manuscript.

## Team roles

- Nikhil Kanaparthi: coordination, repository, preprocessing, and detection/tracking integration.
- Team member 2: embedding extraction, tracklet aggregation, and clustering.
- Team member 3: dataset preparation, evaluation, robustness experiments, and visualizations.

All members will review pull requests, reproduce an experiment, and contribute to the report and presentation.

## Responsible use

Use public benchmark data or authorized footage. Do not commit raw CCTV video, face crops, credentials, or large model weights. Matches are not verified identities and require human review. The report will discuss privacy, false matches, false rejections, and camera/domain bias.

## References

- Meng et al., “Unleashing the Potential of Tracklets for Unsupervised Video Person Re-Identification,” arXiv:2406.14261.
- Nguyen et al., “AG-VPReID: A Challenging Large-Scale Benchmark for Aerial-Ground Video-based Person Re-Identification,” CVPR 2025.
- Zhu et al., “SEAS: ShapE-Aligned Supervision for Person Re-Identification,” CVPR 2024.
