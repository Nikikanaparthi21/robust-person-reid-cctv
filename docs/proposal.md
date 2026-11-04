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

Dates are aligned with the GitHub milestones (Intermediate Report Oct 28, Midterm Nov 3, Final Nov 10).

| Date | Milestone | Evidence |
|---|---|---|
| Oct 6 | Literature review, dataset decision (MARS), repository, baseline specification | Dataset note, `configs/dataset.yaml` |
| Oct 13 | Detector, tracker, tracklet extraction, manifest | Tracklet manifest schema, tracker tests (PR #33) |
| Oct 20 | Frame-level and tracklet-level embedding baseline | Pooling methods, unit tests (PR #31) |
| Oct 27-28 | Clustering, retrieval, evaluation; **Intermediate report** | Real MARS results, status report |
| Nov 3 | **Midterm check**: fine-tuned OSNet, attention pooling on real data, robustness runs | Comparison table, robustness curves |
| Nov 7 | Results frozen | Final tables and figures |
| Nov 10 | **Final**: error analysis, report, presentation, demo, reproducibility package | Final report, slides, tagged release |

## Heilmeier checklist

| Question | Answer |
|---|---|
| What are we trying to do? | Group repeated appearances of the same person in CCTV-style footage into identity clusters for human review. |
| How is it done today and what are the limits? | Frame-by-frame matching, which is noisy under blur, darkness, low resolution and occlusion. |
| What is new and why might it work? | Pool embeddings over tracklets (temporal aggregation, including learned attention) and cluster with validation-tuned thresholds. |
| Who cares? | Security reviewers and investigators who must search long footage; success reduces review time. |
| Risks | False merges link different people, false splits duplicate them; privacy and camera/domain bias. Outputs are leads, not verified identities. |
| Cost | No paid data or services: public MARS dataset, free tooling, existing laptops plus one GPU session (Colab or university) for fine-tuning; about 6 weeks of three students' time. |
| Duration | Six weeks, Sep 30 to Nov 10, 2026. |
| Midterm exam of success (Nov 3) | Fine-tuned OSNet and pooling comparison on real MARS; tracklet pooling beats the frame baseline in mAP and clustering F1; robustness runs started. |
| Final exam of success (Nov 10) | Reproducible pipeline, robustness and error analysis, report, slides and demo; all results regenerate from one documented command. |

## Responsible use

Use public benchmark data or authorized footage. Do not commit raw CCTV video, face crops, credentials, or large model weights. Matches are not verified identities and require human review. The report will discuss privacy, false matches, false rejections, and camera/domain bias.

## References

- Meng et al., “Unleashing the Potential of Tracklets for Unsupervised Video Person Re-Identification,” arXiv:2406.14261.
- Nguyen et al., “AG-VPReID: A Challenging Large-Scale Benchmark for Aerial-Ground Video-based Person Re-Identification,” CVPR 2025.
- Zhu et al., “SEAS: ShapE-Aligned Supervision for Person Re-Identification,” CVPR 2024.
