# Nikhil's Literature Survey: Robust CCTV Person Re-Identification

Project: Robust Person Re-Identification and Automatic Identity Clustering in Long-Duration CCTV Footage

## Survey purpose

This survey reviews 15 recognized publications connected directly to our pipeline: pedestrian detection, multi-object tracking, ReID embeddings, tracklets, temporal and quality-aware pooling, unsupervised clustering, retrieval, robustness, and evaluation. The project asks whether tracklet-level and quality-aware representations improve anonymous identity grouping over single-frame embeddings in low-quality CCTV footage.

## Selected publications and project relevance

1. Zheng et al. (2016), MARS, ECCV. Introduced a large-scale video ReID benchmark with automatically generated tracklets. We will use MARS for the main frame-versus-tracklet evaluation and will treat tracklet noise as an experimental issue.
2. Zhou, Yang, Cavallaro, and Xiang (2019), Omni-Scale Feature Learning for Person Re-Identification, ICCV. Proposed OSNet, a compact multi-scale appearance backbone. OSNet is our practical pretrained embedding baseline.
3. Li et al. (2020), Temporal Aggregation with Clip-Level Attention for Video-Based Person Re-Identification, WACV. Shows that informative clips should receive more weight than redundant frames. This motivates our quality-weighted pooling comparison.
4. Zhang et al. (2020), Multi-Granularity Reference-Aided Attentive Feature Aggregation for Video-Based Person Re-Identification, CVPR. Aggregates video features at multiple granularities and supports testing attention or frame-quality weighting.
5. Yang et al. (2020), Spatial-Temporal Graph Convolutional Network for Video-Based Person Re-Identification, CVPR. Models relationships among body regions and frames. It provides a future direction if simple pooling is not sufficient.
6. Zhong, Zheng, Li, and Yang (2020), Hierarchical Clustering with Hard-Batch Triplet Loss for Person Re-Identification, CVPR. Demonstrates the importance of hierarchical clustering and pseudo-label quality. We will compare hierarchical clustering with DBSCAN.
7. Ge, Chen, and Li (2020), Self-Paced Contrastive Learning with Hybrid Memory for Domain Adaptive Person Re-Identification, NeurIPS. Shows how self-paced learning can stabilize unsupervised representations. It motivates measuring cluster confidence before optional adaptation.
8. He et al. (2021), TransReID: Transformer-Based Object Re-Identification, ICCV. Provides a stronger transformer-based embedding alternative. It is an optional comparison after the reproducible OSNet baseline.
9. Wang et al. (2021), Pyramid Spatial-Temporal Aggregation for Video-Based Person Re-Identification, ICCV. Uses multi-level spatial-temporal aggregation and supports the hypothesis that temporal context helps under occlusion and blur.
10. Chen, Lagadec, and Bremond (2021), ICE: Inter-Instance Contrastive Encoding for Unsupervised Person Re-Identification, ICCV. Uses hard and soft similarity relationships to improve unsupervised ReID and motivates robustness testing.
11. Xie, Xu, Wang, and Yamasaki (2021), Unsupervised Video Person Re-Identification via Noise and Hard Frame Aware Clustering, ICME. Explicitly models noisy and difficult frames, directly supporting our frame-quality scoring and hard-frame ablations.
12. Du et al. (2023), StrongSORT: Make DeepSORT Great Again, IEEE Transactions on Multimedia, 25, 8725-8737. Improves association using detection, appearance, motion, and track management. It is a strong optional comparison with ByteTrack.
13. Zhang et al. (2022), ByteTrack: Multi-Object Tracking by Associating Every Detection Box, ECCV. Recovers useful trajectories from low-confidence detections and forms the first tracking stage of our pipeline.
14. Dai, Wang, Yuan, Zhu, and Tan (2022), Cluster Contrast for Unsupervised Person Re-Identification, ACCV, 1142-1160. Uses cluster-level memory and contrastive learning, reinforcing that clustering quality controls unsupervised identity learning.
15. Meng, Wang, Li, and Xue (2025), Unleashing the Potential of Tracklets for Unsupervised Video Person Re-Identification, IEEE Transactions on Information Forensics and Security. Filters noisy frames, partitions tracklets, refines clusters, and is the closest recent publication to our quality-aware tracklet backbone.

## Synthesis and research gap

The literature shows three consistent findings. First, tracking quality limits ReID quality: missed or fragmented detections cannot be repaired later. Second, a tracklet contains useful temporal information but also blur, occlusion, redundant frames, and association errors, so uniform pooling should be compared with quality-aware pooling. Third, clustering errors can become pseudo-label errors; therefore DBSCAN, hierarchical clustering, cluster count, outlier rate, silhouette score, and retrieval metrics should be reported together.

Recent papers often use supervised training, large compute, or iterative pseudo-label learning. Our contribution is a transparent and reproducible baseline that isolates the value of tracklet aggregation and quality filtering. We will compare: (1) frame-level OSNet embeddings, (2) mean-pooled tracklet embeddings, and (3) quality-filtered or quality-weighted tracklet embeddings. MARS will provide the main quantitative evaluation; authorized CCTV footage will be used only for an anonymous qualitative demonstration and failure analysis.

## Recommended experiments

- ByteTrack first; StrongSORT as an optional comparison.
- Pretrained OSNet embeddings with cosine normalization.
- Store crops, embeddings, bounding boxes, confidence, frame index, and track ID in a manifest.
- Compare uniform sampling, mean pooling, and quality-weighted pooling.
- Test blur, darkness, occlusion, low resolution, tracklet length, and sampling rate.
- Compare DBSCAN and agglomerative clustering with the same embeddings.
- Report Rank-1/Rank-5, mAP where applicable, ARI/NMI when labels are available, silhouette score, fragmentation, ID switches, and runtime.
- Keep identities anonymous and do not infer names or sensitive attributes.

## Official source links

MARS: https://www.ecva.net/papers/eccv_2016/papers_ECCV/html/9801_ECCV_2016_paper.php
OSNet: https://openaccess.thecvf.com/content_ICCV_2019/html/Zhou_Omni-Scale_Feature_Learning_for_Person_Re-Identification_ICCV_2019_paper.html
Temporal aggregation: https://openaccess.thecvf.com/content_WACV_2020/html/Li_Temporal_Aggregation_with_Clip-level_Attention_for_Video-based_Person_Re-identification_WACV_2020_paper.html
HCT: https://openaccess.thecvf.com/content_CVPR_2020/html/Zhong_Hierarchical_Clustering_With_Hard-Batch_Triplet_Loss_for_Person_Re-Identification_CVPR_2020_paper.html
ICE: https://openaccess.thecvf.com/ICCV2021/html/Chen_ICE_Inter-Instance_Contrastive_Encoding_for_Unsupervised_Person_Re-Identification_ICCV_2021_paper.html
PSTA: https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Pyramid_Spatial-Temporal_Aggregation_for_Video-Based_Person_Re-Identification_ICCV_2021_paper.html
ByteTrack: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf
Cluster Contrast: https://openaccess.thecvf.com/content/ACCV2022/html/Dai_Cluster_Contrast_for_Unsupervised_Person_Re-Identification_ACCV_2022_paper.html
StrongSORT: https://doi.org/10.1109/TMM.2023.3240881
SSR-C: https://ieeexplore.ieee.org/abstract/document/11314528

Status: This is the report-ready literature-survey draft for GitHub issue #9. Before PDF submission, verify complete APA metadata, DOI, volume, issue, and page numbers from the official publisher pages and select the required 13-paper subset for the intermediate report.
