# Abhijith's Literature Survey: Robust CCTV Person Re-Identification

## Project connection

This literature survey reviews 15 peer-reviewed publications related to the project pipeline: person detection, multi-object tracking, person-ReID embeddings, tracklet generation, temporal aggregation, quality filtering, clustering, retrieval, robustness, and evaluation.

The project investigates whether tracklet-level and quality-aware representations can improve identity grouping compared with independent frame-level embeddings in low-quality CCTV footage.

## Literature review

### Paper 1: MARS: A Video Benchmark for Large-Scale Person Re-identification

**APA citation:**  

Zheng, L., Bie, Z., Sun, Y., Wang, J., Su, C., Wang, S., & Tian, Q. (2016). MARS: A video benchmark for large-scale person re-identification. In B. Leibe, J. Matas, N. Sebe, & M. Welling (Eds.), *Computer vision – ECCV 2016* (pp. 868–884). Springer. https://doi.org/10.1007/978-3-319-46466-4_52

**Official source:**  

https://www.microsoft.com/en-us/research/publication/mars-video-benchmark-large-scale-person-re-identification/

**Publication details:**

- Authors: Liang Zheng, Zhi Bie, Yifan Sun, Jingdong Wang, Chi Su, Shengjin Wang, and Qi Tian
- Year: 2016
- Venue: European Conference on Computer Vision, ECCV
- Pipeline stage: Dataset, tracklets, video person re-identification, and evaluation

**Research problem:**  

Image-based person re-identification does not fully capture the temporal information available in video. The paper introduces a large-scale video person-ReID benchmark for evaluating whether information from multiple frames can improve identity matching across different camera views.

**Method:**  

The authors introduce the MARS dataset, which contains automatically generated pedestrian tracklets collected from multiple synchronized cameras. The tracklets are produced using a pedestrian detector and a tracking system. The dataset includes both genuine pedestrian trajectories and distractor tracklets, making the benchmark closer to realistic surveillance conditions.

**Main contribution:**  

The paper provides a large-scale benchmark for video-based person re-identification and establishes evaluation procedures for comparing identity retrieval methods. It demonstrates that deep visual representations learned from consecutive pedestrian frames can provide useful information for matching people across cameras.

**Connection to this project:**  

MARS is the primary benchmark selected for this project. Its tracklets allow us to compare independent frame-level embeddings with aggregated tracklet-level embeddings. The dataset also supports analysis of tracking noise, viewpoint changes, lighting variation, and distractor detections.

This paper directly supports the dataset preparation, tracklet generation, embedding aggregation, retrieval, and evaluation stages of our pipeline.

### Paper 2: Omni-Scale Feature Learning for Person Re-Identification

**APA citation:**

Zhou, K., Yang, Y., Cavallaro, A., & Xiang, T. (2019). Omni-scale feature learning for person re-identification. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)* (pp. 3702–3711). Computer Vision Foundation.

**Official source:**

https://openaccess.thecvf.com/content_ICCV_2019/html/Zhou_Omni-Scale_Feature_Learning_for_Person_Re-Identification_ICCV_2019_paper.html

**Publication details:**

- Authors: Kaiyang Zhou, Yongxin Yang, Andrea Cavallaro, and Tao Xiang
- Year: 2019
- Venue: IEEE/CVF International Conference on Computer Vision, ICCV
- Pipeline stage: Person-ReID embeddings

**Research problem:**

Person-ReID systems must distinguish people who may look similar while handling large appearance changes caused by camera viewpoints, pose, lighting, and scale. Features based only on a single spatial scale may miss either fine details or broader information about a person.

**Method:**

The authors propose the Omni-Scale Network, or OSNet. The network uses multiple feature streams to learn visual information at different spatial scales. It also uses an aggregation mechanism that dynamically combines these features according to the input image.

**Main contribution:**

OSNet provides a lightweight person-ReID backbone that captures both local details and global appearance information. The model is designed to provide strong ReID representations while using fewer computational resources than larger networks.

**Connection to this project:**

OSNet is a suitable pretrained encoder for the frame-level embedding baseline in this project. We can extract an embedding for each detected person crop and then compare independent frame embeddings with mean-pooled and quality-filtered tracklet embeddings.

This paper supports the person-ReID embedding stage and the comparison between frame-level and tracklet-level representations.

### Paper 3: Temporal Aggregation with Clip-level Attention for Video-based Person Re-identification

**APA citation:**

Li, M., Xu, H., Wang, J., Li, W., & Sun, Y. (2020). Temporal aggregation with clip-level attention for video-based person re-identification. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)* (pp. 3376–3384).

**Official source:**

https://openaccess.thecvf.com/content_WACV_2020/html/Li_Temporal_Aggregation_with_Clip-level_Attention_for_Video-based_Person_Re-identification_WACV_2020_paper.html

**Publication details:**

- Authors: Mengliu Li, Han Xu, Jinjun Wang, Wenpeng Li, and Yongli Sun
- Year: 2020
- Venue: IEEE/CVF Winter Conference on Applications of Computer Vision, WACV
- Pipeline stage: Temporal aggregation and tracklet representation

**Research problem:**

A tracklet may contain redundant, blurred, occluded, or low-quality frames. Simple average or maximum pooling treats every frame equally and may allow poor frames to reduce the quality of the final tracklet representation.

**Method:**

The authors propose Temporal Aggregation with Clip-level Attention Network. A tracklet is divided into clips, and the method learns attention weights for the clips. The weighted clip representations are then combined into a robust tracklet-level feature.

**Main contribution:**

The paper shows that attention-based temporal aggregation can select more informative portions of a video instead of treating all frames equally.

**Connection to this project:**

This paper directly motivates the comparison between mean pooling and quality-aware or weighted pooling. Our project can use detection confidence, blur, crop quality, and temporal consistency to identify more useful frames before constructing a tracklet embedding.

---

### Paper 4: Multi-Granularity Reference-Aided Attentive Feature Aggregation for Video-Based Person Re-Identification

**APA citation:**

Zhang, Z., Lan, C., Zeng, W., & Chen, Z. (2020). Multi-granularity reference-aided attentive feature aggregation for video-based person re-identification. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 10407–10416).

**Official source:**

https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Multi-Granularity_Reference-Aided_Attentive_Feature_Aggregation_for_Video-Based_Person_Re-Identification_CVPR_2020_paper.html

**Publication details:**

- Authors: Zhizheng Zhang, Cuiling Lan, Wenjun Zeng, and Zhibo Chen
- Year: 2020
- Venue: IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR
- Pipeline stage: Spatial-temporal feature aggregation

**Research problem:**

Video frames contain different levels of useful information. Some frames may provide detailed clothing cues, while other frames may provide broader body-shape or pose information. A single aggregation scale may not capture all useful information.

**Method:**

The paper introduces a multi-granularity reference-aided attention module. It compares feature nodes with representative reference features and learns attention at different spatial and temporal granularities.

**Main contribution:**

The method produces a more discriminative video-level representation by combining information from multiple feature granularities.

**Connection to this project:**

This work supports the idea that aggregation should consider more than a simple average. It provides a theoretical basis for testing whether local crop quality and broader tracklet-level information should receive different weights.

---

### Paper 5: Spatial-Temporal Graph Convolutional Network for Video-Based Person Re-Identification

**APA citation:**

Yang, J., Zheng, W.-S., Yang, Q., Chen, Y.-C., & Tian, Q. (2020). Spatial-temporal graph convolutional network for video-based person re-identification. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 3289–3299).

**Official source:**

https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Spatial-Temporal_Graph_Convolutional_Network_for_Video-Based_Person_Re-Identification_CVPR_2020_paper.html

**Publication details:**

- Authors: Jinrui Yang, Wei-Shi Zheng, Qize Yang, Ying-Cong Chen, and Qi Tian
- Year: 2020
- Venue: IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR
- Pipeline stage: Spatial-temporal representation learning

**Research problem:**

People may be partially occluded or visually similar to other individuals. Information from only one frame may therefore be insufficient for reliable identity matching.

**Method:**

The authors use graph convolutional networks to model spatial relationships within a person image and temporal relationships between consecutive frames.

**Main contribution:**

The method combines body-structure information and temporal information to create more robust video person-ReID features.

**Connection to this project:**

Although our initial system uses a simpler pretrained encoder and pooling strategy, this paper explains why temporal context and relationships between frames can improve ReID. It also motivates future extensions beyond mean pooling.

---

### Paper 6: Hierarchical Clustering With Hard-Batch Triplet Loss for Person Re-Identification

**APA citation:**

Zeng, K., Ning, M., Wang, Y., & Guo, Y. (2020). Hierarchical clustering with hard-batch triplet loss for person re-identification. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 13657–13665).

**Official source:**

https://openaccess.thecvf.com/content_CVPR_2020/html/Zeng_Hierarchical_Clustering_With_Hard-Batch_Triplet_Loss_for_Person_Re-Identification_CVPR_2020_paper.html

**Publication details:**

- Authors: Kaiwei Zeng, Munan Ning, Yaohua Wang, and Yang Guo
- Year: 2020
- Venue: IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR
- Pipeline stage: Clustering and pseudo-label generation

**Research problem:**

Unsupervised person-ReID systems must create labels from visual similarity. Poor clusters create incorrect pseudo-labels, which can damage later model training.

**Method:**

The method uses hierarchical clustering to generate pseudo-labels and hard-batch triplet loss to improve the learned representation.

**Main contribution:**

The paper demonstrates that clustering quality has a direct effect on unsupervised person-ReID performance.

**Connection to this project:**

Our project compares DBSCAN and agglomerative clustering. This paper supports reporting cluster quality, outlier rates, and clustering errors instead of treating the clustering output as automatically correct.

---

### Paper 7: Self-Paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID

**APA citation:**

Ge, Y., Zhu, F., Chen, D., Zhao, R., & Li, H. (2020). Self-paced contrastive learning with hybrid memory for domain adaptive object re-ID. In *Advances in Neural Information Processing Systems, 33*.

**Official source:**

https://papers.neurips.cc/paper_files/paper/2020/hash/821fa74b50ba3f7cba1e6c53e8fa6845-Abstract.html

**Publication details:**

- Authors: Yixiao Ge, Feng Zhu, Dapeng Chen, Rui Zhao, and Hongsheng Li
- Year: 2020
- Venue: Advances in Neural Information Processing Systems, NeurIPS
- Pipeline stage: Unsupervised learning and cluster refinement

**Research problem:**

Unlabeled target data often produces noisy clusters and unreliable pseudo-labels. Training directly on all pseudo-labels can make the representation less reliable.

**Method:**

The authors propose a hybrid memory containing source-domain classes, target-domain clusters, and unclustered instances. A self-paced strategy gradually uses more reliable clusters during training.

**Main contribution:**

The method shows how cluster reliability and contrastive learning can be combined for unsupervised person and object ReID.

**Connection to this project:**

This paper supports the idea of measuring cluster confidence before using cluster assignments for later analysis. It is relevant to our use of DBSCAN, hierarchical clustering, outlier handling, and pseudo-label quality.

---

### Paper 8: TransReID: Transformer-Based Object Re-Identification

**APA citation:**

He, S., Luo, H., Wang, P., Wang, F., Li, H., & Jiang, W. (2021). TransReID: Transformer-based object re-identification. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)* (pp. 15013–15022).

**Official source:**

https://openaccess.thecvf.com/content/ICCV2021/html/He_TransReID_Transformer-Based_Object_Re-Identification_ICCV_2021_paper.html

**Publication details:**

- Authors: Shuting He, Hao Luo, Pichao Wang, Fan Wang, Hao Li, and Wei Jiang
- Year: 2021
- Venue: IEEE/CVF International Conference on Computer Vision, ICCV
- Pipeline stage: Person-ReID embeddings

**Research problem:**

CNN-based ReID models may lose global relationships and detailed information because of local convolution and downsampling operations.

**Method:**

TransReID uses a transformer-based architecture that represents an image as a sequence of patches. It also introduces a jigsaw patch module and side-information embeddings for camera and viewpoint information.

**Main contribution:**

The paper establishes a transformer-based ReID approach that models wider relationships across image regions and incorporates camera-related information.

**Connection to this project:**

TransReID provides a stronger optional comparison with the OSNet baseline. It is useful for analyzing whether improvements come from the aggregation method or from using a more powerful embedding encoder.

---

### Paper 9: Pyramid Spatial-Temporal Aggregation for Video-Based Person Re-Identification

**APA citation:**

Wang, Y., Zhang, P., Gao, S., Geng, X., Lu, H., & Wang, D. (2021). Pyramid spatial-temporal aggregation for video-based person re-identification. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)* (pp. 12026–12035).

**Official source:**

https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Pyramid_Spatial-Temporal_Aggregation_for_Video-Based_Person_Re-Identification_ICCV_2021_paper.html

**Publication details:**

- Authors: Yingquan Wang, Pingping Zhang, Shang Gao, Xia Geng, Hu Lu, and Dong Wang
- Year: 2021
- Venue: IEEE/CVF International Conference on Computer Vision, ICCV
- Pipeline stage: Temporal and spatial aggregation

**Research problem:**

Short-term frame information and long-term tracklet information can provide different clues for identity matching. A single aggregation level may fail when occlusion or appearance changes occur.

**Method:**

The PSTA framework progressively aggregates frame-level features at multiple temporal levels. It uses spatial and temporal reference attention to suppress interference and emphasize useful features.

**Main contribution:**

The paper shows how hierarchical temporal aggregation can combine short-term and long-term information in video person-ReID.

**Connection to this project:**

This work supports experiments involving tracklet length, sampling rate, temporal context, and quality-aware aggregation. It provides a more advanced reference point for comparison with our simpler mean-pooling baseline.

---

### Paper 10: ICE: Inter-Instance Contrastive Encoding for Unsupervised Person Re-Identification

**APA citation:**

Chen, H., Lagadec, B., & Bremond, F. (2021). ICE: Inter-instance contrastive encoding for unsupervised person re-identification. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)* (pp. 14960–14969).

**Official source:**

https://openaccess.thecvf.com/content/ICCV2021/html/Chen_ICE_Inter-Instance_Contrastive_Encoding_for_Unsupervised_Person_Re-Identification_ICCV_2021_paper.html

**Publication details:**

- Authors: Hao Chen, Benoit Lagadec, and François Bremond
- Year: 2021
- Venue: IEEE/CVF International Conference on Computer Vision, ICCV
- Pipeline stage: Unsupervised representation learning

**Research problem:**

Unsupervised ReID methods must learn identity-discriminative features without human identity labels. Treating every image independently may ignore relationships between different samples of the same person.

**Method:**

ICE uses inter-instance similarity scores as hard and soft pseudo-label information. The method combines class-level contrast, hard-instance contrast, and consistency between augmented views.

**Main contribution:**

The paper shows that pairwise relationships between instances can improve unsupervised ReID and reduce the effect of noisy labels and augmentations.

**Connection to this project:**

This paper is relevant to identity clustering and embedding robustness. It supports evaluating whether better similarity relationships lead to more accurate clusters and retrieval results.

---

### Paper 11: Unsupervised Video Person Re-identification via Noise and Hard Frame Aware Clustering

**APA citation:**

Xie, P., Xu, X., Wang, Z., & Yamasaki, T. (2021). Unsupervised video person re-identification via noise and hard frame aware clustering. In *Proceedings of the IEEE International Conference on Multimedia and Expo (ICME)*.

**Official source:**

https://arxiv.org/abs/2106.05441

**Publication details:**

- Authors: Pengyu Xie, Xin Xu, Zheng Wang, and Toshihiko Yamasaki
- Year: 2021
- Venue: IEEE International Conference on Multimedia and Expo, ICME
- Pipeline stage: Tracklet quality filtering and unsupervised clustering

**Research problem:**

Video tracklets may contain two different types of difficult frames: noisy frames caused by detection errors or heavy occlusion, and hard frames that are difficult to identify but may contain useful information.

**Method:**

The method introduces graph trimming to remove unreliable frame nodes and node resampling to give more attention to difficult but informative frames.

**Main contribution:**

The paper separates noisy frames from hard frames instead of removing every difficult frame. This helps preserve useful information while reducing the effect of unreliable observations.

**Connection to this project:**

This paper directly supports our quality-filtering experiments. We should distinguish between frames that are genuinely corrupted and frames that are merely difficult, then report how filtering changes tracklet embeddings and clustering performance.

---

### Paper 12: StrongSORT: Make DeepSORT Great Again

**APA citation:**

Du, Y., Zhao, Z., Song, Y., Zhao, Y., Su, F., Gong, T., & Meng, H. (2023). StrongSORT: Make DeepSORT great again. *IEEE Transactions on Multimedia, 25*, 8725–8737. https://doi.org/10.1109/TMM.2023.3240881

**Official source:**

https://doi.org/10.1109/TMM.2023.3240881

**Publication details:**

- Authors: Yutao Du, Ziqi Zhao, Yifan Song, Yichao Zhao, Fuxin Su, Tianyu Gong, and Hongsheng Meng
- Year: 2023
- Venue: IEEE Transactions on Multimedia
- Pipeline stage: Multi-object tracking and tracklet association

**Research problem:**

Tracking-by-detection systems may suffer from missed detections, fragmented trajectories, and incorrect identity associations.

**Method:**

StrongSORT revisits DeepSORT and improves its detection, appearance embedding, and association components. It also introduces AFLink for connecting short tracklets and Gaussian-smoothed interpolation for addressing missing detections.

**Main contribution:**

The paper provides a stronger tracking baseline and addresses both missing associations and missing detections.

**Connection to this project:**

StrongSORT is an optional comparison with the selected ByteTrack tracker. It is especially relevant when evaluating tracklet fragmentation, identity switches, missing detections, and the quality of the tracklet manifest.

---

### Paper 13: ByteTrack: Multi-Object Tracking by Associating Every Detection Box

**APA citation:**

Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., & Wang, X. (2022). ByteTrack: Multi-object tracking by associating every detection box. In *Computer Vision – ECCV 2022* (pp. 1–21). Springer.

**Official source:**

https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/315_ECCV_2022_paper.php

**Publication details:**

- Authors: Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, and Xinggang Wang
- Year: 2022
- Venue: European Conference on Computer Vision, ECCV
- Pipeline stage: Person detection association and tracking

**Research problem:**

Many tracking systems discard low-confidence detection boxes. However, low-confidence boxes may represent partially occluded or visually difficult people, and removing them can produce fragmented trajectories.

**Method:**

ByteTrack associates nearly every detection box. It uses high-confidence detections first and then uses lower-confidence detections to recover true objects while filtering background detections.

**Main contribution:**

The method shows that carefully associating low-confidence detections can improve identity continuity and reduce missed trajectories.

**Connection to this project:**

ByteTrack is the tracker selected in the project configuration. It will generate the tracklet manifest used by the ReID embedding stage. Its behavior should be evaluated using tracklet length, fragmentation, identity switches, confidence scores, and missed detections.

---

### Paper 14: Cluster Contrast for Unsupervised Person Re-Identification

**APA citation:**

Dai, Z., Wang, G., Yuan, W., Zhu, S., & Tan, P. (2022). Cluster contrast for unsupervised person re-identification. In *Proceedings of the Asian Conference on Computer Vision (ACCV)* (pp. 1142–1160).

**Official source:**

https://openaccess.thecvf.com/content/ACCV2022/html/Dai_Cluster_Contrast_for_Unsupervised_Person_Re-Identification_ACCV_2022_paper.html

**Publication details:**

- Authors: Zuozhuo Dai, Guangyuan Wang, Weihao Yuan, Siyu Zhu, and Ping Tan
- Year: 2022
- Venue: Asian Conference on Computer Vision, ACCV
- Pipeline stage: Clustering and contrastive representation learning

**Research problem:**

Unsupervised ReID systems often assign pseudo-labels using clustering, but the feature dictionary may become inconsistent as the network changes during training.

**Method:**

Cluster Contrast stores feature vectors at the cluster level and applies contrastive learning between cluster representations. A momentum update is used to maintain more consistent cluster features.

**Main contribution:**

The work demonstrates that cluster-level contrastive learning can improve unsupervised person-ReID representations.

**Connection to this project:**

The paper supports the project’s focus on cluster quality. It suggests that identity grouping should be evaluated using cluster-level measures such as purity, ARI, NMI, outlier rate, and retrieval accuracy.

---

### Paper 15: Unleashing the Potential of Tracklets for Unsupervised Video Person Re-Identification

**APA citation:**

Meng, N., Wang, Q., Li, B., & Xue, X. (2025). Unleashing the potential of tracklets for unsupervised video person re-identification. *IEEE Transactions on Information Forensics and Security, 21*, 431–445. https://doi.org/10.1109/TIFS.2025.3648202

**Official source:**

https://ieeexplore.ieee.org/abstract/document/11314528

**Publication details:**

- Authors: Nanxing Meng, Qizao Wang, Bin Li, and Xiangyang Xue
- Year: 2025
- Venue: IEEE Transactions on Information Forensics and Security
- Pipeline stage: Tracklet filtering, sub-tracklets, clustering, and unsupervised ReID

**Research problem:**

Tracklets contain useful temporal and spatial information, but simple averaging can be affected by noisy tracking results and inconsistent frames.

**Method:**

The paper proposes Self-Supervised Refined Clustering. It uses Noise-Filtered Tracklet Partition to reduce the effect of noisy frames, divides tracklets into sub-tracklets, and progressively clusters and merges the resulting representations.

**Main contribution:**

The work provides a recent framework that treats tracklet structure as an important source of self-supervision instead of simply averaging all frames.

**Connection to this project:**

This paper is closely related to our research question. It supports comparing mean pooling with quality-filtered and partitioned tracklet representations. It also provides a recent reference for analyzing how tracklet noise affects unsupervised identity clustering.




## Cross-paper synthesis

### Detection and tracking

The literature shows that reliable person detection and tracking are essential because errors at the beginning of the pipeline affect every later stage. ByteTrack demonstrates that low-confidence detections should not automatically be discarded because they may represent partially occluded people. StrongSORT further shows that association, appearance information, and missing-detection recovery can improve trajectory continuity.

For this project, tracking quality will be documented through the number of detections, tracklet length, fragmented trajectories, identity switches, confidence scores, and missing frames.

### Person-ReID embeddings

OSNet provides a lightweight multi-scale CNN baseline for extracting appearance embeddings from person crops. TransReID provides a stronger transformer-based alternative that captures broader relationships between image patches.

The initial project will use a documented pretrained encoder and will compare frame-level embeddings before considering more computationally expensive encoders.

### Tracklets and temporal aggregation

MARS provides tracklets for video-based person-ReID evaluation. The temporal aggregation papers show that all frames should not necessarily receive equal importance. Blur, occlusion, detection errors, and redundant frames can reduce the quality of a pooled representation.

This motivates the comparison of:

1. Individual frame embeddings
2. Mean-pooled tracklet embeddings
3. Quality-filtered or quality-weighted tracklet embeddings

### Quality filtering

The literature distinguishes between noisy frames and difficult but informative frames. Removing every difficult frame may discard useful identity information. A better strategy is to document the filtering rule and compare its effect on tracklet quality and identity retrieval.

The project will record detection confidence, crop quality, frame index, track ID, and any filtering decision.

### Clustering and retrieval

HCT, ICE, and Cluster Contrast show that clustering quality strongly influences unsupervised ReID. Incorrect clusters can create incorrect pseudo-labels and reduce the reliability of later retrieval.

The project will compare DBSCAN and agglomerative clustering using the same normalized embeddings. It will report cluster counts, outlier counts, cluster purity, adjusted Rand index, pairwise precision, pairwise recall, F1 score, and retrieval performance where ground-truth labels are available.

### Robustness and evaluation

The literature identifies viewpoint changes, lighting changes, blur, occlusion, low resolution, and visually similar people as major challenges. The project will evaluate these conditions through controlled robustness comparisons and failure analysis.

The planned metrics include:

- Rank-1 accuracy
- Rank-5 accuracy
- Mean average precision where applicable
- Pairwise precision
- Pairwise recall
- Pairwise F1 score
- Cluster purity
- Adjusted Rand index
- Silhouette score
- Tracklet fragmentation
- Identity switches
- Runtime

## Research gap

Existing studies often use supervised training, large computational resources, or iterative pseudo-label refinement. These methods can achieve strong performance but may make it difficult to isolate the individual contribution of tracklet aggregation and quality filtering.

This project addresses that gap through a transparent and reproducible baseline. It separates the contribution of:

1. The frame-level ReID encoder
2. Tracklet construction
3. Mean aggregation
4. Quality filtering
5. Clustering and retrieval

The main research question is whether quality-aware tracklet representations improve identity grouping compared with independent frame-level embeddings in low-quality CCTV footage.

## Relevance to the project pipeline

| Paper | Main pipeline stage | Direct project use |
|---|---|---|
| MARS | Dataset and tracklets | Main video person-ReID benchmark |
| OSNet | ReID embeddings | Lightweight frame-level baseline |
| TACAN | Temporal aggregation | Supports attention-based pooling comparison |
| MG-RAFA | Feature aggregation | Supports multi-granularity aggregation |
| STGCN | Spatial-temporal modeling | Future advanced temporal baseline |
| HCT | Clustering | Supports pseudo-label and cluster-quality analysis |
| SPCL | Unsupervised learning | Supports reliable cluster refinement |
| TransReID | ReID embeddings | Optional transformer comparison |
| PSTA | Temporal aggregation | Supports hierarchical temporal analysis |
| ICE | Unsupervised ReID | Supports similarity-based representation learning |
| NHAC | Quality filtering | Supports noisy and hard-frame analysis |
| StrongSORT | Tracking | Optional comparison with ByteTrack |
| ByteTrack | Tracking | Selected project tracking method |
| Cluster Contrast | Clustering | Supports cluster-level representation learning |
| SSR-C | Tracklet refinement | Closest recent reference for quality-aware tracklets |

## Limitations of the reviewed literature

The reviewed methods have several limitations relevant to this project:

- Some methods require supervised identity labels.
- Some methods require extensive training resources.
- Many advanced methods are difficult to reproduce exactly.
- Tracking errors may be treated differently across datasets.
- Performance can vary across cameras and domains.
- Strong benchmark performance may not transfer directly to low-quality CCTV footage.
- Clustering results may depend strongly on hyperparameters.
- Automated identity grouping requires human review and should not be treated as verified identity recognition.

## Conclusion

The literature supports the design of a reproducible detection-to-retrieval pipeline based on person detection, ByteTrack tracking, OSNet embeddings, tracklet aggregation, and clustering.

The most important experimental comparison is between frame-level embeddings, mean-pooled tracklet embeddings, and quality-aware tracklet embeddings. The reviewed research indicates that temporal information can improve representation quality, but noisy, blurred, occluded, or incorrectly associated frames must be handled carefully.

The project will therefore report both recognition performance and clustering quality, while documenting technical limitations, privacy considerations, false matches, false splits, and camera-domain bias.