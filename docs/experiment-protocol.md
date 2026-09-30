# Experiment Protocol

## Core comparison

Compare independent frame embeddings against aggregated tracklet embeddings using the same detector, tracker, encoder, split, similarity function, and hardware.

## Baseline configuration

- Detector: documented pretrained person detector.
- Tracker: ByteTrack or equivalent.
- Encoder: documented pretrained person-ReID encoder such as OSNet.
- Aggregation: individual frame vector versus mean tracklet vector.
- Similarity: cosine distance.
- Clustering: DBSCAN and agglomerative clustering.
- Evaluation: rank-1/rank-5, mAP where supported, pairwise precision/recall/F1, purity, ARI, IDF1/ID switches where supported, runtime, throughput, and memory.

## Controlled conditions

1. Clean versus synthetic blur.
2. Original versus reduced resolution.
3. Normal versus darkened footage.
4. Unoccluded versus partially occluded samples.
5. Higher versus lower sampling rates with original timestamps preserved.

## Required experiment record

Record dataset/version/license, split, source manifest or hash, models/checkpoints, input size, sampling rate, pooling method, clustering parameters, random seed, hardware/software environment, metrics, runtime, and failure cases.

## Error categories

Missed detection; fragmented tracklet; identity switch; false merge; false split; occlusion or pose failure; camera/lighting domain shift; unknown or low-confidence match.
