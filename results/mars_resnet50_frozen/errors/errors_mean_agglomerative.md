# Error analysis: mean pooling + agglomerative (threshold 0.20)

Test identities: 50; retrieval queries: 145.

| Error type | Count |
|---|---|
| identity_merge | 26 |
| distractor_merge | 24 |
| cross_camera_split | 18 |
| mixed_split | 29 |
| wrong_identity_top1 | 81 |
| distractor_top1 | 4 |

## False merges (10 random of 50, seed 255)
- cluster 0 (distractor_merge): ids [0, 732, 948]; e.g. ['0_3_2087', '732_1_7', '732_1_9']
- cluster 10 (distractor_merge): ids [0, 182, 208, 220, 284, 394, 468, 570, 582, 970, 1054, 1088, 1340, 1490, 1500]; e.g. ['0_5_2358', '182_2_2', '182_3_3']
- cluster 44 (distractor_merge): ids [0, 220, 284]; e.g. ['0_6_3039', '220_6_18', '220_6_19']
- cluster 56 (identity_merge): ids [684, 1124]; e.g. ['684_2_13', '684_2_15', '1124_2_9']
- cluster 57 (distractor_merge): ids [0, 530]; e.g. ['0_2_1889', '530_2_7']
- cluster 69 (identity_merge): ids [160, 208, 220]; e.g. ['160_6_4', '160_6_7', '160_6_9']
- cluster 79 (distractor_merge): ids [0, 570, 1042]; e.g. ['0_2_941', '0_2_1348', '570_2_7']
- cluster 82 (identity_merge): ids [732, 970]; e.g. ['732_1_4', '970_1_3', '970_1_4']
- cluster 92 (distractor_merge): ids [0, 4, 158, 222]; e.g. ['0_5_2489', '0_5_2603', '0_5_2662']
- cluster 94 (identity_merge): ids [208, 1264]; e.g. ['208_4_31', '1264_1_1', '1264_1_4']

## False splits (10 random of 47, seed 255)
- id 160 (cross_camera_split): 4 clusters; e.g. ['160_2_1', '160_2_2', '160_3_3']
- id 222 (mixed_split): 9 clusters; e.g. ['222_1_1', '222_1_2', '222_1_3']
- id 440 (mixed_split): 4 clusters; e.g. ['440_2_1', '440_3_2', '440_3_3']
- id 468 (mixed_split): 5 clusters; e.g. ['468_3_1', '468_3_2', '468_3_3']
- id 476 (mixed_split): 4 clusters; e.g. ['476_3_1', '476_3_2', '476_3_3']
- id 570 (cross_camera_split): 7 clusters; e.g. ['570_1_1', '570_1_2', '570_2_3']
- id 582 (mixed_split): 5 clusters; e.g. ['582_1_1', '582_1_2', '582_1_3']
- id 732 (mixed_split): 6 clusters; e.g. ['732_1_1', '732_1_2', '732_1_3']
- id 872 (cross_camera_split): 3 clusters; e.g. ['872_1_1', '872_1_2', '872_1_3']
- id 1088 (cross_camera_split): 4 clusters; e.g. ['1088_1_1', '1088_1_2', '1088_1_3']

## Retrieval misses (10 random of 85, seed 255)
- 220_1_4 -> 476_3_2 (wrong_identity_top1, sim 0.83)
- 220_1_8 -> 222_5_25 (wrong_identity_top1, sim 0.78)
- 222_1_4 -> 284_6_10 (wrong_identity_top1, sim 0.67)
- 530_1_1 -> 182_4_4 (wrong_identity_top1, sim 0.81)
- 530_1_2 -> 182_4_4 (wrong_identity_top1, sim 0.84)
- 570_1_1 -> 182_4_4 (wrong_identity_top1, sim 0.82)
- 582_1_3 -> 0_5_2823 (distractor_top1, sim 0.67)
- 950_1_1 -> 160_6_4 (wrong_identity_top1, sim 0.84)
- 1088_1_1 -> 442_3_7 (wrong_identity_top1, sim 0.84)
- 1500_1_2 -> 970_4_9 (wrong_identity_top1, sim 0.91)
