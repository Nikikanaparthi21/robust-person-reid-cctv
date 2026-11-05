# Error analysis: attention pooling + agglomerative (threshold 0.20)

Test identities: 50; retrieval queries: 145.

| Error type | Count |
|---|---|
| identity_merge | 32 |
| distractor_merge | 21 |
| cross_camera_split | 18 |
| mixed_split | 30 |
| wrong_identity_top1 | 80 |
| distractor_top1 | 4 |

## False merges (10 random of 53, seed 255)
- cluster 0 (distractor_merge): ids [0, 732, 948]; e.g. ['0_3_2087', '732_1_7', '732_1_9']
- cluster 9 (distractor_merge): ids [0, 4, 10, 970, 1362]; e.g. ['0_5_2815', '4_1_1', '4_1_2']
- cluster 30 (identity_merge): ids [440, 582]; e.g. ['440_3_3', '440_5_5', '582_3_8']
- cluster 40 (identity_merge): ids [160, 208]; e.g. ['160_6_4', '160_6_7', '160_6_9']
- cluster 41 (identity_merge): ids [222, 530]; e.g. ['222_5_18', '530_5_12']
- cluster 43 (identity_merge): ids [4, 158, 468, 684, 1362]; e.g. ['4_5_5', '158_5_26', '468_5_4']
- cluster 61 (identity_merge): ids [208, 1264]; e.g. ['208_4_31', '1264_1_1', '1264_1_4']
- cluster 65 (identity_merge): ids [208, 222, 284]; e.g. ['208_6_44', '222_6_26', '222_6_27']
- cluster 68 (distractor_merge): ids [0, 220, 284]; e.g. ['0_6_3039', '220_6_18', '220_6_19']
- cluster 77 (distractor_merge): ids [0, 530, 570]; e.g. ['0_2_1318', '530_2_5', '570_2_4']

## False splits (10 random of 48, seed 255)
- id 160 (cross_camera_split): 4 clusters; e.g. ['160_2_1', '160_2_2', '160_3_3']
- id 282 (mixed_split): 3 clusters; e.g. ['282_2_1', '282_2_2', '282_2_3']
- id 440 (mixed_split): 4 clusters; e.g. ['440_2_1', '440_3_2', '440_3_3']
- id 468 (mixed_split): 5 clusters; e.g. ['468_3_1', '468_3_2', '468_3_3']
- id 520 (mixed_split): 4 clusters; e.g. ['520_3_1', '520_3_2', '520_3_3']
- id 580 (cross_camera_split): 3 clusters; e.g. ['580_4_1', '580_5_2', '580_5_3']
- id 594 (mixed_split): 2 clusters; e.g. ['594_3_1', '594_3_2', '594_3_3']
- id 732 (mixed_split): 7 clusters; e.g. ['732_1_1', '732_1_2', '732_1_3']
- id 872 (cross_camera_split): 3 clusters; e.g. ['872_1_1', '872_1_2', '872_1_3']
- id 1088 (cross_camera_split): 4 clusters; e.g. ['1088_1_1', '1088_1_2', '1088_1_3']

## Retrieval misses (10 random of 84, seed 255)
- 220_1_4 -> 476_3_1 (wrong_identity_top1, sim 0.83)
- 220_1_8 -> 222_5_25 (wrong_identity_top1, sim 0.77)
- 222_1_5 -> 158_5_22 (wrong_identity_top1, sim 0.63)
- 352_1_2 -> 476_5_9 (wrong_identity_top1, sim 0.80)
- 530_1_1 -> 182_4_4 (wrong_identity_top1, sim 0.81)
- 530_1_2 -> 182_4_4 (wrong_identity_top1, sim 0.83)
- 530_1_3 -> 476_4_6 (wrong_identity_top1, sim 0.87)
- 582_1_3 -> 0_5_2823 (distractor_top1, sim 0.67)
- 950_1_2 -> 160_6_4 (wrong_identity_top1, sim 0.77)
- 1088_1_3 -> 442_3_7 (wrong_identity_top1, sim 0.80)
