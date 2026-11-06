# Error analysis: attention pooling + agglomerative (threshold 0.30)

Test identities: 50; retrieval queries: 145.

| Error type | Count |
|---|---|
| identity_merge | 0 |
| distractor_merge | 9 |
| cross_camera_split | 2 |
| mixed_split | 18 |
| wrong_identity_top1 | 2 |
| distractor_top1 | 6 |

## False merges (9 random of 9, seed 255)
- cluster 4 (distractor_merge): ids [0, 580]; e.g. ['0_4_2296', '580_5_2', '580_5_3']
- cluster 5 (distractor_merge): ids [0, 1362]; e.g. ['0_1_299', '0_2_1072', '0_2_1718']
- cluster 9 (distractor_merge): ids [0, 4]; e.g. ['0_5_2859', '4_1_1', '4_1_2']
- cluster 11 (distractor_merge): ids [0, 1288]; e.g. ['0_3_2044', '1288_1_1', '1288_1_2']
- cluster 14 (distractor_merge): ids [0, 1362]; e.g. ['0_2_1889', '1362_1_1', '1362_1_2']
- cluster 20 (distractor_merge): ids [0, 1290]; e.g. ['0_1_311', '0_2_1548', '1290_1_1']
- cluster 21 (distractor_merge): ids [0, 582]; e.g. ['0_2_1664', '582_1_1', '582_1_2']
- cluster 27 (distractor_merge): ids [0, 352]; e.g. ['0_1_182', '352_2_3', '352_2_4']
- cluster 65 (distractor_merge): ids [0, 732]; e.g. ['0_5_2758', '732_1_11']

## False splits (10 random of 20, seed 255)
- id 160 (mixed_split): 2 clusters; e.g. ['160_2_1', '160_2_2', '160_3_3']
- id 468 (mixed_split): 3 clusters; e.g. ['468_3_1', '468_3_2', '468_3_3']
- id 580 (cross_camera_split): 2 clusters; e.g. ['580_4_1', '580_5_2', '580_5_3']
- id 582 (mixed_split): 2 clusters; e.g. ['582_1_1', '582_1_2', '582_1_3']
- id 732 (mixed_split): 4 clusters; e.g. ['732_1_1', '732_1_2', '732_1_3']
- id 950 (mixed_split): 2 clusters; e.g. ['950_1_1', '950_1_2', '950_1_3']
- id 970 (mixed_split): 2 clusters; e.g. ['970_1_1', '970_1_2', '970_1_3']
- id 1124 (mixed_split): 2 clusters; e.g. ['1124_1_1', '1124_1_2', '1124_1_3']
- id 1362 (mixed_split): 2 clusters; e.g. ['1362_1_1', '1362_1_2', '1362_2_3']
- id 1490 (mixed_split): 2 clusters; e.g. ['1490_1_1', '1490_1_2', '1490_1_3']

## Retrieval misses (8 random of 8, seed 255)
- 8_1_1 -> 0_5_2358 (distractor_top1, sim 0.61)
- 8_1_2 -> 0_5_2358 (distractor_top1, sim 0.61)
- 8_1_3 -> 0_5_2358 (distractor_top1, sim 0.65)
- 352_1_1 -> 440_2_1 (wrong_identity_top1, sim 0.72)
- 582_1_3 -> 0_2_1664 (distractor_top1, sim 0.75)
- 732_1_11 -> 0_5_2758 (distractor_top1, sim 0.77)
- 970_1_5 -> 4_5_7 (wrong_identity_top1, sim 0.66)
- 1264_1_8 -> 0_2_1056 (distractor_top1, sim 0.72)
