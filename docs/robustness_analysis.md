# Robustness Analysis: Frame vs Tracklet Embeddings Under Degradation

Issue #24. All numbers come from `results/mars_osnet_robustness/robustness.csv`; full tables are in
[results.md](results.md) and the plot in `results/mars_osnet_robustness/robustness_map.png`.

## Question

CCTV footage is often blurry, dark, low-resolution or partly blocked, and may be recorded at a low frame rate.
Do tracklet-level embeddings stay more reliable than single-frame embeddings under these conditions?

## Setup

- **Model:** OSNet (`osnet_x1_0`) fine-tuned on MARS train (`models/osnet_ft/best.pth`, not in Git).
- **Data:** MARS test, 100 identities (seed 255), up to 16 frames per tracklet, 150 distractor tracklets.
  Half the identities tune clustering thresholds, the other 50 are the test set (145 query tracklets).
- **Degradations** (`src/cctv_reid/degradations.py`), applied to the 128x256 encoder input, then all images are
  re-embedded:

| Family | Mild | Severe |
|---|---|---|
| Blur (Gaussian sigma) | 2 | 4 |
| Darkening (brightness factor) | 0.5 | 0.25 |
| Low resolution (down-sampling factor) | 0.5 | 0.25 |
| Occlusion (black band, fraction of height, random position per frame) | 25% | 50% |
| Lower sampling rate (keep every n-th frame) | 2, 4 | 8 |

- **Command:** `make robustness ZIP=/path/bbox_test.zip DEVICE=mps` (about 70 minutes on an M-series Mac).

## Results (mAP)

| Condition | frame | mean pooling | gain from pooling |
|---|---:|---:|---:|
| clean | 0.825 | 0.914 | +0.089 |
| blur sigma 2 | 0.772 | 0.881 | +0.109 |
| blur sigma 4 | 0.622 | 0.781 | +0.158 |
| darkness x0.5 | 0.759 | 0.878 | +0.120 |
| darkness x0.25 | 0.574 | 0.725 | +0.151 |
| low-res x0.5 | 0.826 | 0.910 | +0.084 |
| low-res x0.25 | 0.739 | 0.860 | +0.121 |
| occlusion 25% | 0.663 | 0.912 | +0.250 |
| occlusion 50% | 0.331 | 0.789 | +0.458 |
| keep every 2nd | 0.823 | 0.911 | +0.088 |
| keep every 4th | 0.814 | 0.900 | +0.086 |
| keep every 8th | 0.812 | 0.876 | +0.063 |

The other pooling methods (attention, quality-based) are compared in [pooling_analysis.md](pooling_analysis.md).

## Findings

1. **Tracklet pooling wins in every condition for retrieval.** Mean pooling has higher mAP than frame-level in
   all 12 conditions and equal or higher rank-1 in all 12.
2. **The worse the images, the bigger the gain.** The gain grows from +0.09 on clean data to +0.15 for severe
   blur and darkness and +0.46 for 50% occlusion.
3. **Darkness is the hardest condition for tracklets.** Mean pooling's lowest mAP is darkness x0.25 (0.725),
   followed by blur sigma 4 (0.781). Darkening hits every frame the same way, so averaging cannot recover the
   lost information.
4. **Occlusion is the hardest for single frames but the easiest for tracklets.** Frame-level mAP collapses to
   0.331 at 50% occlusion. Because the band is in a different place in each frame, different frames show
   different body parts and averaging combines them. Real occlusion is usually persistent, so this result is
   optimistic (see Limitations).
5. **Low resolution matters little at x0.5.** Scores are almost unchanged (0.910 vs 0.914), likely because MARS
   crops are already small (256x128).
6. **Fewer frames cost little.** Keeping every 8th frame (about 2 frames per tracklet) still gives 0.876 mAP;
   even two frames beat one.
7. **Clustering does not always follow retrieval.** Agglomerative F1 is higher for mean pooling in 8 of 12
   conditions but equal or lower than frame-level for blur sigma 4 (0.638 vs 0.696), low-res x0.25, keep every
   4th and keep every 8th (0.718 vs 0.780). Thresholds are tuned on 50 identities from a coarse grid, so
   clustering scores are noisier than retrieval scores.

## Limitations

- **Occlusion is independent per frame.** This favours pooling. A persistent-occlusion test (same band position
  for the whole tracklet) would be more realistic and is not yet run.
- **Training saw similar corruption.** Fine-tuning used random erasing (`T.RandomErasing(p=0.5)` in
  `scripts/finetune_osnet.py`), which looks like the occlusion test, so the model may be unusually robust to it.
- **Synthetic degradations.** Real CCTV mixes several effects (motion blur, noise, compression, uneven lighting);
  each test here applies one clean, uniform effect.
- **Small, single-seed test.** One seed, 50 test identities, 145 queries, MARS-style protocol (not the official
  splits). Small differences should not be over-interpreted.
- **Sampling was simulated on the 16-frame subset**, not by changing the real video frame rate.

## Conclusion

In every degraded condition, tracklet-level embeddings retrieve the right person more reliably than single
frames, and the advantage grows as image quality drops. This supports the project's research question for
retrieval. For clustering the picture is mixed under severe blur and very sparse sampling, which should be
stated in the final report.

## Possible follow-ups

- Persistent occlusion (fixed band per tracklet).
- Combined degradations (e.g. dark + blur) and real low-quality footage.
- Several seeds to report confidence intervals.
