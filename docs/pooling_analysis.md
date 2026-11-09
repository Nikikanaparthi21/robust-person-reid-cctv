# Tracklet Pooling Analysis: Attention vs Mean and Quality-Based Pooling

Issue #22. All numbers come from the committed result files (`results/*/results.csv`,
`results/mars_osnet_robustness/robustness.csv`); full tables are in [results.md](results.md).

## Question

A tracklet is a short sequence of crops of one person (up to 16 frames here). Each frame gets an embedding
from the encoder. Pooling turns those frame embeddings into one tracklet embedding. Does a smarter pooling
rule beat a plain average?

## Methods compared

All methods live in `src/cctv_reid/embeddings.py` and return an L2-normalised vector.

| Method | Rule |
|---|---|
| `frame` | No pooling: one randomly chosen frame per query tracklet; every gallery frame is a separate item. |
| `mean` | Plain average of the frame embeddings. |
| `quality_filtered` | Average only the sharpest 50% of frames. |
| `quality_weighted` | Softmax-weighted average, sharper frames weigh more (temperature 0.2). |
| `attention` | Softmax-weighted average, frames closer to the tracklet's median embedding weigh more (temperature 0.1). |

Frame quality is a sharpness score (variance of a Laplacian-style filter), converted to a 0-1 percentile rank.

**Note on "learned" attention.** The issue title says "learned temporal aggregation", but `attention_pool` has
no trainable parameters: the weights come from a fixed rule (similarity to the median frame). It is a
parameter-free attention baseline, not a trained attention module. A trained attention layer is listed under
future work.

## Results

### Clean images, three encoders (mAP)

| Encoder | frame | mean | attention |
|---|---:|---:|---:|
| ResNet-50 (ImageNet, frozen) | 0.157 | 0.272 | 0.272 |
| OSNet (ImageNet, frozen) | 0.187 | 0.340 | 0.350 |
| OSNet (fine-tuned on MARS train) | 0.825 | 0.914 | 0.914 |

Quality scores were only computed in the robustness run (fine-tuned OSNet), so the quality-based methods
appear in the next table only.

### Fine-tuned OSNet, all five methods (mAP)

| Condition | frame | mean | attention | quality_filtered | quality_weighted |
|---|---:|---:|---:|---:|---:|
| clean | 0.825 | **0.914** | **0.914** | 0.909 | 0.907 |
| blur sigma 4 | 0.622 | 0.781 | **0.784** | 0.774 | 0.773 |
| darkness x0.25 | 0.574 | 0.725 | **0.734** | 0.725 | 0.724 |
| low-res x0.25 | 0.739 | 0.860 | **0.864** | 0.848 | 0.846 |
| occlusion 50% | 0.331 | **0.789** | 0.698 | 0.713 | 0.747 |
| keep every 8th frame | 0.812 | **0.876** | **0.876** | 0.825 | 0.873 |

## Findings

1. **Pooling beats single frames everywhere.** Across all 12 robustness conditions, mean pooling improves mAP over
   frame-level by +0.06 to +0.46 and never loses on rank-1. On clean data the gain holds for all three encoders.
2. **The pooling rule barely matters.** Attention is within 0.01 mAP of mean in 11 of 12 conditions. The
   quality-based methods are never better than mean: 0.00 to 0.08 mAP below it. Their worst cases are 50%
   occlusion (quality-filtered 0.713 vs 0.789) and keep-every-8th (quality-filtered 0.825 vs 0.876, because it
   keeps only 1 of the 2 remaining frames).
3. **Attention hurts under heavy occlusion.** At 50% occlusion attention loses 0.091 mAP to mean (0.698 vs 0.789)
   and 0.041 rank-1 (0.841 vs 0.883).
4. **Clustering F1 is noisier than retrieval.** Agglomerative F1 differences between methods reach 0.1-0.2 in
   some conditions (e.g. quality-filtered 0.914 vs mean 0.741 at 25% occlusion), but they are not consistent
   in direction. The clustering threshold is chosen from a coarse grid on 50 validation identities, so we treat
   single-condition F1 differences as unreliable and draw conclusions from mAP and rank-1.

## Why attention and quality weighting do not help (likely explanations, not tested)

- **Averaging already removes noise.** When corruption differs from frame to frame, a plain average cancels much
  of it. Re-weighting has little left to fix.
- **Sharp weights shrink the effective number of frames.** With temperature 0.1, attention puts most weight on a
  few frames near the median. Under 50% occlusion every frame is badly damaged (in a different place), so the
  median is a poor reference and averaging all 16 frames is better than trusting a few.
- **Sharpness is a poor quality signal for occlusion.** The occluding bar is a hard black edge, which can
  *raise* the Laplacian sharpness score, so occluded frames may be ranked as high quality.

## Limitations

- Single seed (255), 50 test identities, 145 query tracklets: differences under about 0.01-0.02 mAP should not be
  read as real.
- MARS-style protocol, not the official MARS query/gallery lists; numbers are not comparable to published results.
- Occlusion is placed independently in each frame, which favours pooling compared with real, persistent
  occlusion (the same object blocking the person across frames).
- Attention is parameter-free (see the note above), and only one quality signal (sharpness) was tried.

## Conclusion

Aggregating frames into a tracklet embedding is what improves re-identification. With a ReID-trained encoder,
the simple mean is as good as or better than the attention and quality-based rules we tried, and it is the most
robust under heavy occlusion. We keep mean pooling as the default.

## Future work

- A trained attention module (e.g. a small MLP scoring each frame, trained with the encoder).
- Better per-frame quality signals (detector confidence, occlusion or pose estimates).
- Multiple seeds and the official MARS splits to obtain confidence intervals.
