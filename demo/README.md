# Review viewer (demo)

A static web page for a human reviewer: pick a person, see other appearances across cameras ranked by similarity,
and mark each as **Possible match** or **Not a match**. Decisions are stored in the browser and can be exported as CSV.
It also lists automatically formed identity clusters that span several cameras.

## Open the sample (no data needed)
Open `demo/index.html` in a browser. It falls back to `demo/sample/`, which uses synthetic embeddings and placeholder
silhouettes. It demonstrates the interface only and is not a result.

## Build it on real MARS results (on your machine)
```bash
python scripts/build_demo.py \
  --features data/embeddings/mars_test_osnet_ft.npy \
  --names data/embeddings/mars_test_osnet_ft_names.txt \
  --zip "/path/to/MARS dataset/data/bbox_test.zip" \
  --out demo/data
open demo/index.html        # macOS; or double-click the file
```
`demo/data/` (thumbnails cut from the MARS zip) is git-ignored. Do not commit or publish MARS images.

## What is shown
Camera, tracklet number, number of frames, three frames from the tracklet, similarity (cosine, from the fine-tuned OSNet
embeddings mean-pooled over the tracklet), and whether the tracklet is in the same cluster as the query.
"Reveal benchmark ground truth" shows whether each candidate really is the same MARS identity. This is only possible on a
labelled benchmark and is meant for evaluation, not for the reviewer.

## Limits
- **No wall-clock time in MARS.** The page shows camera and tracklet only. For real video, pass a tracklet-to-seconds map to
  `demo_data.build(..., times=...)`; the manifest in `docs/tracklet_manifest.md` already stores `timestamp_seconds`.
- Candidates are the top-k by similarity, not confirmed matches. A high score can still be a different person.
- Use only public benchmark data or footage you are authorized to review. The tool does not name or identify anyone.
