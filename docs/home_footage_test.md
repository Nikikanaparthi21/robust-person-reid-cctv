# Home-footage test plan (3 cameras)

Goal: run the whole pipeline on real camera video and measure how often a person seen on one camera is found on the others.
Processing happens on your own computer; no footage is uploaded anywhere.

## Consent and privacy
Record only your own home and only people who agree to be in the test (family, friends). Tell them it is a test, keep the
videos and the `demo/data/` outputs on your machine, and do not commit them (`demo/data/` is git-ignored; also keep the
raw videos out of the repository). Delete the footage after the project if participants ask.

## What to record
- **3 cameras, 10-15 minutes each, recorded at the same time** (overlap in time is fine and realistic).
- **3-4 volunteers in clearly different clothes.** Each walks past all three cameras at least 3 times, in different orders,
  sometimes alone and sometimes with another person in view.
- **Make it harder on purpose** in part of the recording: one volunteer in a darker jacket or with a bag/hat, someone partly
  blocked by furniture or a door, and (if the cameras have night/IR mode) one short pass in the dark.
- **Resolution/format:** 720p or 1080p, H.264 `.mp4`. If the files are H.265/HEVC or `.dav`/`.avi`, convert once:
  `ffmpeg -i in.dav -c:v libx264 -crf 23 -an cam1.mp4`. Browsers can only play H.264, which the viewer's Play clip needs.
- **Clock:** write down the real clock time of each file's first frame (cameras are often a few seconds off). To sync them,
  have one person clap or wave at each camera at the start and note the offset.

## Ground-truth log (15 minutes of work, makes the results measurable)
Fill `data/examples/ground_truth_template.csv` (copy it to `data/ground_truth.csv`): one row per time a person is in view of a
camera, using the same camera names and the same time style (clock time if you give `@start` times, otherwise `HH:MM:SS`
from the start of that video). Half-second accuracy is plenty.
When two volunteers are in the same camera at the same moment, time alone cannot say which tracklet is which, so those tracklets
are reported as ambiguous and left out of scoring. Prefer one person at a time in most passes.

## Run
```bash
python scripts/run_video_pipeline.py \
  --video "Entrance=/path/cam1.mp4@2026-11-06T09:00:00" "Hallway=/path/cam2.mp4@2026-11-06T09:00:03" "Garage=/path/cam3.mp4@2026-11-06T09:00:01" \
  --weights models/osnet_ft/best.pth --device mps --out demo/data
python scripts/eval_video.py --run demo/data --truth data/ground_truth.csv
open demo/index.html
```
`eval_video.py` prints each tracklet's label plus rank-1 / rank-5 / mAP (cross-camera retrieval) and clustering precision, recall
and F1. Report the home-footage numbers separately from MARS: they test a different camera setup (domain shift).

## Long recordings (hours): process selected segments
Do not run hours of footage. Pick 3-5 busy segments of about 10 minutes where people cross between cameras, and process only those:
```bash
python scripts/run_video_pipeline.py \
  --video "Entrance=/path/cam1.mp4@2026-10-29T08:00:00" "Hallway=/path/cam2.mp4@2026-10-29T08:00:03" "Garage=/path/cam3.mp4@2026-10-29T08:00:01" \
  --segment 00:12:00 00:22:00 --segment 01:30:00 01:40:00 \
  --weights models/osnet_ft/best.pth --device mps --detector mobilenet --fps 3 --out demo/data
```
`--segment START END` is video time (HH:MM:SS from the start of each file) and applies to every video, so cut the cameras at the
same clock times (if the files started at different times, pass a different pair per run or align the files first). Clock times in
the viewer and in the evaluation stay absolute. `--detector mobilenet --fps 3` is much faster than the default; use the default
resnet50 detector at 5 fps if people are missed. Do a 1-minute segment first to check speed and quality.

**Important for scoring:** inside every processed segment, log *every* appearance of every person (including visitors, labelled
`Visitor1`, ...). A person who is on camera but missing from the log is counted as an unlabeled distractor and lowers the scores.

## Motion-triggered recordings (clips joined together)

If your files only contain motion clips, video time is not wall time (the clock jumps between clips). Use the burned-in clock:

```
python scripts/run_video_pipeline.py --read-clock \
  --video "CAM 1=/path/cam1.asf@2026-10-29" "CAM 2=/path/cam2.asf@2026-10-29" "CAM 3=/path/cam3.asf@2026-10-29" \
  --clock-segment 08:18:00 08:21:00 --detector mobilenet --fps 3 --weights models/osnet_ft/best.pth --device mps
python scripts/eval_video.py --run demo/data --truth data/ground_truth.csv
```
`scripts/check_clock.py` shows how well the clock is read and where each file jumps. Camera names in `--video` must match
the `camera` column of the ground-truth log. Tracking restarts at every clock jump, so a tracklet never spans two clips.
