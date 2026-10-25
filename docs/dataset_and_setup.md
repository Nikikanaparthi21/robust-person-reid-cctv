# MARS Dataset and Reproducible Environment

## 1. Project overview

This project focuses on robust person re-identification and automatic identity clustering in CCTV footage.

The planned pipeline includes:

1. Dataset preparation
2. Person detection
3. Multi-object tracking
4. Tracklet generation
5. Person-ReID embedding extraction
6. Tracklet-level feature aggregation
7. Identity clustering and retrieval
8. Evaluation and documentation

This document records the dataset, environment, configuration, storage, privacy, and reproducibility requirements for the project.

## 2. Dataset

The project uses the MARS dataset, also known as the Motion Analysis and Re-identification Set.

Official dataset page:

https://zheng-lab-anu.github.io/Project/project_mars.html

MARS is a video-based person re-identification dataset collected using six near-synchronized cameras.

The official dataset page reports:

- 1,261 identities
- 20,478 tracklets
- Approximately 1,191,003 bounding boxes
- Six cameras
- 3,248 distractors

The dataset package contains:

- `bbox_train`
- `bbox_test`

The official page reports:

- Training set: 625 identities, 8,298 tracklets, and 509,914 bounding boxes
- Testing set: 636 identities, 12,180 tracklets, and approximately 681,089 bounding boxes

## 3. Dataset access

The official MARS page provides download options through Google Drive/SharePoint and Baidu.

The complete package is approximately 6.26 GB.

The raw dataset must be downloaded by authorized project members from the official or approved source. Raw dataset files are not committed to this repository because of their size and usage restrictions.

The repository must not contain:

- Raw MARS images
- Private CCTV footage
- Dataset credentials
- Passwords
- API keys
- Personal access tokens
- Large pretrained model weights
- Generated private imagery

## 4. Filename convention

MARS filenames contain identity, camera, tracklet, and frame information.

Example:

```text
0065C1T0002F0016.jpg