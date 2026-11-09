# Privacy and Ethics

Issue #28. This project re-identifies people across camera views. That is a surveillance capability, so this
document sets the rules for how data is handled, how results may be used, and what the system must not be used
for. It applies to code, data, results, the review viewer (`demo/`), the report and the presentation.

This is a course project policy, not legal advice. If anyone plans to use real CCTV footage beyond a short,
authorized demonstration, check the applicable law and university rules first.

## 1. Intended use

- **In scope:** a research comparison of frame-level and tracklet-level person representations on a public
  benchmark (MARS), under controlled image degradations.
- **In scope:** a short qualitative demonstration on footage the team is explicitly authorized to use.
- **Out of scope:** live deployment, monitoring real people, identifying anyone by name, crime prediction, or
  any automated decision about a person.

## 2. What the system does and does not do

- It compares **whole-body appearance** (clothing, shape, colour) between person crops. It does **not** use face
  recognition, and none should be added.
- Its output is a **similarity score**. A high score means two crops look alike; it does not mean they are the
  same person. Results are **leads for human review**, never confirmed identities.
- It does not know who anyone is. MARS identities are anonymous numbers, and no name or ID database is linked.

## 3. Data rules

### Public benchmark (MARS)

- Download MARS only from the official source and follow the terms stated there. Do not redistribute it.
- Never commit MARS images or the dataset zip files to the repository.
- Figures in the report and slides should prefer aggregate plots. If example crops are shown, use a few,
  small, low-resolution ones, and only where the dataset terms allow it.

### Authorized CCTV footage (demonstration only)

- Use footage only with **written permission** from the owner of the cameras, for this course demonstration.
- Keep it **local** on a team member's machine. Never upload it to GitHub, shared drives, chat tools or online
  AI tools.
- Record the source and a file hash locally (see `docs/team-start.md`), not the footage itself.
- Use the shortest clip that shows the method working. Do not collect extra footage "just in case".
- People who are seen only from behind, or whose face is hidden, must still appear as reviewable entries. The
  system must not treat a missing face as a missing person (`docs/team-start.md`).

### Derived data counts as personal data

Person crops, thumbnails, embeddings (`.npy`/`.npz`), cluster assignments and exported review CSVs are all
derived from images of real people. Treat them exactly like the raw footage: local only, never committed, and
deleted on the retention schedule below. Only aggregate metrics (the CSV and Markdown files in `results/`) are
committed.

### Retention

- Delete local CCTV clips and everything derived from them (crops, thumbnails, embeddings, review CSVs) at the
  end of the course, once grades are final.
- Delete downloaded MARS data and fine-tuned weights at the same time unless a team member has a separate,
  documented reason to keep them for research.

## 4. Human review

- The review viewer (`demo/index.html`) shows ranked candidates and lets a reviewer mark each one
  **Possible match** or **Not a match**. Every result is shown with its similarity score and the disclaimer
  that it is a lead, not an identification.
- No result triggers any action automatically. A person reviewing a candidate decides, and records the
  decision.
- "Reveal benchmark ground truth" exists only to measure accuracy on labelled benchmark data. It does not exist
  for real footage.
- Timestamps: MARS has no wall-clock time, so the viewer shows camera and tracklet instead. For real footage,
  times come from the tracklet manifest (`docs/tracklet_manifest.md`), not from guesses.

## 5. Known failure modes and their harms

| Failure | What it means | Possible harm |
|---|---|---|
| False merge | Two different people are grouped as one | Someone is wrongly linked to another person's movements |
| False split | One person is split into several groups | Their appearances are missed or double-counted |
| Look-alike match | Similar clothing gives a high score | A bystander is flagged because of what they wear |
| Domain shift | Accuracy drops in conditions unlike the training data | Results look reliable but are not |

From our own results (`docs/results.md`, `docs/robustness_analysis.md`):

- Even the best model (fine-tuned OSNet with tracklet pooling) splits each person into about 1.6 groups on
  average, and some top-ranked candidates are different people.
- Accuracy falls sharply in poor conditions: mAP drops from 0.914 on clean images to 0.725 when images are
  darkened to 25% brightness. Real CCTV is often dark, blurry and low-resolution.
- MARS was recorded with six cameras on a single university campus. People, clothing and camera conditions
  elsewhere differ, so these accuracy numbers should not be assumed to transfer.
- The evaluation uses one seed and 50 test identities, and we did not measure whether accuracy differs across
  groups of people (for example by skin tone, age or clothing style). Unequal error rates are possible and
  untested.

## 6. Misuse we prohibit

- Identifying, locating or tracking a specific real person.
- Linking results to names, ID photos, social media or any other identity database.
- Adding face recognition or any biometric beyond whole-body appearance.
- Publishing or sharing matches of real people, including in the report, slides or demo video.
- Using results as evidence, or for any decision about a person (discipline, access, policing, employment).
- Running the system on footage without the owner's written permission.

## 7. Safeguards in this repository

- `.gitignore` blocks video files (any letter case), image files except committed result plots, dataset zips,
  model weights (`.pt`, `.pth`, `.ckpt`, `.safetensors`, `.onnx`), embeddings (`.npy`, `.npz`), `data/raw/`,
  `data/private/`, `data/embeddings/`, `demo/data/` and `.env` files.
- Every pull request must include a privacy check confirming that no raw CCTV, face crops, credentials or
  large weights were added (`CONTRIBUTING.md`).
- The review viewer always shows the "leads for human review" disclaimer.
- `.gitignore` is a safety net, not a guarantee: check `git status` before every commit, and never use
  `git add -f` on data files.

## 8. Checklist before sharing results

- [ ] No real person's image, crop or thumbnail in the report, slides or demo video (MARS examples only where
      the dataset terms allow).
- [ ] Every match shown is described as a candidate for review, with its score.
- [ ] Limitations from section 5 are stated in the report.
- [ ] No raw footage or derived data is in the repository or in any shared folder.
