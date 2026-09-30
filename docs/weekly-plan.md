# Six Week Execution Plan

This provisional calendar begins September 30, 2026. If the instructor provides a different final due date, shift the dates together while keeping the six-week structure.

## Operating rules

- Every task has one primary owner and one reviewer.
- Every change is merged through a pull request linked to a GitHub task.
- A task is Done only after its acceptance criteria and reproducibility notes are present.
- Keep raw CCTV video, face crops, credentials, and large model weights outside GitHub.
- Record dataset version, split, checkpoint, sampling rate, clustering parameters, seed, metrics, and runtime for every experiment.

## Weekly schedule

| Week and deadline | Primary focus | Required acceptance evidence |
|---|---|---|
| Week 1: Sep 30-Oct 6 | Dataset decision, literature, environment, baseline specification | Dataset/license note, environment setup, smoke test, baseline config, literature notes |
| Week 2: Oct 7-Oct 13 | Detection, tracking, and tracklet manifest | Annotated sample output, tracklet JSON/CSV schema, timestamp validation, runtime log |
| Week 3: Oct 14-Oct 20 | Frame-level and tracklet-level embeddings | Reproducible embedding command, feature schema, retrieval examples, unit tests |
| Week 4: Oct 21-Oct 27 | Clustering and identity retrieval | DBSCAN and hierarchical baseline, cluster gallery, threshold note, retrieval metrics |
| Week 5: Oct 28-Nov 3 | Robustness experiments and error analysis | Blur/low-resolution/dark/occlusion/sampling results, error taxonomy, comparison table |
| Week 6: Nov 4-Nov 10 | Final analysis, report, presentation, packaging | Final figures, limitations, reproducibility checklist, demo, report, slides |

## Team ownership

| Workstream | Primary owner | Reviewer |
|---|---|---|
| Detection, tracking, preprocessing | Nikhil Kanaparthi | Team member 3 |
| Embeddings, tracklet aggregation, clustering | Team member 2 | Nikhil Kanaparthi |
| Metrics, robustness, visualizations, report | Team member 3 | Team member 2 |
| Literature, testing, presentation | All members | All members |

Replace Team member 2 and Team member 3 after the team confirms names and preferences.

## Weekly review checklist

At each deadline, review the board, close completed items, move blocked items to Blocked, attach results to the related pull request, and record one short decision note. If a task slips, reduce optional scope before extending the core pipeline.
