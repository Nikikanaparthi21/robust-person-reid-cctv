# Team Start Guide

## First day checklist

1. Accept the GitHub invitation and clone the private repository.
2. Read README.md, CONTRIBUTING.md, docs/proposal.md, docs/weekly-plan.md, and docs/experiment-protocol.md.
3. Choose one Week 1 task and assign one reviewer in the project board.
4. Install the documented environment from requirements.txt and run the configuration smoke test.
5. Decide the public benchmark and record its license and download instructions. Do not upload the dataset.
6. Record the authorized CCTV demonstration data locally with a source hash and keep it outside GitHub.

## Existing CCTV workflow to integrate

The current prototype supports person discovery, gallery generation, target selection, and footage extraction. The planned interface is:

- discovery creates a person gallery and person index;
- a reviewer selects a person or cluster;
- export produces matching clips, segments.csv, manifest.json, timestamps, and source hashes.

Track-only or face-hidden people must remain visible as reviewable entries; the system must not assume that missing face evidence means the person was not detected.

## Definition of Done

A task is complete only when the code or document is committed through a pull request, a smoke test or experiment command has been run, results are recorded, limitations are documented, and no private media or credentials are included.

## Scope guardrails

The required deliverable is the research comparison, not a live deployment, full-stack application, crime prediction system, or collection of multiple advanced models. Optional work must not delay the frame-versus-tracklet baseline and four controlled comparisons.
