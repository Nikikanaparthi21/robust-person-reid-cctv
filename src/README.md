# Source layout

Planned modules:

- detection.py: detector wrapper and normalized person detections.
- tracking.py: tracker integration and tracklet manifest creation.
- embeddings.py: frame and tracklet feature extraction.
- clustering.py: similarity, DBSCAN, and hierarchical baselines.
- evaluation.py: retrieval, clustering, tracking, and runtime metrics.
- io_utils.py: manifests, timestamps, hashes, and privacy-safe outputs.

Keep modules deterministic where possible and expose configuration through YAML rather than hard-coded paths.
