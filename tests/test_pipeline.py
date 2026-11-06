import numpy as np
import pytest

from cctv_reid import clustering, degradations, embeddings, evaluation, io_utils, synthetic
from cctv_reid.detection import Detection, from_array
from cctv_reid.tracking import ByteTrackLite, iou_matrix


# ---------- io_utils ----------
def test_parse_mars_name():
    assert io_utils.parse_mars_name("0001C1T0002F015.jpg") == (1, 1, 2, 15)
    assert io_utils.parse_mars_name("-001C3T0010F100.jpg")[0] == -1
    assert io_utils.parse_mars_name("00-1C1T0001F001.jpg") == (-1, 1, 1, 1)  # junk folder in bbox_test
    assert io_utils.parse_mars_name("0000C2T0003F010.jpg")[0] == 0  # distractor
    with pytest.raises(ValueError):
        io_utils.parse_mars_name("not_a_mars_name.jpg")


def test_set_seed_is_reproducible():
    a = io_utils.set_seed(255).normal(size=3)
    b = io_utils.set_seed(255).normal(size=3)
    assert np.allclose(a, b)


# ---------- degradations ----------
def test_degradations_shape_and_effect():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(64, 32, 3), dtype=np.uint8)
    assert degradations.blur(img, 2.0).shape == img.shape
    assert degradations.blur(img, 0).tolist() == img.tolist()
    assert degradations.darken(img, 0.5).mean() < img.mean()
    assert degradations.low_resolution(img, 0.25).shape == img.shape
    occ = degradations.occlude(img, 0.25, np.random.default_rng(1))
    assert (occ == 0).all(axis=(1, 2)).sum() == 16
    assert degradations.subsample_frames(list(range(10)), 3) == [0, 3, 6, 9]


def test_blur_reduces_sharpness():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(64, 32, 3), dtype=np.uint8)
    s0 = embeddings.frame_sharpness(img.mean(axis=2))
    s1 = embeddings.frame_sharpness(degradations.blur(img, 2.0).mean(axis=2))
    assert s1 < s0


# ---------- embeddings ----------
def test_pooling_outputs_unit_vectors():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(10, 16))
    q = rng.random(10)
    for pooled in (embeddings.mean_pool(f), embeddings.quality_filtered_pool(f, q),
                   embeddings.quality_weighted_pool(f, q), embeddings.attention_pool(f)):
        assert pooled.shape == (16,)
        assert np.isclose(np.linalg.norm(pooled), 1.0)


def test_quality_filter_ignores_bad_frames():
    good = np.tile(np.array([1.0, 0.0]), (4, 1))
    bad = np.tile(np.array([0.0, 1.0]), (4, 1))
    frames = np.vstack([good, bad])
    quality = np.array([1.0] * 4 + [0.0] * 4)
    pooled = embeddings.quality_filtered_pool(frames, quality, keep=0.5)
    assert np.allclose(pooled, [1.0, 0.0])
    assert embeddings.mean_pool(frames)[1] > 0.5  # plain mean is pulled toward the bad frames


def test_attention_downweights_outliers():
    rng = np.random.default_rng(0)
    base = np.array([1.0, 0, 0, 0])
    good = base + rng.normal(scale=0.05, size=(9, 4))
    outlier = np.array([[0, 0, 1.0, 0]])
    frames = np.vstack([good, outlier])
    assert embeddings.attention_pool(frames) @ base > embeddings.mean_pool(frames) @ base


def test_quality_pool_requires_scores():
    f = np.random.default_rng(0).normal(size=(6, 4))
    with pytest.raises(ValueError):
        embeddings.pool_tracklets(f, {"a": [0, 1, 2], "b": [3, 4, 5]}, "quality_weighted")


# ---------- evaluation ----------
def test_pairwise_prf_perfect_and_split():
    ids = np.array([0, 0, 1, 1])
    assert evaluation.pairwise_prf(ids, np.array([5, 5, 7, 7]))["f1"] == 1.0
    split = evaluation.pairwise_prf(ids, np.array([0, 1, 2, 3]))
    assert split["recall"] == 0.0
    merged = evaluation.pairwise_prf(ids, np.array([0, 0, 0, 0]))
    assert merged["recall"] == 1.0 and merged["precision"] < 1.0


def test_cluster_metrics_known_values():
    ids = np.array([0, 0, 1, 1])
    m = evaluation.cluster_metrics(ids, np.array([0, 0, 1, 1]))
    assert m["purity"] == 1.0 and m["ari"] == 1.0 and m["clusters_per_identity"] == 1.0
    m = evaluation.cluster_metrics(ids, np.array([0, 0, 0, 0]))
    assert m["merged_cluster_fraction"] == 1.0


def test_retrieval_camera_exclusion_and_ranks():
    # Gallery has same-id same-camera item (must be ignored) and correct cross-camera item.
    q = np.array([[1.0, 0.0]])
    g = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    r = evaluation.retrieval_metrics(q, [7], [0], g, [7, 7, 8], [0, 1, 1])
    assert r["rank1"] == 1.0 and r["mAP"] == 1.0
    # If the nearest cross-camera item is wrong, rank-1 fails but rank-5 passes.
    g2 = np.array([[0.9, 0.1], [0.5, 0.5]])
    r2 = evaluation.retrieval_metrics(q, [7], [0], g2, [8, 7], [1, 1])
    assert r2["rank1"] == 0.0 and r2["rank5"] == 1.0 and r2["mAP"] == 0.5


# ---------- clustering ----------
def _blobs(rng, n_ids=5, per=8, noise=0.05):
    centers = embeddings.l2_normalize(rng.normal(size=(n_ids, 16)))
    x = np.vstack([c + rng.normal(scale=noise, size=(per, 16)) for c in centers])
    return x, np.repeat(np.arange(n_ids), per)


@pytest.mark.parametrize("method", ["dbscan", "agglomerative"])
def test_clustering_recovers_clean_blobs(method):
    x, y = _blobs(np.random.default_rng(0))
    thr, f1 = clustering.tune_threshold(x, y, method, grid=np.linspace(0.05, 0.9, 12))
    labels = clustering.cluster(x, method, thr)
    assert evaluation.pairwise_prf(y, labels)["f1"] > 0.95
    assert f1 > 0.95


def test_dbscan_noise_becomes_singletons():
    x = np.array([[1.0, 0], [0.99, 0.01], [0, 1.0]])
    labels = clustering.cluster(x, "dbscan", 0.05, min_samples=2)
    assert len(set(labels)) == 2 and labels[0] == labels[1] != labels[2]


# ---------- tracking ----------
def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    iou = iou_matrix(a, b)[0]
    assert np.isclose(iou[0], 1.0) and np.isclose(iou[1], 1 / 3) and iou[2] == 0


def _walk(start_x, frame, speed=3.0, y=0.0):
    x = start_x + speed * frame
    return [x, y, x + 20, y + 50]


def test_tracker_keeps_identity_through_low_score_frames():
    frames = []
    for f in range(20):
        dets = from_array(f, [_walk(0, f), _walk(200, f, y=100)], [0.9, 0.9])
        if 8 <= f <= 11:  # person A is "occluded": only a low-score detection remains
            dets[0] = Detection(f, *_walk(0, f), 0.3)
        frames.append(dets)
    tracks = ByteTrackLite(min_hits=3).run(frames)
    assert len(tracks) == 2  # low-score recovery avoids fragmenting A into two tracks
    assert sorted(len(v) for v in tracks.values()) == [20, 20]


def test_tracker_drops_short_tracks():
    frames = [from_array(0, [[0, 0, 10, 10]], [0.9]), from_array(1, [], [])]
    assert ByteTrackLite(min_hits=3).run(frames) == {}


# ---------- synthetic benchmark (pipeline sanity only) ----------
def test_synthetic_dataset_shapes_and_determinism():
    cfg = synthetic.SynthConfig(n_ids=5)
    a = synthetic.make_dataset(cfg, np.random.default_rng(1))
    b = synthetic.make_dataset(cfg, np.random.default_rng(1))
    assert np.allclose(a["frames"], b["frames"])
    assert a["frames"].shape[1] == cfg.dim and len(a["index"]) == 5 * 3 * 2


def test_synthetic_run_end_to_end_small():
    cfg = synthetic.SynthConfig(n_ids=8, min_frames=4, max_frames=6)
    rows = synthetic.run_all([0], base=cfg, conditions={"clean": {}})
    assert {r["method"] for r in rows} == set(synthetic.METHODS)
    for r in rows:
        assert 0 <= r["rank1"] <= 1 and 0 <= r["agglomerative_f1"] <= 1


# ---------- MARS-style evaluation on fake MARS-named features ----------
def test_mars_eval_runs_on_mars_named_features():
    from cctv_reid import mars_eval

    cfg = synthetic.SynthConfig(n_ids=12, n_cams=3, tracklets_per_cam=1, min_frames=5, max_frames=6)
    ds = synthetic.make_dataset(cfg, np.random.default_rng(3))
    names, counter = [], {}
    for i, (pid, cam) in enumerate(zip(ds["ids"], ds["cams"])):
        key = (pid, cam)
        counter[key] = counter.get(key, 0) + 1
        names.append(f"{pid + 1:04d}C{cam + 1}T0001F{counter[key]:03d}.jpg")
    rows = mars_eval.evaluate(ds["frames"], names, seed=255, max_ids=12, max_frames=16, quality=ds["quality"])
    by = {r["method"]: r for r in rows}
    assert set(by) == {"frame", "mean", "attention", "quality_filtered", "quality_weighted"}
    assert by["mean"]["rank1"] >= by["frame"]["rank1"] - 1e-9
    assert all(0 <= r["agglomerative_f1"] <= 1 for r in rows)


# ---------- MARS subset selection ----------
def test_select_frames_subset_rules():
    from cctv_reid.mars_names import select_frames

    names = []
    for pid in range(1, 21):  # 20 identities, 2 tracklets of 30 frames each
        for trk in (1, 2):
            names += [f"{pid:04d}C{trk}T{trk:04d}F{f:03d}.jpg" for f in range(1, 31)]
    names += [f"0000C1T{t:04d}F{f:03d}.jpg" for t in range(1, 6) for f in range(1, 31)]  # distractors
    names += ["00-1C1T0001F001.jpg", "00-1C1T0001F002.jpg"]  # junk
    sel = select_frames(names, seed=255, max_ids=5, max_frames=8, max_distractor_tracklets=2)
    parsed = [io_utils.parse_mars_name(n) for n in sel]
    assert sel == select_frames(names, seed=255, max_ids=5, max_frames=8, max_distractor_tracklets=2)
    assert len({p[0] for p in parsed if p[0] > 0}) == 5
    assert all(p[0] != -1 for p in parsed)
    assert len({(p[0], p[1], p[2]) for p in parsed if p[0] == 0}) == 2
    per_track = {}
    for p in parsed:
        per_track.setdefault(p[:3], []).append(p[3])
    assert all(len(v) <= 8 for v in per_track.values())
    assert all(v[0] == 1 and v[-1] == 30 for v in per_track.values() if len(v) == 8)  # even spread keeps ends
    assert select_frames(names, seed=1, max_ids=5) != select_frames(names, seed=2, max_ids=5)


def test_tune_threshold_counts_matches_frame_level_scoring():
    x, y = _blobs(np.random.default_rng(0), n_ids=4, per=3)
    counts = np.tile([1, 5, 9], 4)  # unequal tracklet lengths
    thr, f1 = clustering.tune_threshold(x, y, "agglomerative", grid=np.linspace(0.05, 0.9, 10), counts=counts)
    labels = clustering.cluster(x, "agglomerative", thr)
    expect = evaluation.pairwise_prf(np.repeat(y, counts), np.repeat(labels, counts))["f1"]
    assert np.isclose(f1, expect)


# ---------- tracking metrics / runtime ----------
def test_tracking_metrics_perfect_and_switch():
    box = lambda x: [x, 0, x + 10, 20]
    gt = {f: [(1, box(f))] for f in range(10)}
    perfect = {f: [(7, box(f))] for f in range(10)}
    m = evaluation.tracking_metrics(gt, perfect)
    assert m["idf1"] == 1.0 and m["id_switches"] == 0
    # Same person tracked under id 7 then id 8: one switch, IDF1 drops (best id covers 6/10).
    switched = {f: [(7 if f < 6 else 8, box(f))] for f in range(10)}
    m = evaluation.tracking_metrics(gt, switched)
    assert m["id_switches"] == 1
    assert np.isclose(m["idf1"], 2 * 6 / (2 * 6 + 4 + 4))
    assert evaluation.tracking_metrics(gt, {})["idf1"] == 0.0


def test_tracking_metrics_on_bytetracklite():
    frames, gt = [], {}
    for f in range(20):
        frames.append(from_array(f, [_walk(0, f), _walk(200, f, y=100)], [0.9, 0.9]))
        gt[f] = [(1, _walk(0, f)), (2, _walk(200, f, y=100))]
    tracks = ByteTrackLite(min_hits=3).run(frames)
    pred = {}
    for tid, hits in tracks.items():
        for d in hits:
            pred.setdefault(d.frame, []).append((tid, list(d.box)))
    m = evaluation.tracking_metrics(gt, pred)
    assert m["idf1"] == 1.0 and m["id_switches"] == 0


def test_runtime_helpers():
    assert evaluation.peak_memory_mb() > 0
    assert evaluation.throughput(100, 2.0) == 50.0 and evaluation.throughput(5, 0) == 0.0


def test_error_analysis_taxonomy_on_mars_named_features():
    from cctv_reid import error_analysis

    cfg = synthetic.SynthConfig(n_ids=12, n_cams=3, tracklets_per_cam=1, min_frames=5, max_frames=6)
    ds = synthetic.make_dataset(cfg, np.random.default_rng(3))
    names, counter = [], {}
    for pid, cam in zip(ds["ids"], ds["cams"]):
        counter[(pid, cam)] = counter.get((pid, cam), 0) + 1
        names.append(f"{pid + 1:04d}C{cam + 1}T0001F{counter[(pid, cam)]:03d}.jpg")
    res = error_analysis.analyze(ds["frames"], names, "mean", max_ids=12)
    assert res["n_queries"] > 0 and set(res["summary"]) <= {
        "identity_merge", "distractor_merge", "cross_camera_split", "mixed_split",
        "wrong_identity_top1", "distractor_top1"}
    assert "Error analysis" in error_analysis.to_markdown(res)


# ---------- fine-tuning helpers (torch-free) ----------
def test_split_identities_disjoint_and_deterministic():
    from cctv_reid.finetune_utils import split_identities

    pids = [p for p in range(1, 41) for _ in range(3)] + [0, 0, -1]
    tr, va = split_identities(pids, 0.1, seed=255)
    assert tr.isdisjoint(va) and len(va) == 4 and len(tr) == 36 and 0 not in tr | va
    assert (tr, va) == split_identities(pids, 0.1, seed=255)


def test_pk_batches_shape_and_small_identity_handling():
    from collections import Counter

    from cctv_reid.finetune_utils import pk_batches

    labels = [0] * 10 + [1] * 2 + [2] * 10 + [3] * 10  # identity 1 has fewer than K images
    batches = list(pk_batches(labels, p=4, k=3, n_batches=5, seed=1))
    assert len(batches) == 5 and all(len(b) == 12 for b in batches)
    for b in batches:
        assert set(Counter(labels[i] for i in b).values()) == {3}
    assert batches == list(pk_batches(labels, p=4, k=3, n_batches=5, seed=1))
    with pytest.raises(ValueError):
        list(pk_batches(labels, p=5, k=2, n_batches=1))


# ---------- robustness helpers (torch-free) ----------
def test_robustness_conditions_and_quality():
    from cctv_reid import robustness

    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 255, size=(6, 64, 32, 3), dtype=np.uint8)
    for name in robustness.CONDITIONS:
        out = robustness.apply_condition(imgs[0], name, np.random.default_rng(1))
        assert out.shape == imgs[0].shape and out.dtype == np.uint8
    assert robustness.apply_condition(imgs[0], "clean", rng).tolist() == imgs[0].tolist()
    blurred = np.stack([robustness.apply_condition(i, "blur_s4", rng) for i in imgs])
    q_clean, q_blur = robustness.sharpness_quality(imgs), robustness.sharpness_quality(blurred)
    assert q_clean.min() == 0 and q_clean.max() == 1 and q_blur.shape == (6,)
    mixed = robustness.sharpness_quality(np.concatenate([imgs[:3], blurred[:3]]))
    assert mixed[:3].mean() > mixed[3:].mean()  # sharper frames rank higher


def test_subsample_indices_per_tracklet():
    from cctv_reid import robustness

    names = [f"0001C1T0001F{f:03d}.jpg" for f in range(1, 9)] + [f"0002C1T0001F{f:03d}.jpg" for f in range(1, 4)]
    assert robustness.subsample_indices(names, 1).tolist() == list(range(11))
    assert robustness.subsample_indices(names, 4).tolist() == [0, 4, 8]  # first frame of each 4-block; tracklet 2 keeps its first
