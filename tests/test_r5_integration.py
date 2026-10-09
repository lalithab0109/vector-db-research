"""R5 build invariants without requiring geometric-nearest ownership."""
import math
from dataclasses import replace
import numpy as np
import pytest

from vecstudy.core.config import IndexConfig, QueryConfig, MaintainConfig, ExperimentConfig
from vecstudy.core.metrics import Counters
from vecstudy.core.r5_leaves import build_r5_leaves, finalize_r5_leaves
from vecstudy.harness.diagnostics import segment_geometry_diagnostics
from vecstudy.harness.runner import run_experiment
from vecstudy.index.segment import Segment
from vecstudy.index.query import search
from vecstudy.maintain.inplace import InPlaceMaintainer
from vecstudy.maintain.outofplace import OutOfPlaceMaintainer


def _cfg(**kwargs):
    return IndexConfig(dim=2, build_mode="r5", target_capacity=4, alpha=1.5,
                       **kwargs)


def _check(seg, expected, cap):
    owned = np.concatenate([p.owned for p in seg.postings]) if seg.postings else np.array([])
    assert sorted(owned.tolist()) == sorted(expected)
    assert len(np.unique(owned)) == len(owned)
    for p in seg.postings:
        assert len(p.owned) <= cap
        np.testing.assert_array_equal(p.owned, p.members)
        np.testing.assert_array_equal(p.centroid, seg.levels[-1].centroids[p.id])


def test_baseline_default_equals_explicit(synthetic_200):
    X = synthetic_200
    cfg = IndexConfig(dim=X.shape[1], leaf_size=20, delta=.2, branching_factor=4)
    a = Segment.build(np.arange(len(X)), X, cfg, Counters())
    b = Segment.build(np.arange(len(X)), X, replace(cfg, build_mode="baseline"), Counters())
    for pa, pb in zip(a.postings, b.postings):
        np.testing.assert_array_equal(pa.owned, pb.owned)
        np.testing.assert_array_equal(pa.members, pb.members)
        np.testing.assert_array_equal(pa.centroid, pb.centroid)
    for la, lb in zip(a.levels, b.levels):
        np.testing.assert_array_equal(la.centroids, lb.centroids)


def test_r5_build_labels_ownership_hierarchy_and_search(synthetic_200):
    X = synthetic_200
    cfg = IndexConfig(dim=X.shape[1], build_mode="r5", target_capacity=20,
                      alpha=1.5, tiny_cluster_fraction=0, depth=3, branching_factor=4)
    centers, labels, _ = build_r5_leaves(X, cfg, Counters())
    seg = Segment.build(np.arange(len(X)), X, cfg, Counters())
    _check(seg, range(len(X)), 30)
    np.testing.assert_array_equal(seg.levels[-1].centroids, centers)
    for p in seg.postings:
        np.testing.assert_array_equal(p.owned, np.flatnonzero(labels == p.id))
    assert len(seg.levels) == 3
    for q in X[:5]:
        res = search(q, [seg], QueryConfig(k=1, beam_widths=(100,), n_postings=100), Counters())
        assert res[0][0] == int(np.argmax(X @ q))


def test_authoritative_non_nearest_labels_preserved(monkeypatch):
    from vecstudy.core import r5_leaves
    X = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    def cluster(*args, **kwargs):
        assert kwargs["geometry"] == "l2"
        return np.zeros((2, 2)), np.array([0, 1, 1]), {}
    monkeypatch.setattr(r5_leaves, "capacity_aware_kmeans", cluster)
    cfg = IndexConfig(dim=2, build_mode="r5", target_capacity=2, tiny_cluster_fraction=0)
    seg = Segment.build(np.arange(3), X, cfg, Counters())
    assert seg.postings[1].owned.tolist() == [1, 2]
    assert segment_geometry_diagnostics(seg)["owner_vs_ip_mismatch_count"] == 1
    _check(seg, range(3), 3)


def test_recursive_oversized_splitting_terminates():
    X = np.arange(64, dtype=np.float32).reshape(32, 2)
    cfg = _cfg(tiny_cluster_fraction=0)
    centers, labels, stats = finalize_r5_leaves(X, np.zeros(32, dtype=np.int64), 1, cfg, Counters())
    assert stats["oversized_splits"] >= 3
    assert np.bincount(labels).max() <= 6
    assert len(labels) == len(X)
    for pid in range(len(centers)):
        np.testing.assert_allclose(centers[pid], X[labels == pid].mean(axis=0))


def test_degenerate_split_falls_back_safely(monkeypatch):
    from vecstudy.core import r5_leaves
    monkeypatch.setattr(r5_leaves, "capacity_aware_kmeans",
                        lambda X, *a, **kw: (None, np.zeros(len(X), dtype=np.int64), {}))
    _, labels, stats = finalize_r5_leaves(np.ones((32, 2)), np.zeros(32, dtype=np.int64),
                                         1, _cfg(tiny_cluster_fraction=0), Counters())
    assert stats["oversized_splits"] > 1
    assert np.bincount(labels).max() <= 6


def test_tiny_merge_uses_nearest_legal_l2_centroid():
    X = np.array([[0, 0]] * 3 + [[10, 0]] * 3 + [[.1, 0]], dtype=np.float32)
    labels = np.array([0] * 3 + [1] * 3 + [2])
    centers, result, stats = finalize_r5_leaves(X, labels, 3,
        _cfg(tiny_cluster_fraction=.5), Counters())
    assert stats["tiny_merges"] == 1
    assert len(centers) == 2
    assert result[-1] == result[0]
    assert np.bincount(result).max() <= 6


def test_tiny_remains_when_no_legal_merge():
    X = np.ones((7, 2), dtype=np.float32)
    _, labels, stats = finalize_r5_leaves(X, np.array([0] * 6 + [1]), 2,
        _cfg(tiny_cluster_fraction=.5), Counters())
    assert stats["tiny_merges"] == 0
    assert np.bincount(labels).tolist() == [6, 1]


def test_tiny_merge_skips_full_nearest_destination():
    X = np.array([[0, 0]] * 6 + [[10, 0]] * 2 + [[.1, 0]], dtype=np.float32)
    _, labels, stats = finalize_r5_leaves(X, np.array([0] * 6 + [1] * 2 + [2]), 3,
        _cfg(tiny_cluster_fraction=.5), Counters())
    assert labels[-1] == labels[6]
    assert np.bincount(labels).tolist() == [6, 3]
    assert stats["tiny_merges"] == 1


def test_empty_groups_removed_and_empty_segment_searchable():
    X = np.array([[1, 0]], dtype=np.float32)
    centers, labels, _ = finalize_r5_leaves(X, np.array([2]), 4, _cfg(), Counters())
    assert centers.shape == (1, 2)
    assert labels.tolist() == [0]
    seg = Segment.build(np.array([], dtype=np.int64), X, _cfg(), Counters())
    _check(seg, [], 6)
    assert search(X[0], [seg], QueryConfig(), Counters()) == []


def test_global_ids_and_reproducible_build(synthetic_200):
    X = synthetic_200
    vids = np.arange(0, len(X), 2)
    cfg = replace(_cfg(), dim=X.shape[1])
    a = Segment.build(vids, X, cfg, Counters())
    b = Segment.build(vids, X, cfg, Counters())
    _check(a, vids, 6)
    for pa, pb in zip(a.postings, b.postings):
        np.testing.assert_array_equal(pa.owned, pb.owned)
        np.testing.assert_array_equal(pa.centroid, pb.centroid)
    assert a.build_stats == b.build_stats


@pytest.mark.parametrize("kwargs", [{"delta": 0}, {"rng_rule": True},
    {"target_capacity": 0}, {"alpha": 0}, {"lambda_balance": -1},
    {"tiny_cluster_fraction": -1}, {"tiny_cluster_fraction": np.nan},
    {"max_iter": 0}, {"build_mode": "unknown"}, {"branching_factor": 1},
    {"depth": 0}, {"target_capacity": 800, "alpha": 1e308}])
def test_invalid_r5_configuration(kwargs):
    with pytest.raises(ValueError):
        IndexConfig(dim=2, **{"build_mode": "r5", **kwargs})


def test_duplicate_segment_ids_rejected():
    with pytest.raises(ValueError, match="unique"):
        Segment.build(np.array([0, 0]), np.ones((1, 2)), _cfg(), Counters())


def test_small_capacity_automatically_increases_leaf_count():
    cfg = IndexConfig(dim=2, build_mode="r5", target_capacity=10, alpha=.2)
    seg = Segment.build(np.arange(11), np.ones((11, 2)), cfg, Counters())
    _check(seg, range(11), 2)
    assert len(seg.postings) >= math.ceil(11 / 2)


def test_inplace_r5_updates_preserve_capacity():
    X = np.array([[1, 0]] * 10, dtype=np.float32)
    maint = InPlaceMaintainer(X, _cfg(), MaintainConfig(strategy="inplace"), Counters())
    for vid in range(10):
        maint.insert(vid, X[vid])
    maint.delete(0)
    _check(maint.searchable_units()[0], range(1, 10), 6)
    with pytest.raises(ValueError, match="duplicate"):
        maint.insert(1, X[1])


def test_outofplace_r5_flush_and_compaction():
    X = np.ones((20, 2), dtype=np.float32)
    maint = OutOfPlaceMaintainer(X, _cfg(), MaintainConfig(strategy="outofplace",
        memtable_limit=5, compaction_threshold=2), Counters())
    for vid in range(20):
        maint.insert(vid, X[vid])
        maint.maintain()
    units = maint.searchable_units()
    assert len(units) == 1
    _check(units[0], range(20), 6)


def test_outofplace_memtable_is_capacity_bounded_and_ids_unique():
    X = np.ones((15, 2), dtype=np.float32)
    maint = OutOfPlaceMaintainer(X, _cfg(), MaintainConfig(strategy="outofplace",
        memtable_limit=100, compaction_threshold=100), Counters())
    for vid in range(15):
        maint.insert(vid, X[vid])
        for unit in maint.searchable_units():
            assert all(len(p.owned) <= 6 for p in unit.postings)
    assert sum(u.n_vectors for u in maint.searchable_units()) == 15
    with pytest.raises(ValueError, match="unique"):
        maint.insert(0, X[0])


def test_r5_harness_metrics(monkeypatch, tmp_path):
    from vecstudy.harness import runner
    X = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
    monkeypatch.setattr(runner, "_load_dataset", lambda *a: (X, X[:1], np.array([[0]])))
    exp = ExperimentConfig(index=_cfg(), query=QueryConfig(k=1, n_postings=10),
                           maintain=MaintainConfig(strategy="inplace"))
    row = run_experiment(exp, tmp_path)
    assert row["recall"] == 1
    assert row["bloat"] == 0
    assert row["final_leaf_postings"] == 1
    assert row["build_time_seconds"] >= 0
    assert "owner_vs_ip_mismatch_rate" in row
    assert row["oversized_splits"] == 0
    assert row["tiny_merges"] == 0
