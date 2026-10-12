import json
import csv
import numpy as np
import pytest

from vecstudy.core.config import IndexConfig, QueryConfig
from vecstudy.core.metrics import Counters
from vecstudy.core.boundary import compute_boundary_statistics, choose_delta
from vecstudy.harness.datasets import compute_ground_truth, recall_at_k
from vecstudy.harness.overlap_experiment import (
    run_overlap_comparison, snapshot_digest, evaluate_queries, save_results)
from vecstudy.index.overlap import apply_fixed_overlap
from vecstudy.index.query import search
from vecstudy.index.segment import Segment


def test_same_snapshot_comparison(synthetic_200, tmp_path, monkeypatch):
    import vecstudy.harness.overlap_experiment as harness
    import vecstudy.index.overlap as overlap
    X = synthetic_200
    cfg = IndexConfig(dim=32, build_mode='r5', target_capacity=20, depth=2, branching_factor=4)
    seg = Segment.build(np.arange(200), X, cfg, Counters())
    fingerprint = snapshot_digest(seg)
    queries = X[:3]
    gt = compute_ground_truth(queries, X, k=10, cache_dir=tmp_path)
    query_cfg = QueryConfig(k=10, n_postings=2)
    boundary_calls = []
    boundary = harness.compute_boundary_statistics
    def counted_boundary(*args, **kwargs):
        boundary_calls.append(kwargs['geometry'])
        return boundary(*args, **kwargs)
    monkeypatch.setattr(harness, 'compute_boundary_statistics', counted_boundary)
    budget_calls = []
    selector = overlap.choose_delta
    def counted_selector(r, b):
        if b:
            budget_calls.append(b)
        return selector(r, b)
    monkeypatch.setattr(overlap, 'choose_delta', counted_selector)
    rows = run_overlap_comparison(seg, np.arange(200), queries, gt, query_cfg)
    assert boundary_calls == ['l2']
    assert budget_calls == [.05, .1, .2, .3]
    assert len(rows) == 7 and snapshot_digest(seg) == fingerprint
    assert len({r['snapshot_sha256'] for r in rows}) == 1
    # Baseline runs precisely the unchanged search and recall implementation.
    results = []
    scans = 0
    for q in queries:
        c = Counters()
        results.append([v for v, _ in search(q, [seg], query_cfg, c)])
        scans += c.vectors_scanned
    assert rows[0]['recall_at_10'] == recall_at_k(results, gt, 10)
    assert rows[0]['total_scanned'] == scans
    assert rows[0]['scans_per_query'] == scans / 3
    _, _, r, c1, c2 = boundary(X, seg.levels[-1].centroids, geometry='l2')
    for row in rows:
        assert row['boundary_geometry'] == 'Euclidean L2' and not row['rng']
        if row['threshold_mode'] == 'dynamic':
            assert row['delta'] == choose_delta(r, row['target_bloat'])
        selected = 0 if row['delta'] is None else np.count_nonzero(r <= row['delta'])
        assert row['selected_count'] == row['replicated_count'] == selected
        assert row['actual_bloat'] == pytest.approx(selected / 200)
    rerun = run_overlap_comparison(seg, np.arange(200), queries, gt, query_cfg)
    assert rerun == rows
    save_results(rows, {'same_snapshot': True}, tmp_path)
    assert json.loads((tmp_path / 'results.json').read_text())['results'] == rows
    with (tmp_path / 'results.csv').open() as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 7
    assert float(csv_rows[0]['scans_per_query']) == scans / 3


def test_exact_noninteger_scan_average(monkeypatch):
    import vecstudy.harness.overlap_experiment as harness
    counts = iter([2, 3])
    def fake_search(q, units, cfg, counters):
        counters.vectors_scanned = next(counts)
        return [(0, 1.)]
    monkeypatch.setattr(harness, 'search', fake_search)
    result = evaluate_queries(None, np.zeros((2, 2)), np.tile(np.arange(10), (2, 1)), QueryConfig())
    assert result['total_scanned'] == 5
    assert result['scans_per_query'] == 2.5
    assert result['recall_at_10'] == .1


@pytest.mark.parametrize('delta', [-1, np.nan, np.inf, -np.inf])
def test_invalid_fixed_delta(synthetic_200, delta):
    seg = Segment.build(np.arange(200), synthetic_200,
        IndexConfig(dim=32, build_mode='r5', target_capacity=20), Counters())
    _, _, r, c1, c2 = compute_boundary_statistics(synthetic_200, seg.levels[-1].centroids)
    with pytest.raises(ValueError, match='delta'):
        apply_fixed_overlap(seg, np.arange(200), r, c1, c2, delta)


def test_fixed_uses_l2_not_legacy_ip(monkeypatch):
    from vecstudy.core.types import Posting, HierarchyLevel
    import vecstudy.index.postings as legacy
    # Unnormalized means cause differing IP/L2 ranking and boundary fractions.
    X = np.array([[.8, .6], [.6, .8]], dtype=np.float32)
    centers = np.array([[.1, .1], [1., 0.]], dtype=np.float32)
    seg = Segment([HierarchyLevel(centers.copy(), [])],
        [Posting(0, centers[0].copy(), np.array([0]), np.array([0])),
         Posting(1, centers[1].copy(), np.array([1]), np.array([1]))], X)
    def forbidden(*args, **kwargs):
        raise AssertionError('legacy geometric assignment must not run')
    monkeypatch.setattr(legacy, 'assign', forbidden)
    _, _, r, c1, c2 = compute_boundary_statistics(X, centers, geometry='l2')
    _, _, r_ip, _, _ = compute_boundary_statistics(X, centers, geometry='ip')
    delta = .5
    assert not np.array_equal(r <= delta, r_ip <= delta)
    out, metrics = apply_fixed_overlap(seg, np.arange(2), r, c1, c2, delta)
    assert metrics['selected_count'] == np.count_nonzero(r <= delta)
    for a, b in zip(seg.postings, out.postings):
        np.testing.assert_array_equal(a.owned, b.owned)
