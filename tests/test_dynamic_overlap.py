import numpy as np
import pytest

from vecstudy.core.boundary import choose_delta, compute_boundary_statistics
from vecstudy.core.config import IndexConfig
from vecstudy.core.metrics import Counters
from vecstudy.core.types import Posting, HierarchyLevel
from vecstudy.index.segment import Segment
from vecstudy.index.overlap import apply_budget_overlap


@pytest.mark.parametrize('budget, expected', [(0.05, 4), (.10, 9), (.20, 19), (.30, 29)])
def test_initial_budgets(budget, expected):
    r = np.arange(100, dtype=float)
    assert choose_delta(r, budget) == expected
    assert np.mean(r <= expected) == budget


def test_order_statistics_ties_and_purity():
    r = np.array([3., 0., 1., 1., 1.])
    original = r.copy()
    assert choose_delta(r, 0) is None
    assert choose_delta(r, 1) == 3
    assert choose_delta(r, .21) == 1  # ceil(1.05)=2, no interpolation
    assert np.mean(r <= choose_delta(r, .21)) == .8
    for _ in range(3):
        assert choose_delta(r, .21) == choose_delta(r[::-1], .21)
    np.testing.assert_array_equal(r, original)
    deltas = [choose_delta(r, b) for b in [.01, .2, .4, .8, 1]]
    assert deltas == sorted(deltas)


@pytest.mark.parametrize('budget', [-.1, 1.1, np.nan, np.inf, -np.inf])
def test_invalid_budget(budget):
    with pytest.raises(ValueError):
        choose_delta([0, 1], budget)


@pytest.mark.parametrize('r', [[], [[0, 1]], [-1, 0], [np.nan], [-np.inf]])
def test_invalid_r(r):
    with pytest.raises(ValueError):
        choose_delta(r, .1)


def test_infinity_and_zero_distance():
    _, _, r, _, _ = compute_boundary_statistics(
        np.array([[0., 0.]]), np.array([[0., 0.], [1., 0.]]))
    assert np.isposinf(r[0])
    assert choose_delta([0., 1., np.inf], 0) is None
    assert choose_delta([0., 1., np.inf], .5) == 1  # N=3, not finite-only N=2
    for b in [.8, 1]:
        with pytest.raises(ValueError, match='infinite threshold'):
            choose_delta([0., 1., np.inf], b)
    _, _, r, _, _ = compute_boundary_statistics(
        np.array([[0., 0.]]), np.zeros((2, 2)))
    assert choose_delta(r, 1) == 0


def snapshot():
    centers = np.array([[0., 0.], [1., 0.], [0., 1.]], dtype=np.float32)
    # Global IDs are noncontiguous; aligned inputs deliberately permuted.
    postings = [Posting(i, centers[i].copy(), np.array(ids), np.array(ids))
                for i, ids in enumerate([[4], [1], [7, 9]])]
    levels = [HierarchyLevel(np.array([[.5, .5]], dtype=np.float32),
                             [np.array([0, 1, 2])]), HierarchyLevel(centers, [])]
    return Segment(levels, postings, np.zeros((10, 2), dtype=np.float32))


def test_owner_cases_and_invariants():
    seg = snapshot()
    vids = np.array([7, 4, 9, 1])
    r = np.array([0., 0., 1., 0.])
    c1, c2 = np.zeros(4, dtype=int), np.ones(4, dtype=int)
    out, metrics = apply_budget_overlap(seg, vids, r, c1, c2, .25)
    # owner=c1 (4) -> c2; owner=c2 (1) -> c1; outside top2 (7) -> c1.
    assert set(out.postings[0].members) == {4, 1, 7}
    assert set(out.postings[1].members) == {1, 4}
    assert set(out.postings[2].members) == {7, 9}
    for old, new in zip(seg.postings, out.postings):
        np.testing.assert_array_equal(old.owned, new.owned)
        np.testing.assert_array_equal(old.centroid, new.centroid)
        np.testing.assert_array_equal(old.members, old.owned)
        assert len(new.members) == len(np.unique(new.members))
    for old, new in zip(seg.levels, out.levels):
        np.testing.assert_array_equal(old.centroids, new.centroids)
        for a, b in zip(old.children, new.children):
            np.testing.assert_array_equal(a, b)
    assert out.vectors is seg.vectors
    assert metrics['selected_count'] == metrics['replicated_count'] == 3
    assert metrics['selected_fraction'] == metrics['replication_fraction'] == .75
    assert metrics['realized_bloat'] == 3 / 4
    assert metrics['overshoot'] == .5
    counts = {v: sum(v in p.members for p in out.postings) for v in vids}
    assert counts == {7: 2, 4: 2, 9: 1, 1: 2}
    repeat, again = apply_budget_overlap(out, vids, r, c1, c2, .25)
    assert again == metrics
    for a, b in zip(out.postings, repeat.postings):
        np.testing.assert_array_equal(a.members, b.members)
    no, zero = apply_budget_overlap(seg, vids, r, c1, c2, 0)
    assert zero['derived_delta'] is None and zero['realized_bloat'] == 0
    for a, b in zip(seg.postings, no.postings):
        np.testing.assert_array_equal(a.members, b.members)
    # Output arrays do not alias the snapshot's mutable ownership/hierarchy arrays.
    out.postings[0].owned[0] = 0
    out.levels[0].children[0][0] = 2
    assert seg.postings[0].owned.tolist() == [4]
    assert seg.levels[0].children[0].tolist() == [0, 1, 2]


@pytest.mark.parametrize('case', ['coverage', 'duplicate', 'centroid', 'destination', 'tombstone'])
def test_invalid_snapshot_or_alignment(case):
    seg = snapshot()
    vids = np.array([4, 1, 7, 9])
    first, second = np.zeros(4, dtype=int), np.ones(4, dtype=int)
    if case == 'coverage':
        vids[0] = 0
    elif case == 'duplicate':
        vids[0] = 1
    elif case == 'centroid':
        seg.postings[0].centroid[0] = 4
    elif case == 'destination':
        second[0] = 0
    else:
        seg.tombstones.add(4)
    with pytest.raises(ValueError):
        apply_budget_overlap(seg, vids, np.zeros(4), first, second, .1)


def test_actual_r5_snapshot(synthetic_200):
    cfg = IndexConfig(dim=32, build_mode='r5', target_capacity=20,
                      depth=2, branching_factor=4)
    vids = np.arange(200)
    seg = Segment.build(vids, synthetic_200, cfg, Counters())
    _, _, r, c1, c2 = compute_boundary_statistics(synthetic_200, seg.levels[-1].centroids)
    out, metrics = apply_budget_overlap(seg, vids, r, c1, c2, .2)
    assert metrics['selected_count'] == metrics['replicated_count']
    assert metrics['realized_bloat'] == pytest.approx(metrics['replication_fraction'])
    for a, b in zip(seg.postings, out.postings):
        np.testing.assert_array_equal(a.owned, b.owned)
        np.testing.assert_array_equal(a.centroid, b.centroid)
