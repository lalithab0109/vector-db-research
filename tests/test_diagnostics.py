"""Final-leaf geometry/ownership diagnostics do not alter the index."""

import numpy as np
import pytest

from vecstudy.core.config import IndexConfig
from vecstudy.core.metrics import Counters
from vecstudy.core.types import HierarchyLevel, Posting
from vecstudy.harness.diagnostics import (
    geometry_diagnostics, reconstruct_owners, segment_geometry_diagnostics,
)
from vecstudy.index.segment import Segment


def _segment(X, centers, owned, members=None):
    X = np.asarray(X, dtype=np.float32).reshape(-1, 2)
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    if members is None:
        members = owned
    postings = [Posting(i, c.copy(), np.array(members[i], dtype=np.int64),
                        np.array(owned[i], dtype=np.int64))
                for i, c in enumerate(centers)]
    return Segment([HierarchyLevel(centers, [])], postings, X)


def test_geometry_agreement():
    seg = _segment([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[0], [1]])
    stats = segment_geometry_diagnostics(seg, batch_size=1)
    assert stats["evaluated_vectors"] == 2
    assert stats["leaf_centroid_count"] == 2
    assert stats["geometry_disagreement_count"] == 0
    assert stats["geometry_disagreement_rate"] == 0
    assert stats["owner_vs_ip_mismatch_count"] == 0


def test_geometry_disagreement_from_centroid_norms():
    # IP chooses c0 (0.8 > 0.5), L2 chooses c1 (0.25 < 0.4).
    seg = _segment([[1, 0]], [[0.8, 0.6], [0.5, 0]], [[0], []])
    stats = segment_geometry_diagnostics(seg)
    assert stats["geometry_disagreement_count"] == 1
    assert stats["geometry_disagreement_rate"] == 1
    assert stats["owner_vs_ip_mismatch_rate"] == 0


def test_single_centroid():
    seg = _segment([[1, 0], [0, 1]], [[0.2, 0.3]], [[0, 1]])
    stats = segment_geometry_diagnostics(seg)
    assert stats["geometry_disagreement_rate"] == 0
    assert stats["owner_vs_ip_mismatch_rate"] == 0


@pytest.mark.parametrize("centers,owned", [([], []), ([[0, 0]], [[]])])
def test_empty_input(centers, owned):
    stats = segment_geometry_diagnostics(_segment([], centers, owned))
    assert stats["evaluated_vectors"] == 0
    assert stats["geometry_disagreement_count"] == 0
    assert stats["geometry_disagreement_rate"] == 0
    assert stats["owner_vs_ip_mismatch_count"] == 0
    assert stats["owner_vs_ip_mismatch_rate"] == 0


def test_owner_reconstruction_and_measured_mismatch():
    seg = _segment([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1], [0]])
    np.testing.assert_array_equal(reconstruct_owners(seg, np.array([0, 1])), [1, 0])
    stats = segment_geometry_diagnostics(seg)
    assert stats["owner_vs_ip_mismatch_count"] == 2
    assert stats["owner_vs_ip_mismatch_rate"] == 1


@pytest.mark.parametrize("owned", [[[0], [0]], [[0, 0], []]])
def test_duplicate_ownership_rejected(owned):
    seg = _segment([[1, 0]], [[1, 0], [0, 1]], owned)
    with pytest.raises(ValueError, match="multiple primary owners"):
        segment_geometry_diagnostics(seg)


@pytest.mark.parametrize("explicit", [False, True])
def test_missing_ownership_rejected(explicit):
    seg = _segment([[1, 0]], [[1, 0]], [[]], members=[[] if explicit else [0]])
    with pytest.raises(ValueError, match="zero primary owners"):
        segment_geometry_diagnostics(seg, np.array([0]) if explicit else None)


def test_exact_ties_choose_lowest_leaf_id():
    seg = _segment([[1, 0]], [[0.5, 0], [0.5, 0]], [[], [0]])
    stats = segment_geometry_diagnostics(seg)
    assert stats["geometry_disagreement_rate"] == 0
    assert stats["owner_vs_ip_mismatch_count"] == 1


def test_tombstones_excluded_and_replicas_counted_once():
    seg = _segment([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[0], [1]],
                   members=[[0, 1], [0, 1]])
    seg.mark_deleted(1)
    assert segment_geometry_diagnostics(seg)["evaluated_vectors"] == 1
    with pytest.raises(ValueError, match="tombstoned"):
        reconstruct_owners(seg, np.array([1]))


def test_batches_match_reference_and_do_not_mutate(synthetic_200):
    X = synthetic_200
    seg = Segment.build(np.arange(len(X)), X,
                        IndexConfig(dim=X.shape[1], leaf_size=20), Counters())
    centers = seg.levels[-1].centroids.copy()
    arrays = [(p.owned.copy(), p.members.copy()) for p in seg.postings]
    ip = np.argmax(X @ centers.T, axis=1)
    l2 = np.argmin(np.sum((X.astype(np.float64)[:, None, :] - centers.astype(np.float64)) ** 2,
                         axis=2), axis=1)
    expected = int(np.count_nonzero(ip != l2))
    for batch in (1, 7, 31):
        stats = segment_geometry_diagnostics(seg, batch_size=batch)
        assert stats["geometry_disagreement_count"] == expected
        assert stats["owner_vs_ip_mismatch_count"] == 0
    np.testing.assert_array_equal(seg.levels[-1].centroids, centers)
    for p, (owned, members) in zip(seg.postings, arrays):
        np.testing.assert_array_equal(p.owned, owned)
        np.testing.assert_array_equal(p.members, members)


def test_segment_aggregation_is_vector_weighted():
    a = _segment([[1, 0]], [[0.8, 0.6], [0.5, 0]], [[0], []])
    b = _segment([[1, 0], [0, 1]], [[1, 0]], [[0, 1]])
    stats = geometry_diagnostics([a, b], batch_size=1)
    assert stats["evaluated_vectors"] == 3
    assert stats["leaf_centroid_count"] == 3
    assert stats["geometry_disagreement_rate"] == pytest.approx(1 / 3)
    assert geometry_diagnostics([])["geometry_disagreement_rate"] == 0


def test_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        segment_geometry_diagnostics(_segment([], [], []), batch_size=0)
