"""Isolated R5 geometry, online balance, hard capacity, and update contracts."""

import numpy as np
import pytest

from vecstudy.core.capacity_clustering import capacity_aware_kmeans
from vecstudy.core.metrics import Counters


@pytest.mark.parametrize("geometry,expected", [("l2", 1), ("ip", 0)])
def test_controlled_geometry_difference(geometry, expected):
    # x dot c: .8 versus .5; squared L2: .4 versus .25.
    centers, labels, info = capacity_aware_kmeans(
        [[1, 0]], 2, geometry=geometry, initial_centroids=[[.8, .6], [.5, 0]],
        lambda_balance=0, max_iter=1,
    )
    assert labels.tolist() == [expected]
    np.testing.assert_array_equal(centers[expected], [1, 0])
    assert info["iterations"] == 1


@pytest.mark.parametrize("geometry", ["l2", "ip"])
def test_expected_assignment_when_geometries_agree(geometry):
    _, labels, _ = capacity_aware_kmeans(
        [[1, 0], [0, 1]], 2, geometry=geometry,
        initial_centroids=[[1, 0], [0, 1]], lambda_balance=0, max_iter=1,
    )
    assert labels.tolist() == [0, 1]


@pytest.mark.parametrize("geometry", ["l2", "ip"])
def test_capacity_forces_non_nearest_assignment_inside_one_batch(geometry):
    _, labels, info = capacity_aware_kmeans(
        [[1, 0], [1, 0]], 2, geometry=geometry,
        initial_centroids=[[1, 0], [-1, 0]], target_capacity=1,
        alpha=1, lambda_balance=0, max_iter=1, batch_size=100,
    )
    assert labels.tolist() == [0, 1]
    assert info["cluster_sizes"].tolist() == [1, 1]
    assert info["capacity_violations"] == 0


@pytest.mark.parametrize("geometry", ["l2", "ip"])
def test_balance_penalty_changes_choice_before_capacity_is_full(geometry):
    kwargs = dict(n_clusters=2, geometry=geometry, initial_centroids=[[1, 0], [.9, 0]],
                  target_capacity=10, alpha=2, max_iter=1)
    _, plain, _ = capacity_aware_kmeans([[1, 0], [1, 0]], lambda_balance=0, **kwargs)
    _, balanced, _ = capacity_aware_kmeans([[1, 0], [1, 0]], lambda_balance=2, **kwargs)
    assert plain.tolist() == [0, 0]
    assert balanced.tolist() == [0, 1]


@pytest.mark.parametrize("geometry", ["l2", "ip"])
def test_feasible_hard_capacity_and_complete_labels(synthetic_200, geometry):
    centers, labels, info = capacity_aware_kmeans(
        synthetic_200, 7, geometry=geometry, target_capacity=24,
        alpha=1.25, seed=7, max_iter=5, batch_size=13,
    )
    assert info["max_capacity"] == 30
    assert labels.shape == (200,)
    assert labels.dtype == np.int64
    assert np.all((labels >= 0) & (labels < 7))
    sizes = np.bincount(labels, minlength=7)
    np.testing.assert_array_equal(sizes, info["cluster_sizes"])
    assert sizes.sum() == 200
    assert sizes.max() <= 30
    assert info["capacity_violations"] == 0
    for pid in np.flatnonzero(sizes):
        np.testing.assert_allclose(centers[pid], synthetic_200[labels == pid].mean(axis=0),
                                   atol=1e-7)


@pytest.mark.parametrize("geometry", ["l2", "ip"])
def test_impossible_capacity_rejected(geometry):
    with pytest.raises(ValueError, match="Impossible capacity"):
        capacity_aware_kmeans(np.ones((5, 2)), 2, geometry=geometry,
                              target_capacity=2, alpha=1)


def test_capacity_uses_ceiling():
    _, _, info = capacity_aware_kmeans(np.ones((3, 2)), 1,
                                      target_capacity=2, alpha=1.1)
    assert info["max_capacity"] == 3
    assert info["cluster_sizes"].tolist() == [3]


@pytest.mark.parametrize("geometry", ["l2", "ip"])
def test_ties_choose_lowest_cluster_id(geometry):
    _, labels, _ = capacity_aware_kmeans(
        [[1, 0], [1, 0]], 3, geometry=geometry,
        initial_centroids=[[1, 0], [1, 0], [1, 0]],
        lambda_balance=0, max_iter=1,
    )
    assert labels.tolist() == [0, 0]


@pytest.mark.parametrize("geometry", ["l2", "ip"])
def test_fixed_seed_and_batch_partition_reproducibility(synthetic_200, geometry):
    kwargs = dict(n_clusters=5, geometry=geometry, seed=42, max_iter=4,
                  target_capacity=40, alpha=1.5)
    reference = capacity_aware_kmeans(synthetic_200, batch_size=7, **kwargs)
    for batch in (7, 19):
        actual = capacity_aware_kmeans(synthetic_200, batch_size=batch, **kwargs)
        np.testing.assert_array_equal(actual[0], reference[0])
        np.testing.assert_array_equal(actual[1], reference[1])
        np.testing.assert_array_equal(actual[2]["cluster_sizes"], reference[2]["cluster_sizes"])
        assert actual[2]["iterations"] == reference[2]["iterations"]


@pytest.mark.parametrize("geometry", ["l2", "ip"])
def test_empty_clusters_retain_previous_centers_without_normalization(geometry):
    init = np.array([[2, 0], [0, 3], [-4, 0]], dtype=np.float32)
    centers, labels, info = capacity_aware_kmeans(
        [[2, 0], [2, 0]], 3, geometry=geometry, initial_centroids=init,
        lambda_balance=0,
    )
    assert labels.tolist() == [0, 0]
    np.testing.assert_array_equal(centers, init)
    assert info["cluster_sizes"].tolist() == [2, 0, 0]
    assert info["converged"]
    np.testing.assert_array_equal(init, [[2, 0], [0, 3], [-4, 0]])


@pytest.mark.parametrize("geometry", ["l2", "ip"])
def test_single_cluster_arithmetic_mean(geometry):
    X = np.array([[2, 0], [0, 4]], dtype=np.float32)
    centers, labels, info = capacity_aware_kmeans(X, 1, geometry=geometry,
                                                 target_capacity=2, alpha=1)
    np.testing.assert_array_equal(centers, [[1, 2]])
    np.testing.assert_array_equal(labels, [0, 0])
    assert info["iterations"] == 2
    assert info["converged"]
    np.testing.assert_array_equal(X, [[2, 0], [0, 4]])


def test_more_clusters_than_vectors():
    centers, labels, info = capacity_aware_kmeans([[1, 0]], 3, seed=0)
    assert centers.shape == (3, 2)
    assert labels.tolist() == [0]
    assert info["cluster_sizes"].tolist() == [1, 0, 0]


def test_empty_dataset():
    centers, labels, info = capacity_aware_kmeans(np.empty((0, 2)), 3)
    np.testing.assert_array_equal(centers, np.zeros((3, 2)))
    assert labels.shape == (0,)
    assert info["cluster_sizes"].tolist() == [0, 0, 0]
    assert info["iterations"] == 0
    assert info["converged"]


@pytest.mark.parametrize("geometry", ["l2", "ip"])
def test_batched_build_count_and_iteration_limit(monkeypatch, synthetic_200, geometry):
    from vecstudy.core import capacity_clustering
    original = capacity_clustering.sim_many_to_many
    batch_lengths = []

    def counted_block(A, B, counters, bucket):
        assert len(A) <= 17
        batch_lengths.append(len(A))
        return original(A, B, counters, bucket)

    monkeypatch.setattr(capacity_clustering, "sim_many_to_many", counted_block)
    counters = Counters()
    _, _, info = capacity_aware_kmeans(synthetic_200, 5, geometry=geometry,
                                      batch_size=17, max_iter=1, counters=counters)
    assert sum(batch_lengths) == 200
    assert counters.distance_computations == {"build": 200 * 5}
    assert info["iterations"] == 1
    assert not info["converged"]


@pytest.mark.parametrize("kwargs", [
    {"n_clusters": 0}, {"target_capacity": 0}, {"target_capacity": 1.5},
    {"alpha": 0}, {"alpha": np.inf}, {"lambda_balance": -1},
    {"lambda_balance": np.nan}, {"max_iter": 0}, {"batch_size": 0},
    {"geometry": "cosine"}, {"initial_centroids": [[1, 0]]},
])
def test_invalid_configuration(kwargs):
    params = {"n_clusters": 2, **kwargs}
    with pytest.raises(ValueError):
        capacity_aware_kmeans([[1, 0], [0, 1]], **params)


def test_nonfinite_vectors_rejected():
    with pytest.raises(ValueError, match="finite"):
        capacity_aware_kmeans([[np.nan, 0]], 1)
