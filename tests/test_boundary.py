"""tests/test_boundary.py — Unit tests for boundary statistic r = d2/d1 - 1."""

import numpy as np
import pytest

from vecstudy.core.boundary import (
    compute_boundary_statistics,
    summarize_boundary_distribution,
)


def test_equidistant_points_yield_zero_r():
    # Centroids at [-1, 0] and [1, 0]
    centroids = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    # Vectors along the y-axis are equidistant from both centroids
    X = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, -2.5]], dtype=np.float32)

    d1, d2, r, c1, c2 = compute_boundary_statistics(X, centroids, geometry="l2")

    np.testing.assert_allclose(d1, d2, atol=1e-6)
    np.testing.assert_allclose(r, [0.0, 0.0, 0.0], atol=1e-6)
    assert set(c1.tolist() + c2.tolist()) == {0, 1}


def test_point_closer_to_one_centroid():
    centroids = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    # x is at [1, 0]: d1 = 1 to c0, d2 = 9 to c1 => r = 9/1 - 1 = 8.0
    X = np.array([[1.0, 0.0]], dtype=np.float32)

    d1, d2, r, c1, c2 = compute_boundary_statistics(X, centroids, geometry="l2")

    assert c1.tolist() == [0]
    assert c2.tolist() == [1]
    assert d1[0] == pytest.approx(1.0)
    assert d2[0] == pytest.approx(9.0)
    assert r[0] == pytest.approx(8.0)


def test_coinciding_with_centroid_yields_inf_r():
    centroids = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    # Vector exactly on centroid 0
    X = np.array([[0.0, 0.0]], dtype=np.float32)

    d1, d2, r, c1, c2 = compute_boundary_statistics(X, centroids, geometry="l2")

    assert d1[0] == pytest.approx(0.0)
    assert d2[0] == pytest.approx(2.0)
    assert np.isinf(r[0])


def test_chunking_equivalence(synthetic_200):
    X = synthetic_200
    rng = np.random.RandomState(42)
    centroids = rng.randn(8, X.shape[1]).astype(np.float32)

    d1_full, d2_full, r_full, c1_full, c2_full = compute_boundary_statistics(
        X, centroids, geometry="l2", chunk_size=len(X)
    )
    d1_chunk, d2_chunk, r_chunk, c1_chunk, c2_chunk = compute_boundary_statistics(
        X, centroids, geometry="l2", chunk_size=17
    )

    np.testing.assert_allclose(d1_full, d1_chunk, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(d2_full, d2_chunk, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(r_full, r_chunk, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(c1_full, c1_chunk)
    np.testing.assert_array_equal(c2_full, c2_chunk)


def test_ordering_and_distinct_top2(synthetic_200):
    X = synthetic_200
    centroids = X[:10]

    d1, d2, r, c1, c2 = compute_boundary_statistics(X, centroids, geometry="l2")

    # Invariant: d1 <= d2
    assert np.all(d1 <= d2 + 1e-6)
    # Invariant: c1 != c2
    assert np.all(c1 != c2)
    # Invariant: r >= 0 for finite values
    finite = np.isfinite(r)
    assert np.all(r[finite] >= -1e-6)


@pytest.mark.parametrize("geometry", ["l2", "ip"])
def test_both_geometries(geometry):
    centroids = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    X = np.array([[0.8, 0.6]], dtype=np.float32)

    d1, d2, r, c1, c2 = compute_boundary_statistics(X, centroids, geometry=geometry)
    assert len(r) == 1
    assert c1[0] == 0  # Closer to [1, 0] than [0, 1]
    assert c2[0] == 1


def test_invalid_arguments():
    X = np.ones((10, 4), dtype=np.float32)
    c_single = np.ones((1, 4), dtype=np.float32)
    c_dim_mismatch = np.ones((3, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="at least 2 centroids"):
        compute_boundary_statistics(X, c_single)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        compute_boundary_statistics(X, c_dim_mismatch)

    with pytest.raises(ValueError, match="geometry must be"):
        compute_boundary_statistics(X, np.ones((3, 4), dtype=np.float32), geometry="invalid")

    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        compute_boundary_statistics(X, np.ones((3, 4), dtype=np.float32), chunk_size=0)


def test_summarize_boundary_distribution():
    r = np.array([0.01, 0.03, 0.05, 0.10, 0.20, 0.35, np.inf])
    summary = summarize_boundary_distribution(r, thresholds=(0.02, 0.05, 0.10, 0.20))

    assert summary["count"] == 7
    assert summary["finite_count"] == 6
    assert summary["quantiles"]["p50"] == pytest.approx(0.075)

    # Monotonicity check on cumulative thresholds
    fractions = [t["fraction"] for t in summary["thresholds"]]
    assert fractions == sorted(fractions)
    assert summary["thresholds"][0]["count"] == 1  # 0.01 <= 0.02
    assert summary["thresholds"][1]["count"] == 3  # 0.01, 0.03, 0.05 <= 0.05
