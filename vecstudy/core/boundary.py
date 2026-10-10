"""core/boundary.py — Boundary statistics (r = d2/d1 - 1) and empirical CDF.

Computes relative distance to cluster boundaries for vectors against centroids:
    d1 = distance to nearest centroid c1
    d2 = distance to second-nearest centroid c2
    r  = d2 / d1 - 1

Smaller r indicates proximity to the Voronoi boundary between c1 and c2.
The empirical CDF of r corresponds directly to the predicted replication bloat
for any relative overlap threshold delta.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from .distance import sim_many_to_many
from .metrics import Counters


def compute_boundary_statistics(
    X: np.ndarray,
    centroids: np.ndarray,
    *,
    geometry: Literal["l2", "ip"] = "l2",
    chunk_size: int = 32_768,
    counters: Counters | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (d1, d2, r, c1, c2) for all rows of X against centroids.

    Parameters
    ----------
    X : float32 (N, D)
        Input vectors.
    centroids : float32 (K, D)
        Cluster centroids.
    geometry : 'l2' or 'ip'
        'l2': Euclidean distance ||x - c||
        'ip': Inner product distance (1.0 - x . c)
    chunk_size : int
        Batch size to bound peak memory to O(chunk_size * K).
    counters : Counters | None
        Optional counters bucket for distance counting.

    Returns
    -------
    d1 : float64 (N,)
        Distance to nearest centroid.
    d2 : float64 (N,)
        Distance to second-nearest centroid.
    r  : float64 (N,)
        Relative boundary statistic (d2 / d1 - 1).
    c1 : int64 (N,)
        Nearest centroid index.
    c2 : int64 (N,)
        Second-nearest centroid index.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if geometry not in ("l2", "ip"):
        raise ValueError("geometry must be 'l2' or 'ip'")

    X = np.asarray(X, dtype=np.float32)
    centroids = np.asarray(centroids, dtype=np.float32)

    if X.ndim != 2 or centroids.ndim != 2:
        raise ValueError("X and centroids must be 2D arrays")
    if X.shape[1] != centroids.shape[1]:
        raise ValueError(f"Dimension mismatch: X has {X.shape[1]}, centroids has {centroids.shape[1]}")

    N, D = X.shape
    K = centroids.shape[0]
    if K < 2:
        raise ValueError("Boundary statistics require at least 2 centroids")

    c1 = np.empty(N, dtype=np.int64)
    c2 = np.empty(N, dtype=np.int64)
    d1 = np.empty(N, dtype=np.float64)
    d2 = np.empty(N, dtype=np.float64)
    r = np.empty(N, dtype=np.float64)

    centers64 = centroids.astype(np.float64)
    center_norms = np.sum(centers64 ** 2, axis=1) if geometry == "l2" else None

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        block = X[start:end]

        if counters is not None:
            sims = sim_many_to_many(block, centroids, counters, "build").astype(np.float64)
        else:
            sims = (block @ centroids.T).astype(np.float64)

        if geometry == "l2":
            block64 = block.astype(np.float64)
            block_norms = np.sum(block64 ** 2, axis=1)[:, None]
            sq_dist = block_norms + center_norms[None, :] - 2.0 * sims
            np.maximum(sq_dist, 0.0, out=sq_dist)
            dist_matrix = np.sqrt(sq_dist)
        else:
            dist_matrix = 1.0 - sims
            np.maximum(dist_matrix, 0.0, out=dist_matrix)

        # Quickselect top 2 smallest distances in O(K) time
        top2 = np.argpartition(dist_matrix, kth=1, axis=1)[:, :2]
        rows = np.arange(end - start)
        cand0 = dist_matrix[rows, top2[:, 0]]
        cand1 = dist_matrix[rows, top2[:, 1]]

        swap = cand0 > cand1
        best_ids = np.where(swap, top2[:, 1], top2[:, 0])
        second_ids = np.where(swap, top2[:, 0], top2[:, 1])
        best_dist = np.where(swap, cand1, cand0)
        second_dist = np.where(swap, cand0, cand1)

        c1[start:end] = best_ids
        c2[start:end] = second_ids
        d1[start:end] = best_dist
        d2[start:end] = second_dist

        # r = d2 / d1 - 1
        with np.errstate(divide="ignore", invalid="ignore"):
            r_chunk = np.where(
                best_dist == 0.0,
                np.where(second_dist == 0.0, 0.0, np.inf),
                (second_dist / best_dist) - 1.0,
            )
        r[start:end] = r_chunk

    return d1, d2, r, c1, c2


def summarize_boundary_distribution(
    r: np.ndarray,
    thresholds: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
) -> dict:
    """Compute summary quantiles and boundary proximity percentages for r.

    Parameters
    ----------
    r : float64 (N,)
        Relative boundary statistics.
    thresholds : tuple of float
        Thresholds delta for which to calculate the cumulative fraction P(r <= delta).

    Returns
    -------
    dict with:
        count, finite_count, mean, median, std, min, max,
        quantiles: {p1, p5, p10, p25, p50, p75, p90, p95, p99},
        thresholds: list of {threshold, count, fraction, percentage}
    """
    r = np.asarray(r, dtype=np.float64)
    N = len(r)
    r_finite = r[np.isfinite(r)]

    quantiles_keys = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    if len(r_finite):
        quantiles_vals = np.percentile(r_finite, quantiles_keys)
        quantiles = {f"p{k}": float(v) for k, v in zip(quantiles_keys, quantiles_vals)}
    else:
        quantiles = {f"p{k}": 0.0 for k in quantiles_keys}

    threshold_stats = []
    for t in thresholds:
        count = int(np.count_nonzero(r <= t))
        fraction = float(count / N) if N else 0.0
        threshold_stats.append({
            "threshold": float(t),
            "count": count,
            "fraction": fraction,
            "percentage": fraction * 100.0,
        })

    return {
        "count": N,
        "finite_count": len(r_finite),
        "mean": float(np.mean(r_finite)) if len(r_finite) else 0.0,
        "median": float(np.median(r_finite)) if len(r_finite) else 0.0,
        "std": float(np.std(r_finite)) if len(r_finite) else 0.0,
        "min": float(np.min(r_finite)) if len(r_finite) else 0.0,
        "max": float(np.max(r_finite)) if len(r_finite) else 0.0,
        "quantiles": quantiles,
        "thresholds": threshold_stats,
    }
