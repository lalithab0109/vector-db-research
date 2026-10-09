"""Isolated R5 clustering experiment; no existing index caller uses this module."""

from __future__ import annotations

import math
import operator

import numpy as np

from .distance import sim_many_to_many
from .metrics import Counters


def capacity_aware_kmeans(
    X: np.ndarray,
    n_clusters: int,
    *,
    target_capacity: int = 800,
    lambda_balance: float = 0.25,
    alpha: float = 1.5,
    geometry: str = "l2",
    max_iter: int = 15,
    seed: int = 0,
    batch_size: int = 1024,
    initial_centroids: np.ndarray | None = None,
    counters: Counters | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Greedy capacity-constrained assignment followed by arithmetic-mean updates.

    Each pass visits vectors in input row order, starting with zero cluster
    counts. Its score is base_cost + lambda_balance * (count-target)/target.
    Base cost is squared L2 or 1-inner-product; clusters already at
    ceil(alpha*target_capacity) receive infinity. Exact ties choose the lowest
    cluster ID. Counts update after EACH vector, including within a score batch.

    Initial centers are seeded input-row samples (without replacement when
    K<=N, with replacement otherwise), identical for both geometries. Optional
    initial_centroids provides controlled experimental initialization. Empty
    clusters retain their preceding center; empty X returns zero centers unless
    initial centers were supplied. Centers are never normalized.

    Stop on identical successive label arrays or max_iter. Returned centers
    correspond to returned labels, except retained empty centers; there is no
    unconstrained final reassignment. Stable labels do not imply an optimum.
    Fixed input order, seed, and numerical environment give deterministic output.

    Memory beyond input/centers is O(N + K*D + batch_size*(K+D)); no persistent
    N*K matrix. Dot products use the existing float32 counted kernel; L2 norm
    terms and balance scores use float64, with negative roundoff L2 costs
    clipped to zero. All scored pairs count under the build bucket.
    """
    for name, value in (("n_clusters", n_clusters), ("target_capacity", target_capacity),
                        ("max_iter", max_iter), ("batch_size", batch_size)):
        try:
            integer = operator.index(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if integer <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if geometry not in ("l2", "ip"):
        raise ValueError("geometry must be 'l2' or 'ip'")
    if not math.isfinite(lambda_balance) or lambda_balance < 0:
        raise ValueError("lambda_balance must be finite and nonnegative")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    capacity = alpha * target_capacity
    if not math.isfinite(capacity):
        raise ValueError("alpha * target_capacity must be finite")
    max_capacity = math.ceil(capacity)
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("X must be a two-dimensional matrix with positive dimension")
    N, D = X.shape
    if N > n_clusters * max_capacity:
        raise ValueError(
            f"Impossible capacity: {N} vectors exceed {n_clusters} clusters "
            f"* C_max={max_capacity}"
        )
    # Check input in batches as well, avoiding an N*D boolean temporary.
    for start in range(0, N, batch_size):
        if not np.isfinite(X[start:start + batch_size]).all():
            raise ValueError("X must contain only finite values")

    if initial_centroids is not None:
        centers = np.array(initial_centroids, dtype=np.float32, copy=True)
        if centers.shape != (n_clusters, D) or not np.isfinite(centers).all():
            raise ValueError("initial_centroids must be finite with shape (n_clusters, D)")
    elif N:
        rng = np.random.RandomState(seed)
        centers = X[rng.choice(N, size=n_clusters, replace=n_clusters > N)].copy()
    else:
        centers = np.zeros((n_clusters, D), dtype=np.float32)

    counters = counters if counters is not None else Counters()
    previous_labels = None
    labels = np.empty(N, dtype=np.int64)
    sizes = np.zeros(n_clusters, dtype=np.int64)
    iterations = 0
    converged = N == 0
    for iteration in range(max_iter if N else 0):
        sizes.fill(0)
        sums = np.zeros((n_clusters, D), dtype=np.float64)
        center_norms = np.einsum("ij,ij->i", centers.astype(np.float64),
                                 centers.astype(np.float64)) if geometry == "l2" else None
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            block = X[start:end]
            costs = sim_many_to_many(block, centers, counters, "build").astype(np.float64)
            if geometry == "l2":
                block64 = block.astype(np.float64)
                costs *= -2.0
                costs += center_norms
                costs += np.einsum("ij,ij->i", block64, block64)[:, None]
                np.maximum(costs, 0.0, out=costs)
            else:
                costs *= -1.0
                costs += 1.0
            for local_i, row in enumerate(costs):
                scores = row + lambda_balance * ((sizes - target_capacity) / target_capacity)
                scores[sizes >= max_capacity] = np.inf
                pid = int(np.argmin(scores))
                # Given N<=K*C_max and unit-size assignments, a feasible cluster
                # always remains. Never fall back to an overflowing destination.
                if not np.isfinite(scores[pid]):
                    raise ValueError("No finite feasible assignment score")
                labels[start + local_i] = pid
                sizes[pid] += 1
            np.add.at(sums, labels[start:end], block)
        nonempty = sizes > 0
        centers[nonempty] = (sums[nonempty] / sizes[nonempty, None]).astype(np.float32)
        iterations = iteration + 1
        if previous_labels is not None and np.array_equal(labels, previous_labels):
            converged = True
            break
        previous_labels = labels.copy()

    info = {
        "iterations": iterations,
        "converged": converged,
        "cluster_sizes": sizes.copy(),
        "max_capacity": max_capacity,
        "capacity_violations": int(np.count_nonzero(sizes > max_capacity)),
    }
    return centers, labels, info
