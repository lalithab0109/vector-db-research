"""L2-only R5 leaf construction and deterministic capacity-safe cleanup.

Initial K is ceil(N/C_target), increased when needed for hard capacity.
Cleanup labels remain authoritative even if their owner is not the geometric
nearest centroid. Routing is built later and is intentionally unchanged.
Split/merge statistics describe this leaf construction, not maintenance history.
"""
import math
import numpy as np

from .capacity_clustering import capacity_aware_kmeans
from .distance import sim_one_to_many


def finalize_r5_leaves(X, labels, n_clusters, cfg, counters):
    """Split owned groups recursively, then merge tiny groups once safely.

    Empty groups are removed. Splits only partition source owners. Every split
    strictly reduces group size; a deterministic halving fallback protects
    against degenerate/nonprogressing clustering. Merges never exceed C_max
    and strictly reduce group count, so no split/merge cycle is possible.
    """
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels)
    if labels.shape != (len(X),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must contain one integer cluster ID per vector")
    if np.any(labels < 0) or np.any(labels >= n_clusters):
        raise ValueError("labels outside cluster range")
    cap = math.ceil(cfg.alpha * cfg.target_capacity)
    groups = []
    splits = 0
    pending = [np.flatnonzero(labels == pid) for pid in range(n_clusters)][::-1]
    while pending:
        ids = pending.pop()
        if not len(ids):
            continue
        if len(ids) <= cap:
            groups.append(ids)
            continue
        # Temporary binary split capacity permits oversized children, which
        # are then recursively processed against the actual final capacity.
        _, sublabels, _ = capacity_aware_kmeans(
            X[ids], 2, target_capacity=max(cfg.target_capacity, math.ceil(len(ids) / 2)),
            alpha=1, lambda_balance=cfg.lambda_balance, geometry="l2",
            seed=cfg.seed + splits, max_iter=cfg.max_iter, counters=counters,
        )
        children = [ids[sublabels == i] for i in range(2)]
        if any(len(child) == len(ids) for child in children):
            children = list(np.array_split(ids, 2))
        splits += 1
        pending.extend(children[::-1])

    centers = [X[ids].mean(axis=0).astype(np.float32) for ids in groups]
    merges = 0
    minimum = cfg.tiny_cluster_fraction * cfg.target_capacity
    for src in range(len(groups)):
        if not 0 < len(groups[src]) < minimum:
            continue
        legal = [dst for dst in range(len(groups)) if dst != src and len(groups[dst])
                 and len(groups[src]) + len(groups[dst]) <= cap]
        if not legal:
            continue
        candidates = np.array([centers[dst] for dst in legal], dtype=np.float32)
        sims = sim_one_to_many(centers[src], candidates, counters, "build")
        costs = np.sum(candidates.astype(np.float64) ** 2, axis=1) - 2 * sims
        dst = legal[int(np.argmin(costs))]  # legal IDs are ascending: stable ties
        groups[dst] = np.sort(np.concatenate([groups[dst], groups[src]]))
        centers[dst] = X[groups[dst]].mean(axis=0).astype(np.float32)
        groups[src] = np.empty(0, dtype=np.int64)
        merges += 1

    groups = [ids for ids in groups if len(ids)]
    centers = np.array([X[ids].mean(axis=0) for ids in groups], dtype=np.float32).reshape(-1, X.shape[1])
    final_labels = np.empty(len(X), dtype=np.int64)
    for pid, ids in enumerate(groups):
        final_labels[ids] = pid
    return centers, final_labels, {"oversized_splits": splits, "tiny_merges": merges}


def build_r5_leaves(X, cfg, counters):
    cap = math.ceil(cfg.alpha * cfg.target_capacity)
    K = max(1, math.ceil(len(X) / cfg.target_capacity), math.ceil(len(X) / cap))
    _, labels, _ = capacity_aware_kmeans(
        X, K, target_capacity=cfg.target_capacity, lambda_balance=cfg.lambda_balance,
        alpha=cfg.alpha, geometry="l2", seed=cfg.seed, max_iter=cfg.max_iter,
        counters=counters,
    )
    return finalize_r5_leaves(X, labels, K, cfg, counters)
