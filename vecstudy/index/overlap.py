"""Post-build, owner-preserving budget overlap; no clustering or search changes."""
from __future__ import annotations

import copy
import numpy as np

from ..core.boundary import choose_delta
from ..core.types import Posting, HierarchyLevel
from .segment import Segment


def apply_budget_overlap(
    snapshot: Segment,
    vector_ids: np.ndarray,
    r: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    target_bloat: float,
    *, rng: bool = False, rng_alpha: float = 1.0,
) -> tuple[Segment, dict]:
    """Return an independent membership treatment and measured calibration metrics.

    Inputs are aligned by explicit global vector IDs and must cover all owners.
    Start from owned-only lists, including on repeated application, so replicas
    never accumulate. Requires a live, sealed snapshot without tombstones.
    The caller supplies existing Part-2 L2 statistics against final leaf centroids.
    """
    delta = choose_delta(r, target_bloat)
    overlay, metrics = apply_fixed_overlap(snapshot, vector_ids, r, c1, c2, delta, rng=rng, rng_alpha=rng_alpha)
    metrics.pop("delta")
    metrics.update(target_bloat=float(target_bloat), derived_delta=delta,
                   overshoot=metrics["realized_bloat"] - float(target_bloat))
    return overlay, metrics


def apply_fixed_overlap(
    snapshot: Segment,
    vector_ids: np.ndarray,
    r: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    delta: float | None,
    *, rng: bool = False, rng_alpha: float = 1.0,
) -> tuple[Segment, dict]:
    """Use the same owner-preserving overlay for an explicit L2 r threshold.

    None disables overlap; delta=0 selects zero-valued boundary statistics.
    Every call resets memberships to owned-only before adding selected replicas.
    """
    rng_alpha = validate_rng_alpha(rng_alpha)
    # Validate r without finite-only filtering or changing its population.
    choose_delta(r, 0)
    if delta is not None:
        delta = float(delta)
        if not np.isfinite(delta) or delta < 0:
            raise ValueError("delta must be None or finite and nonnegative")
    r = np.asarray(r, dtype=np.float64)
    vids, first, second = map(np.asarray, (vector_ids, c1, c2))
    n, k = len(r), len(snapshot.postings)
    for name, values in (("vector_ids", vids), ("c1", first), ("c2", second)):
        if values.shape != (n,) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} must be an aligned one-dimensional integer array")
    if snapshot.tombstones:
        raise ValueError("Budget overlap requires a snapshot without tombstones")
    if not snapshot.levels or len(snapshot.levels[-1].centroids) != k:
        raise ValueError("Final leaf centroids must align with postings")
    if (np.any(vids < 0) or np.any(vids >= len(snapshot.vectors))
            or len(np.unique(vids)) != n):
        raise ValueError("vector_ids must be unique and within the vector array")
    if (np.any(first < 0) or np.any(first >= k) or np.any(second < 0)
            or np.any(second >= k) or np.any(first == second)):
        raise ValueError("c1/c2 must be distinct valid final leaf IDs")
    owned_ids = []
    owner_labels = []
    for pid, posting in enumerate(snapshot.postings):
        if posting.id != pid or not np.array_equal(
                posting.centroid, snapshot.levels[-1].centroids[pid]):
            raise ValueError("Posting IDs/centroids must match final leaves")
        owned = np.asarray(posting.owned)
        if owned.ndim != 1 or not np.issubdtype(owned.dtype, np.integer):
            raise ValueError("Owners must be one-dimensional integer IDs")
        if not np.all(np.isin(owned, posting.members)):
            raise ValueError("Owned IDs must be posting members")
        owned_ids.append(owned)
        owner_labels.append(np.full(len(owned), pid, dtype=np.int64))
    all_ids = np.concatenate(owned_ids)
    labels = np.concatenate(owner_labels)
    order = np.argsort(all_ids)
    if len(all_ids) != n or not np.array_equal(np.sort(all_ids), np.sort(vids)):
        raise ValueError("vector_ids must cover authoritative owners exactly once")
    owners = labels[order][np.searchsorted(all_ids[order], vids)]
    selected = np.zeros(n, dtype=bool) if delta is None else r <= delta
    destinations = np.where(owners == first, second, first)
    accepted = selected.copy()
    if rng:
        accepted &= rng_keep_candidates(snapshot.vectors[vids],
            snapshot.levels[-1].centroids, owners, destinations, selected, rng_alpha=rng_alpha)
    replica_ids = vids[accepted]
    replica_destinations = destinations[accepted]
    replica_order = np.argsort(replica_destinations, kind="stable")
    replica_ids = replica_ids[replica_order]
    replica_destinations = replica_destinations[replica_order]
    cuts = np.searchsorted(replica_destinations, np.arange(k + 1))
    postings = []
    for pid, posting in enumerate(snapshot.postings):
        members = np.concatenate((posting.owned, replica_ids[cuts[pid]:cuts[pid+1]]))
        postings.append(Posting(pid, posting.centroid.copy(), members,
                                posting.owned.copy()))
    levels = [HierarchyLevel(level.centroids.copy(),
                             [child.copy() for child in level.children])
              for level in snapshot.levels]
    overlay = Segment(levels, postings, snapshot.vectors,
                      tombstones=snapshot.tombstones.copy())
    overlay.build_stats = copy.deepcopy(snapshot.build_stats)
    count = int(np.count_nonzero(selected))
    replicated = sum(len(p.members) - len(p.owned) for p in postings)
    fraction = replicated / n
    bloat = sum(len(p.members) for p in postings) / n - 1.0
    metrics = dict(delta=delta,
                   selected_count=count, selected_fraction=count / n,
                   replicated_count=replicated, replication_fraction=fraction,
                   realized_bloat=bloat)
    if rng:
        rejected = count - replicated
        metrics.update(candidate_replica_count=count, rng_accepted_count=replicated,
                       rng_rejected_count=rejected,
                       rng_acceptance_fraction=replicated/count if count else 0.0,
                       rng_rejection_fraction=rejected/count if count else 0.0)
    return overlay, metrics



def rng_keep_candidates(vectors, centroids, owners, destinations, selected, *, rng_alpha=1.0):
    """Keep candidates iff ||x-candidate|| <= rng_alpha * ||owner-candidate||.

    One-candidate adaptation of SPANN representative filtering. Equality passes.
    Explicit Euclidean float64 differences, chunked to bound temporary memory.
    No ownership decisions, random tie-breaking, or budget recalibration.
    """
    rng_alpha = validate_rng_alpha(rng_alpha)
    X = np.asarray(vectors, dtype=np.float64)
    C = np.asarray(centroids, dtype=np.float64)
    owners, destinations, selected = map(np.asarray, (owners, destinations, selected))
    if X.ndim != 2 or C.ndim != 2 or X.shape[1] != C.shape[1]:
        raise ValueError('Vectors/centroids must be compatible 2D arrays')
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(C)):
        raise ValueError('RNG geometry must be finite')
    n = len(X)
    if selected.shape != (n,) or selected.dtype != np.bool_:
        raise ValueError('selected must be an aligned boolean array')
    for ids in (owners, destinations):
        if ids.shape != (n,) or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError('Representative IDs must be aligned integers')
        if np.any(ids < 0) or np.any(ids >= len(C)):
            raise ValueError('Representative IDs outside centroid range')
    if np.any(owners == destinations):
        raise ValueError('Candidate must differ from authoritative owner')
    keep = np.zeros(n, dtype=bool)
    candidates = np.flatnonzero(selected)
    for start in range(0, len(candidates), 8192):
        ids = candidates[start:start+8192]
        candidate_centers = C[destinations[ids]]
        with np.errstate(over='ignore', invalid='ignore'):
            dx = np.linalg.norm(X[ids] - candidate_centers, axis=1)
            dc = np.linalg.norm(C[owners[ids]] - candidate_centers, axis=1)
        if not np.all(np.isfinite(dx)) or not np.all(np.isfinite(dc)):
            raise ValueError('RNG distances must be finite')
        with np.errstate(over="ignore"):
            keep[ids] = dx <= rng_alpha * dc
    return keep



def validate_rng_alpha(rng_alpha):
    """RNG relaxation is distinct from R5's unchanged capacity alpha."""
    value = float(rng_alpha)
    if not np.isfinite(value) or value < 1:
        raise ValueError('rng_alpha must be finite and >= 1')
    return value
