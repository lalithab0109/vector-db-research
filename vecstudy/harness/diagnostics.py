"""Read-only baseline diagnostics, separate from counted index/search work.

Geometry is segment-local and uses final leaf centroids without normalizing
them. Exact score ties choose the lowest leaf row (posting ID). IP scores use
the index's float32 arithmetic; squared-L2 comparisons use float64 arithmetic.
Only a batch-by-K score matrix is retained, never a dataset-by-K matrix.
"""

from __future__ import annotations

import numpy as np

from ..index.segment import Segment


def reconstruct_owners(segment: Segment, vids: np.ndarray) -> np.ndarray:
    """Return authoritative owners aligned with explicitly requested live IDs.

    Raise ValueError for duplicate live ownership anywhere in the segment or
    missing requested ownership. Callers should pass their expected ID domain
    to detect vectors absent from both members and owned. Tombstones are excluded.
    """
    owners: dict[int, int] = {}
    for pid, posting in enumerate(segment.postings):
        if posting.id != pid:
            raise ValueError("Posting IDs must match final leaf row indices")
        for vid in posting.owned:
            vid = int(vid)
            if not segment.is_alive(vid):
                continue
            if vid in owners:
                raise ValueError(f"Vector {vid} has multiple primary owners")
            owners[vid] = pid

    result = np.empty(len(vids), dtype=np.int64)
    for i, vid in enumerate(vids):
        vid = int(vid)
        if not segment.is_alive(vid):
            raise ValueError(f"Vector {vid} is tombstoned, not a live diagnostic input")
        if vid not in owners:
            raise ValueError(f"Vector {vid} has zero primary owners")
        result[i] = owners[vid]
    return result


def segment_geometry_diagnostics(
    segment: Segment,
    vids: np.ndarray | None = None,
    *,
    batch_size: int = 1024,
) -> dict:
    """Compare IP, squared-L2, and authoritative ownership for live vectors.

    By default evaluate the union of live member/owned IDs, once per vector;
    explicit vids may specify the expected segment domain, including missing IDs.
    Replicas never enlarge the denominator. Empty inputs return zero rates.
    Diagnostic scores do not modify research counters or segment contents.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    centroids = np.asarray(segment.levels[-1].centroids, dtype=np.float32)
    K = len(centroids)
    if K != len(segment.postings):
        raise ValueError("Final leaf centroid count must match posting count")
    if vids is None:
        vids = np.array(sorted({
            int(v) for p in segment.postings for ids in (p.owned, p.members)
            for v in ids if segment.is_alive(int(v))
        }), dtype=np.int64)
    else:
        vids = np.asarray(vids, dtype=np.int64)
    if vids.ndim != 1 or len(np.unique(vids)) != len(vids):
        raise ValueError("Diagnostic vector IDs must be a unique one-dimensional array")
    if np.any(vids < 0) or np.any(vids >= len(segment.vectors)):
        raise ValueError("Diagnostic vector ID is outside the vector array")
    owners = reconstruct_owners(segment, vids)
    N = len(vids)
    if N and not K:
        raise ValueError("Nonempty diagnostics require final leaf centroids")

    geometry_count = owner_count = 0
    centers64 = centroids.astype(np.float64)
    center_norms = np.einsum("ij,ij->i", centers64, centers64)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        X = np.asarray(segment.vectors[vids[start:end]], dtype=np.float32)
        scores = X @ centroids.T
        ip_nearest = np.argmax(scores, axis=1)
        del scores
        # ||x||² is constant across centroids, so omit it from argmin.
        scores = X.astype(np.float64) @ centers64.T
        scores *= -2.0
        scores += center_norms
        l2_nearest = np.argmin(scores, axis=1)
        geometry_count += int(np.count_nonzero(ip_nearest != l2_nearest))
        owner_count += int(np.count_nonzero(owners[start:end] != ip_nearest))

    return {
        "evaluated_vectors": N,
        "leaf_centroid_count": K,
        "geometry_disagreement_count": geometry_count,
        "geometry_disagreement_rate": geometry_count / N if N else 0.0,
        "owner_vs_ip_mismatch_count": owner_count,
        "owner_vs_ip_mismatch_rate": owner_count / N if N else 0.0,
    }


def geometry_diagnostics(units: list[Segment], *, batch_size: int = 1024) -> dict:
    """Aggregate segment-local diagnostics, weighting rates by vector counts.

    Counts are vector occurrences across units, not a global deduplicated domain.
    """
    totals = {key: 0 for key in (
        "evaluated_vectors", "leaf_centroid_count",
        "geometry_disagreement_count", "owner_vs_ip_mismatch_count",
    )}
    for unit in units:
        stats = segment_geometry_diagnostics(unit, batch_size=batch_size)
        for key in totals:
            totals[key] += stats[key]
    N = totals["evaluated_vectors"]
    for prefix in ("geometry_disagreement", "owner_vs_ip_mismatch"):
        totals[f"{prefix}_rate"] = totals[f"{prefix}_count"] / N if N else 0.0
    return totals
