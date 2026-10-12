"""Controlled Part 7 treatments on one replica-free R5 snapshot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from ..core.boundary import compute_boundary_statistics
from ..core.config import QueryConfig
from ..core.metrics import Counters
from ..harness.datasets import recall_at_k
from ..index.overlap import apply_budget_overlap, apply_fixed_overlap
from ..index.query import search

TREATMENTS = (
    ('R5 no overlap', 'none', None),
    ('Fixed delta 0.10', 'fixed', .10),
    ('Fixed delta 0.20', 'fixed', .20),
    ('Dynamic 5%', 'dynamic', .05),
    ('Dynamic 10%', 'dynamic', .10),
    ('Dynamic 20%', 'dynamic', .20),
    ('Dynamic 30%', 'dynamic', .30),
)


def snapshot_digest(snapshot):
    """Fingerprint owners, members, centroids, and hierarchy including shapes."""
    h = hashlib.sha256()
    def add(array):
        a = np.asarray(array)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    for p in snapshot.postings:
        h.update(str(p.id).encode())
        for a in (p.owned, p.members, p.centroid):
            add(a)
    for level in snapshot.levels:
        add(level.centroids)
        h.update(str(len(level.children)).encode())
        for child in level.children:
            add(child)
    h.update(str(sorted(snapshot.tombstones)).encode())
    return h.hexdigest()


def validate_treatment(snapshot, treatment, vids, r, c1, c2, delta, *, rng=False, rng_alpha=1.0):
    """Fail before interpreting results if snapshot or replica invariants break."""
    n = len(vids)
    # Dense in population size, independent of maximum global ID.
    sorted_ids = np.sort(vids)
    occurrences = np.zeros(n, dtype=np.int8)
    owners = np.empty(n, dtype=np.int64)
    for old, new in zip(snapshot.postings, treatment.postings, strict=True):
        np.testing.assert_array_equal(old.members, old.owned)
        np.testing.assert_array_equal(new.owned, old.owned)
        np.testing.assert_array_equal(new.centroid, old.centroid)
        assert old.id == new.id
        assert len(new.members) == len(np.unique(new.members))
        indices = np.searchsorted(sorted_ids, new.members)
        assert np.all(indices < n)
        np.testing.assert_array_equal(sorted_ids[indices], new.members)
        np.add.at(occurrences, indices, 1)
        owners[np.searchsorted(sorted_ids, old.owned)] = old.id
    selected = np.zeros(n, dtype=bool) if delta is None else r <= delta
    population_order = np.searchsorted(sorted_ids, vids)
    destinations = np.where(owners[population_order] == c1, c2, c1)
    assert np.all(destinations != owners[population_order])
    if rng:
        from ..index.overlap import rng_keep_candidates
        selected &= rng_keep_candidates(snapshot.vectors[vids], snapshot.levels[-1].centroids,
                                        owners[population_order], destinations, selected, rng_alpha=rng_alpha)
    np.testing.assert_array_equal(occurrences[population_order], 1 + selected.astype(np.int8))
    for pid, p in enumerate(treatment.postings):
        expected_replicas = vids[selected & (destinations == pid)]
        actual_replicas = p.members[~np.isin(p.members, p.owned)]
        np.testing.assert_array_equal(np.sort(actual_replicas), np.sort(expected_replicas))
    for a, b in zip(snapshot.levels, treatment.levels, strict=True):
        np.testing.assert_array_equal(a.centroids, b.centroids)
        for x, y in zip(a.children, b.children, strict=True):
            np.testing.assert_array_equal(x, y)
    assert snapshot.tombstones == treatment.tombstones
    assert treatment.vectors is snapshot.vectors


def evaluate_queries(segment, queries, ground_truth, query_config):
    """Existing search and recall, with exact aggregate occurrence scan counts."""
    results = []
    total_scanned = 0
    for q in queries:
        counters = Counters()
        results.append([vid for vid, _ in search(q, [segment], query_config, counters)])
        total_scanned += counters.vectors_scanned
    count = len(queries)
    if not count or len(ground_truth) != count:
        raise ValueError('Queries must be nonempty and aligned with ground truth')
    return dict(recall_at_10=recall_at_k(results, ground_truth, query_config.k),
                total_scanned=total_scanned, query_count=count,
                scans_per_query=total_scanned / count)


def run_overlap_comparison(snapshot, vector_ids, queries, ground_truth,
                           query_config: QueryConfig, *, progress=None, treatments=TREATMENTS,
                           boundary_statistics=None, calibrated_delta=None):
    """Compute boundary statistics once; evaluate independent membership variants."""
    if query_config.k != 10:
        raise ValueError('Part 7 requires k=10')
    if not len(queries) or len(ground_truth) != len(queries):
        raise ValueError('Queries must be nonempty and aligned with ground truth')
    vids = np.asarray(vector_ids)
    fingerprint = snapshot_digest(snapshot)
    if boundary_statistics is None:
        boundary_statistics = compute_boundary_statistics(
            snapshot.vectors[vids], snapshot.levels[-1].centroids, geometry='l2')
    _, _, r, c1, c2 = boundary_statistics
    rows = []
    for treatment in treatments:
        name, mode, parameter = treatment[:3]
        rng_alpha = treatment[3] if len(treatment) == 4 and treatment[3] is not None else 1.0
        if progress:
            progress(f'Evaluating {name}')
        if mode in ('dynamic', 'dynamic_rng'):
            if calibrated_delta is None:
                overlay, metrics = apply_budget_overlap(snapshot, vids, r, c1, c2, parameter, rng=(mode == 'dynamic_rng'), rng_alpha=rng_alpha)
            else:
                overlay, metrics = apply_fixed_overlap(snapshot, vids, r, c1, c2, calibrated_delta, rng=(mode == 'dynamic_rng'), rng_alpha=rng_alpha)
                metrics.pop('delta')
                metrics.update(derived_delta=calibrated_delta, target_bloat=parameter, overshoot=metrics['realized_bloat']-parameter)
            delta = metrics.pop('derived_delta')
        else:
            overlay, metrics = apply_fixed_overlap(snapshot, vids, r, c1, c2, parameter)
            delta = metrics.pop('delta')
            metrics.update(target_bloat=0.0 if mode == 'none' else None,
                           overshoot=0.0 if mode == 'none' else None)
        validate_treatment(snapshot, overlay, vids, r, c1, c2, delta, rng=(mode == 'dynamic_rng'), rng_alpha=rng_alpha)
        assert snapshot_digest(snapshot) == fingerprint, 'Snapshot mutated'
        scores = evaluate_queries(overlay, queries, ground_truth, query_config)
        count = metrics['replicated_count']
        assert metrics['selected_count'] >= count
        if mode != 'dynamic_rng':
            assert metrics['selected_count'] == count
        else:
            assert metrics['rng_accepted_count'] + metrics['rng_rejected_count'] == metrics['selected_count']
        assert np.isclose(metrics['realized_bloat'], count / len(vids), rtol=0, atol=1e-12)
        metrics['actual_bloat'] = metrics.pop('realized_bloat')
        budget = metrics['target_bloat']
        tie_overshoot = (count - int(np.ceil(len(vids) * budget))) / len(vids) if mode in ('dynamic', 'dynamic_rng') else None
        row = dict(method=name, threshold_mode=mode, delta=delta, **metrics, **scores,
                   tie_overshoot=tie_overshoot, npost=query_config.n_postings,
                   k=query_config.k, leaf_count=len(snapshot.postings),
                   owned_vector_count=len(vids), snapshot_sha256=fingerprint,
                   boundary_geometry='Euclidean L2', rng=(mode == 'dynamic_rng'))
        if mode == 'dynamic_rng':
            row['tie_overshoot'] = (row['selected_count'] - int(np.ceil(len(vids)*budget))) / len(vids)
        if len(treatment) == 4:
            row['rng_alpha'] = rng_alpha if row['rng'] else None
        rows.append(row)
        assert snapshot_digest(snapshot) == fingerprint, 'Snapshot mutated during search'
        if progress:
            progress(json.dumps(row))
    dynamic = [row for row in rows if row['threshold_mode'] == 'dynamic']
    assert [row['delta'] for row in dynamic] == sorted(row['delta'] for row in dynamic)
    assert [row['actual_bloat'] for row in dynamic] == sorted(row['actual_bloat'] for row in dynamic)
    return rows


def save_results(rows, provenance, out_dir, *, stem='results'):
    """Write exact numeric JSON and CSV, independent of plotting."""
    import csv
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{stem}.json').write_text(json.dumps(dict(provenance=provenance, results=rows), indent=2) + '\n')
    with (out_dir / f'{stem}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    for mode, marker, color in [('none', 's', 'black'), ('fixed', '^', 'tab:orange'),
                                ('dynamic', 'o', 'tab:blue')]:
        group = [r for r in rows if r['threshold_mode'] == mode]
        ax.scatter([r['actual_bloat'] * 100 for r in group],
                   [r['recall_at_10'] for r in group], marker=marker, color=color,
                   label=mode, s=65)
        for r in group:
            ax.annotate(r['method'], (r['actual_bloat'] * 100, r['recall_at_10']),
                        xytext=(5, 6), textcoords='offset points', fontsize=8)
    curve = sorted(rows, key=lambda r: r['actual_bloat'])
    ax.plot([r['actual_bloat'] * 100 for r in curve], [r['recall_at_10'] for r in curve],
            color='gray', alpha=.5, linewidth=1, zorder=0)
    ax.set(xlabel='Actual bloat / replication fraction (%)', ylabel='Recall@10',
           title='Part 7: same R5 snapshot, L2 overlap, RNG disabled')
    ax.grid(alpha=.25)
    ax.legend()
    ax.margins(x=.15, y=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
