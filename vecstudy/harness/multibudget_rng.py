"""Bounded multi-budget experiment using the shared overlap evaluator."""
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from ..core.boundary import compute_boundary_statistics, choose_delta
from ..index.overlap import rng_keep_candidates
from .overlap_experiment import run_overlap_comparison, snapshot_digest, save_results
from .relaxed_rng import enrich, best_alpha, RETENTION_TARGET

BUDGETS = (.1, .2, .3, .45)
NEW_BUDGETS = (.1, .3, .45)
CHEAP_ALPHAS = (1., 1.25, 1.5, 1.6, 1.75, 2., 2.5, 3., 5.)


class EvaluationLimit(RuntimeError):
    pass


class QueryBudget:
    def __init__(self):
        self.counts = {b: 0 for b in BUDGETS}

    @property
    def expensive_query_evaluations(self):
        return sum(self.counts.values())

    def reserve(self, budget):
        if budget not in NEW_BUDGETS:
            raise EvaluationLimit('20% is reuse-only; no new evaluation allowed')
        if self.counts[budget] >= 6 or self.expensive_query_evaluations >= 18:
            raise EvaluationLimit('Full-query computation limit reached')
        self.counts[budget] += 1


def bounded_search(evaluate):
    """Return measured rows and bracket; never request more than six points."""
    rows = []
    def measure(a, stage):
        row = evaluate(a, stage)
        rows.append(row)
        return row['recall_gain_retention'] >= RETENTION_TARGET
    low = high = None
    first = measure(1.5, 'initial')
    second = measure(1.75, 'initial')
    if first:
        if measure(1.25, 'lower_boundary'):
            if not measure(1., 'lower_boundary'):
                low, high = 1., 1.25
        else:
            low, high = 1.25, 1.5
    elif second:
        low, high = 1.5, 1.75
    elif measure(2., 'upper_boundary'):
        low, high = 1.75, 2.
    elif measure(3., 'upper_boundary'):
        low, high = 2., 3.
    initial = None if low is None else [low, high]
    if low is not None:
        for _ in range(min(3, 6-len(rows))):
            midpoint = (low+high)/2
            if measure(midpoint, 'refinement'):
                high = midpoint
            else:
                low = midpoint
    return rows, dict(initial_bracket=initial, final_bracket=None if low is None else [low, high],
                      refinement_count=sum(r['stage']=='refinement' for r in rows),
                      refinement_limited_by_cap=low is not None and len(rows)==6 and sum(r['stage']=='refinement' for r in rows)<3)


def stage_a(snapshot, vids, stats, references):
    _, _, ratios, c1, c2 = stats
    owner_by_id = np.empty(len(snapshot.vectors), dtype=np.int64)
    for posting in snapshot.postings:
        owner_by_id[posting.owned] = posting.id
    owners = owner_by_id[vids]
    destinations = np.where(owners == c1, c2, c1)
    assert np.all(destinations != owners)
    rows, candidates = [], {}
    digest = snapshot_digest(snapshot)
    for budget in NEW_BUDGETS:
        ref = references[budget]
        delta = choose_delta(ratios, budget)
        assert delta == ref['delta']
        selected = ratios <= delta
        count = int(selected.sum())
        assert count == ref['selected_count'] == round(len(vids)*budget)
        candidates[budget] = selected
        previous = np.zeros(len(vids), dtype=bool)
        for alpha in CHEAP_ALPHAS:
            mask = rng_keep_candidates(snapshot.vectors[vids], snapshot.levels[-1].centroids,
                                       owners, destinations, selected, rng_alpha=alpha)
            assert not np.any(previous & ~mask) and not np.any(mask & ~selected)
            accepted = int(mask.sum())
            rows.append(dict(target_bloat=budget, delta=delta, rng_alpha=alpha,
                candidate_count=count, accepted_replica_count=accepted, rejected_replica_count=count-accepted,
                acceptance_percent=100*accepted/count, rejection_percent=100*(count-accepted)/count,
                actual_bloat=accepted/len(vids), bloat_reduction=(count-accepted)/len(vids),
                relative_bloat_reduction=(count-accepted)/count,
                candidate_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
                accepted_sha256=hashlib.sha256(mask.tobytes()).hexdigest()))
            previous = mask
    assert snapshot_digest(snapshot) == digest
    return rows, candidates, owners, destinations


def common_measured(rows):
    qualifying = {b: {r['rng_alpha'] for r in rows if r['target_bloat']==b
                       and r['rng_alpha'] is not None and r['recall_gain_retention'] >= RETENTION_TARGET} for b in BUDGETS}
    common = set.intersection(*qualifying.values())
    alpha = min(common) if common else None
    return dict(rng_alpha=alpha, points=[r for r in rows if alpha is not None and r['rng_alpha']==alpha])


def run_multibudget(snapshot, vids, queries, gt, query_cfg, provenance, out_dir, root):
    experiment_start = time.perf_counter()
    root, out_dir = Path(root), Path(out_dir)
    paths = [p for folder in ('r2_part6','r2_part7','r2_part8','r2_rng','r2_relaxed_rng')
             for p in (root/'results'/folder).glob('*') if p.is_file()]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    part8 = json.loads((root/'results/r2_part8/sweep_results.json').read_text())
    old20 = json.loads((root/'results/r2_relaxed_rng/relaxed_rng_results.json').read_text())
    refs = {b: next(r for r in part8['results'] if r['target_bloat']==b) for b in BUDGETS}
    baseline = part8['results'][0]
    digest = snapshot_digest(snapshot)
    for artifact in (part8, old20):
        for key in ('dataset_checksum','query_checksum','ground_truth_checksum','query_configuration'):
            assert artifact['provenance'][key] == provenance[key], key
        config = dict(artifact['provenance']['configuration'])
        if 'r5_balance_alpha' in config:
            config['alpha'] = config.pop('r5_balance_alpha')
        assert config == provenance['configuration']
        assert all(r['snapshot_sha256']==digest for r in artifact['results'])
    old_reference = old20['provenance']['baseline_reference']
    for key in ('recall_at_10','total_scanned','scans_per_query','snapshot_sha256'):
        assert old_reference[key]==baseline[key]
    endpoint20 = next(r for r in old20['results'] if r['rng_alpha'] is None)
    for key in ('delta','selected_count','replicated_count','actual_bloat','recall_at_10','total_scanned','scans_per_query'):
        assert endpoint20[key]==refs[.2][key]
    stats = compute_boundary_statistics(snapshot.vectors[vids], snapshot.levels[-1].centroids, geometry='l2')
    cheap, candidates, owners, destinations = stage_a(snapshot, vids, stats, refs)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir/'stage_a_acceptance.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(cheap[0])); writer.writeheader(); writer.writerows(cheap)
    print('Stage A complete: 27 cheap evaluations, zero query evaluations', flush=True)
    rows = [dict(r, stage='reused_20_percent', original_stage=r['stage'], reused=True,
                 new_query_evaluation=False) for r in old20['results'] if r['rng_alpha'] is not None]
    counter = QueryBudget(); searches = {}; partial = None
    try:
        for budget in NEW_BUDGETS:
            measured_masks = {}
            def evaluate(alpha, stage):
                counter.reserve(budget)
                print(f'Full query evaluation {counter.expensive_query_evaluations}: budget={budget}, alpha={alpha}', flush=True)
                row = run_overlap_comparison(snapshot, vids, queries, gt, query_cfg,
                    treatments=[(f'Dynamic {budget:g}, relaxed RNG {alpha:g}', 'dynamic_rng', budget, alpha)],
                    boundary_statistics=stats, calibrated_delta=refs[budget]['delta'],
                    progress=lambda message: print(message, flush=True))[0]
                mask = rng_keep_candidates(snapshot.vectors[vids], snapshot.levels[-1].centroids,
                                           owners, destinations, candidates[budget], rng_alpha=alpha)
                assert row['replicated_count'] == int(mask.sum())
                for previous_alpha, previous_mask in measured_masks.items():
                    lower, upper = (previous_mask, mask) if previous_alpha < alpha else (mask, previous_mask)
                    assert not np.any(lower & ~upper), 'Measured accepted masks must be nested'
                measured_masks[alpha] = mask
                row.update(stage=stage, reused=False, new_query_evaluation=True,
                    r5_balance_alpha=provenance['configuration']['alpha'],
                    candidate_sha256=hashlib.sha256(candidates[budget].tobytes()).hexdigest(),
                    accepted_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                    candidate_destinations_sha256=hashlib.sha256(destinations[candidates[budget]].tobytes()).hexdigest())
                enrich(row, refs[budget], baseline)
                rows.append(row)
                return row
            _, searches[str(budget)] = bounded_search(evaluate)
    except EvaluationLimit as exc:
        partial = str(exc)
    rows.sort(key=lambda r: (r['target_bloat'], r['rng_alpha']))
    for budget in BUDGETS:
        points = [r for r in rows if r['target_bloat']==budget]
        assert [r['replicated_count'] for r in points] == sorted(r['replicated_count'] for r in points)
        assert [r['actual_bloat'] for r in points] == sorted(r['actual_bloat'] for r in points)
    assert snapshot_digest(snapshot)==digest
    for path, sha in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
    best = [best_alpha([r for r in rows if r['target_bloat']==b]) for b in BUDGETS]
    selected20 = json.loads((root/'results/r2_relaxed_rng/best_alpha_analysis.json').read_text())['best_measured_point']
    assert best[1]['rng_alpha']==selected20['rng_alpha'] and best[1]['recall_at_10']==selected20['recall_at_10']
    analysis = dict(retention_threshold=RETENTION_TARGET,
        selection_criterion='Measured qualifying points; minimum bloat, then scans/query, then lower alpha',
        best_points=best, no_rng_references=[refs[b] for b in BUDGETS], baseline_reference=baseline,
        common_measured=common_measured(rows), searches=searches, partial_reason=partial,
        expensive_query_evaluations=counter.expensive_query_evaluations,
        evaluations_by_budget={str(b): n for b,n in counter.counts.items()}, cheap_evaluations=len(cheap))
    prov = dict(provenance, reference_artifact_sha256=hashes, prior_artifacts_unchanged=True,
                snapshot_sha256=digest, budget_calibration_count=3,
                expensive_query_evaluations=counter.expensive_query_evaluations,
                stage_a_completed_before_queries=True,
                timestamp_end_utc=datetime.now(timezone.utc).isoformat(),
                experiment_seconds=time.perf_counter()-experiment_start)
    prov['configuration']=dict(prov['configuration']); prov['configuration']['r5_balance_alpha']=prov['configuration'].pop('alpha')
    prov['source_sha256']={**prov['source_sha256'], 'vecstudy/harness/multibudget_rng.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save_results(rows, prov, out_dir, stem='multibudget_results')
    (out_dir/'best_alpha_by_budget.json').write_text(json.dumps(analysis, indent=2)+'\n')
    plot_saved(out_dir)
    print(json.dumps(analysis), flush=True)
    return rows, analysis


def plot_saved(out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    analysis = json.loads((out_dir/'best_alpha_by_budget.json').read_text())
    points = [r for r in analysis['best_points'] if r is not None]
    x = [100*r['target_bloat'] for r in points]
    for filename, values, ylabel in (
        ('best_alpha_vs_budget.png', [r['rng_alpha'] for r in points], 'Best measured RNG alpha'),
        ('recall_retention_by_budget.png', [100*r['recall_gain_retention'] for r in points], 'Recall-gain retention (%)')):
        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(x, values, 'o-'); ax.set(xlabel='Dynamic overlap budget (%)', ylabel=ylabel, xticks=x)
        if filename.startswith('recall'): ax.axhline(95, color='gray', linestyle='--')
        ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out_dir/filename, dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4)); positions=np.arange(len(points))
    ax.bar(positions-.18, [100*r['target_bloat'] for r in points], .36, label='Before RNG')
    ax.bar(positions+.18, [100*r['actual_bloat'] for r in points], .36, label='After best measured RNG')
    ax.set(xticks=positions, xticklabels=[f'{v:g}%' for v in x], xlabel='Dynamic overlap budget', ylabel='Actual bloat (%)')
    ax.legend(); fig.tight_layout(); fig.savefig(out_dir/'bloat_before_after_rng.png', dpi=180); plt.close(fig)
