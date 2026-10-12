"""Controlled relaxation and measured best-alpha selection; shared R2 evaluator."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from ..core.boundary import compute_boundary_statistics, choose_delta
from ..index.overlap import rng_keep_candidates
from .overlap_experiment import run_overlap_comparison, snapshot_digest, save_results

PREDEFINED = (1., 1.25, 1.5, 2., 3., 5.)
RETENTION_TARGET = .95


def best_alpha(rows):
    eligible = [r for r in rows if r['rng_alpha'] is not None and r['recall_gain_retention'] >= RETENTION_TARGET]
    return min(eligible, key=lambda r: (r['actual_bloat'], r['scans_per_query'], r['rng_alpha'])) if eligible else None


def pareto_frontier(rows):
    return [r for r in rows if not any(
        s['recall_at_10'] >= r['recall_at_10'] and s['actual_bloat'] <= r['actual_bloat']
        and (s['recall_at_10'] > r['recall_at_10'] or s['actual_bloat'] < r['actual_bloat']) for s in rows)]


def enrich(row, endpoint, baseline):
    gain = endpoint['recall_at_10'] - baseline['recall_at_10']
    if gain <= 0:
        raise ValueError('No positive original overlap gain; retention is undefined')
    removed = endpoint['replicated_count'] - row['replicated_count']
    bloat = endpoint['actual_bloat'] - row['actual_bloat']
    scans = endpoint['scans_per_query'] - row['scans_per_query']
    row.update(derived_delta=row['delta'], candidate_count=row['selected_count'],
               candidate_fraction=row['selected_fraction'], accepted_replica_count=row['replicated_count'],
               rejected_replica_count=removed, acceptance_rate=row['replicated_count']/row['selected_count'],
               rejection_rate=removed/row['selected_count'], total_vectors_scanned=row['total_scanned'],
               vectors_scanned_per_query=row['scans_per_query'], replicas_removed=removed,
               relative_replica_reduction=removed/endpoint['replicated_count'],
               bloat_reduction=bloat, relative_bloat_reduction=bloat/endpoint['actual_bloat'],
               scan_reduction_per_query=scans, relative_scan_reduction=scans/endpoint['scans_per_query'],
               recall_at_10_loss=endpoint['recall_at_10']-row['recall_at_10'],
               recall_gain_over_baseline=row['recall_at_10']-baseline['recall_at_10'],
               recall_gain_retention=(row['recall_at_10']-baseline['recall_at_10'])/gain)
    return row


def run_relaxed(snapshot, vids, queries, gt, query_cfg, provenance, out_dir, root):
    root, out_dir = Path(root), Path(out_dir)
    reference_path = root/'results/r2_rng/rng_ablation.json'
    reference = json.loads(reference_path.read_text())
    baseline = reference['baseline_reference']
    off = next(r for r in reference['results'] if r['target_bloat']==.2 and not r['rng'])
    original_rng = next(r for r in reference['results'] if r['target_bloat']==.2 and r['rng'])
    assert snapshot_digest(snapshot)==off['snapshot_sha256'], 'Snapshot provenance mismatch'
    for key in ('dataset_checksum','query_checksum','ground_truth_checksum','configuration','query_configuration'):
        assert provenance[key]==reference['provenance'][key], f'Reference provenance mismatch: {key}'
    # Verify earlier artifacts stay byte-identical during this separate experiment.
    old_paths = [p for folder in ('r2_part6','r2_part7','r2_part8','r2_rng')
                 for p in (root/'results'/folder).glob('*') if p.is_file()]
    old_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in old_paths}
    stats = compute_boundary_statistics(snapshot.vectors[vids], snapshot.levels[-1].centroids, geometry='l2')
    _,_,r,c1,c2=stats
    delta = choose_delta(r,.2)  # The only budget calibration in this experiment.
    assert delta==off['delta']
    candidates = r <= delta
    assert int(candidates.sum())==off['selected_count']==200_000
    owner_by_id=np.empty(len(snapshot.vectors),dtype=np.int64)
    for p in snapshot.postings: owner_by_id[p.owned]=p.id
    owners=owner_by_id[vids];dest=np.where(owners==c1,c2,c1)
    centers=snapshot.levels[-1].centroids
    measured=[]; masks={}
    def evaluate(alpha,stage):
        mask = candidates.copy() if alpha is None else rng_keep_candidates(
            snapshot.vectors[vids], centers, owners, dest, candidates, rng_alpha=alpha)
        assert not np.any(mask & ~candidates)
        for previous,previous_mask in masks.items():
            if alpha is not None and previous is not None:
                low,high=(previous_mask,mask) if previous<alpha else (mask,previous_mask)
                assert not np.any(low & ~high), 'Accepted sets are not nested'
        mode='dynamic' if alpha is None else 'dynamic_rng'
        name='Dynamic 20% no RNG' if alpha is None else f'Relaxed RNG {alpha:g}'
        row=run_overlap_comparison(snapshot,vids,queries,gt,query_cfg,
            treatments=[(name,mode,.2,alpha)], boundary_statistics=stats,calibrated_delta=delta,
            progress=lambda text: print(text,flush=True))[0]
        assert row['replicated_count']==int(mask.sum())
        row.update(stage=stage, r5_balance_alpha=provenance['configuration']['alpha'],
                   candidate_sha256=hashlib.sha256(candidates.tobytes()).hexdigest(),
                   candidate_destinations_sha256=hashlib.sha256(dest[candidates].tobytes()).hexdigest(),
                   accepted_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
        if alpha is None or alpha==1:
            ref=off if alpha is None else original_rng
            for key in ('delta','selected_count','replicated_count','actual_bloat','recall_at_10','total_scanned','scans_per_query'):
                assert row[key]==ref[key], f'Endpoint reproduction failed: {key}'
        enrich(row,off,baseline)
        measured.append(row); masks[alpha]=mask
        return row
    # Reproduce the no-RNG and original-alpha endpoints before interpretation.
    evaluate(None,'endpoint')
    for alpha in PREDEFINED: evaluate(alpha,'predefined')
    predefined=[r for r in measured if r['stage']=='predefined']
    initial_best=best_alpha(predefined)
    bracket=None; refinement=[]; extension=None
    for low,high in zip(predefined,predefined[1:]):
        if low['recall_gain_retention']<RETENTION_TARGET<=high['recall_gain_retention']:
            bracket=[low['rng_alpha'],high['rng_alpha']];break
    if bracket is None and predefined[-1]['recall_gain_retention']<RETENTION_TARGET:
        extension=evaluate(10.,'extension')
        if extension['recall_gain_retention']>=RETENTION_TARGET: bracket=[5.,10.]
    final_bracket=bracket.copy() if bracket else None
    if bracket:
        low,high=bracket
        for _ in range(4):
            midpoint=(low+high)/2
            row=evaluate(midpoint,'refinement');refinement.append(midpoint)
            if row['recall_gain_retention']>=RETENTION_TARGET: high=midpoint
            else: low=midpoint
        final_bracket=[low,high]
    finite=sorted([r for r in measured if r['rng_alpha'] is not None],key=lambda r:r['rng_alpha'])
    assert [r['accepted_replica_count'] for r in finite]==sorted(r['accepted_replica_count'] for r in finite)
    assert [r['actual_bloat'] for r in finite]==sorted(r['actual_bloat'] for r in finite)
    fallback=max(finite,key=lambda r:(r['recall_gain_retention'],-r['actual_bloat'],-r['rng_alpha']))
    analysis=dict(selection_criterion='retention >= .95; lowest bloat, then scans/query, then rng_alpha',
                  retention_threshold=RETENTION_TARGET, predefined_alphas=list(PREDEFINED),
                  best_predefined_alpha=None if initial_best is None else initial_best['rng_alpha'],
                  predefined_highest_retention_fallback=max(predefined,key=lambda r:(r['recall_gain_retention'],-r['actual_bloat'],-r['rng_alpha'])),
                  predefined_qualifiers=[r['rng_alpha'] for r in predefined if r['recall_gain_retention']>=RETENTION_TARGET],
                  refinement_bracket=bracket, final_bracket=final_bracket, refinement_points=refinement,
                  extension_alpha=None if extension is None else 10., best_measured_point=best_alpha(finite),
                  best_measured_rng_alpha=None if best_alpha(finite) is None else best_alpha(finite)['rng_alpha'],
                  fallback_highest_retention_point=fallback,
                  pareto_frontier=[r['rng_alpha'] for r in pareto_frontier(measured)],
                  original_overlap_gain=off['recall_at_10']-baseline['recall_at_10'],
                  baseline_reference=baseline, no_rng_reference=off,
                  recall_nondecreasing_observed=all(a['recall_at_10']<=b['recall_at_10'] for a,b in zip(finite,finite[1:])))
    for path,digest in old_hashes.items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    prov=dict(provenance);prov['configuration']=dict(prov['configuration'])
    prov['configuration']['r5_balance_alpha']=prov['configuration'].pop('alpha')
    prov.update(experiment='Parameterized RNG, dynamic 20% only',rng_alpha_values=list(PREDEFINED),
                budget_calibration_count=1,rng='parameterized; no RNG endpoint included',
                prior_artifacts_unchanged=True, baseline_reference=baseline,
                source_sha256={**prov['source_sha256'], 'vecstudy/harness/relaxed_rng.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    save_results(measured,prov,out_dir,stem='relaxed_rng_results')
    (out_dir/'best_alpha_analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    plot_saved(out_dir)
    print(json.dumps(analysis['best_measured_point']),flush=True)
    return measured,analysis


def plot_saved(out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out_dir=Path(out_dir)
    d=json.loads((out_dir/'relaxed_rng_results.json').read_text());a=json.loads((out_dir/'best_alpha_analysis.json').read_text())
    rows=d['results'];best=a['best_measured_point'];baseline=a['baseline_reference']
    for retention,filename in ((False,'recall_vs_bloat.png'),(True,'recall_retention_vs_bloat_saved.png')):
        fig,ax=plt.subplots(figsize=(9,6))
        if not retention: ax.scatter(0,baseline['recall_at_10'],color='black',marker='s',label='R5 baseline')
        for r in rows:
            x=r['relative_bloat_reduction']*100 if retention else r['actual_bloat']*100
            y=r['recall_gain_retention']*100 if retention else r['recall_at_10']
            ax.scatter(x,y,marker='^' if r['stage']=='refinement' else 'o',color='tab:blue')
            # Label offsets affect presentation only, never measurements/selection.
            offsets = {None:(75,-20), 3.0:(20,45), 5.0:(85,30), 2.0:(40,5),
                       1.75:(15,-20), 1.625:(0,35), 1.59375:(-20,-40),
                       1.5625:(-55,20), 1.5:(-25,-18)}
            offset = offsets.get(r['rng_alpha'],(5,5))
            if retention:
                offset = {None:(-70,15),3.0:(30,30),5.0:(80,10),2.0:(-10,-30),
                          1.75:(0,20),1.625:(20,30),1.59375:(-20,-35),
                          1.5625:(40,-10),1.5:(20,-20)}.get(r['rng_alpha'],(5,5))
            ax.annotate('no RNG' if r['rng_alpha'] is None else f"{r['rng_alpha']:g}",
                        (x,y),xytext=offset,textcoords='offset points',fontsize=8,
                        arrowprops=dict(arrowstyle='-',color='gray',linewidth=.5))
        if best:
            x=best['relative_bloat_reduction']*100 if retention else best['actual_bloat']*100
            y=best['recall_gain_retention']*100 if retention else best['recall_at_10']
            ax.scatter(x,y,marker='*',s=180,color='tab:red',label=f"Best measured rng_alpha={best['rng_alpha']:g}")
        if retention: ax.axhline(95,color='gray',linestyle='--',label='95% retention requirement')
        ax.set(xlabel='Relative bloat reduction vs no RNG (%)' if retention else 'Actual bloat (%)',
               ylabel='Overlap recall-gain retention (%)' if retention else 'Recall@10',
               title='Parameterized RNG: dynamic 20%, SIFT1M')
        ax.grid(alpha=.25);ax.legend();ax.margins(x=.15,y=.15);fig.tight_layout()
        fig.savefig(out_dir/filename,dpi=200);plt.close(fig)
