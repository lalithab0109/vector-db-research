"""Membership-only R2 diagnostic. Never calls search or recall evaluation."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/vecstudy-matplotlib')
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import csv
import hashlib
import json
import pickle
import time
from dataclasses import asdict
import numpy as np
from vecstudy.core.config import IndexConfig
from vecstudy.core.metrics import Counters
from vecstudy.core.boundary import compute_boundary_statistics
from vecstudy.harness.datasets import read_fvecs, l2_normalize, dataset_checksum
from vecstudy.harness.overlap_experiment import snapshot_digest
from vecstudy.index.segment import Segment
from vecstudy.index.overlap import rng_keep_candidates

DELTAS = {0.1: .007516240289318743, .2: .015807504067695266,
          .3: .025079955016746203, .45: .04163085684582124}
CACHE = ROOT / 'data/r2_diagnostic'
OUT = ROOT / 'results/r2_physical_balance'


def distribution(values):
    a = np.asarray(values)
    mean = float(a.mean()) if len(a) else 0.
    excess = a[a > 1200] - 1200
    return dict(count=len(a), total=int(a.sum()), mean=mean,
                median=float(np.median(a)) if len(a) else None,
                p95=float(np.percentile(a, 95)) if len(a) else None,
                p99=float(np.percentile(a, 99)) if len(a) else None,
                max=int(a.max()) if len(a) else None,
                p99_over_mean=float(np.percentile(a, 99)/mean) if mean else None,
                max_over_mean=float(a.max()/mean) if mean else None,
                gt800=int((a > 800).sum()), ge1000=int((a >= 1000).sum()),
                ge1200=int((a >= 1200).sum()), gt1200=int((a > 1200).sum()),
                gt1200_percent=100*len(excess)/len(a) if len(a) else 0.,
                mean_excess=float(excess.mean()) if len(excess) else 0.,
                max_excess=int(excess.max()) if len(excess) else 0)


def destination_buckets(values):
    a = np.asarray(values)
    masks = [a < 800, (a >= 800) & (a < 1000), (a >= 1000) & (a < 1200), a >= 1200]
    return [dict(bucket=label, count=int(mask.sum()), percent=100*int(mask.sum())/len(a) if len(a) else 0.)
            for label, mask in zip(('<800', '800-999', '1000-1199', '>=1200'), masks)]


def membership_stats(primary, owners, c1, c2, selected):
    destinations = np.where(owners == c1, c2, c1)
    assert np.all(destinations != owners)
    mentor = selected & (owners != c1) & (destinations == c1)
    complement = selected & (owners == c1) & (destinations == c2)
    np.testing.assert_array_equal(mentor, selected & (owners != c1))
    assert int(mentor.sum() + complement.sum()) == int(selected.sum())
    incoming = np.bincount(destinations[selected], minlength=len(primary))
    physical = primary + incoming
    assert int(physical.sum()) == len(owners) + int(selected.sum())
    corr = float(np.corrcoef(primary, incoming)[0, 1]) if np.std(primary) and np.std(incoming) else None
    top = np.lexsort((np.arange(len(primary)), -incoming))[:10]
    n = int(selected.sum())
    result = dict(replicas=n, physical=distribution(physical), incoming=distribution(incoming),
        pearson=corr, mentor_count=int(mentor.sum()), mentor_percent_replicas=100*int(mentor.sum())/n if n else 0.,
        mentor_percent_dataset=100*float(mentor.mean()), complement_count=int(complement.sum()),
        complement_percent_replicas=100*int(complement.sum())/n if n else 0.,
        destination_buckets=destination_buckets(primary[destinations[selected]]),
        mentor_destination_buckets=destination_buckets(primary[destinations[mentor]]),
        destination_cumulative={str(t): int((primary[destinations[selected]] >= t).sum()) for t in (800,1000,1200)},
        mentor_destination_cumulative={str(t): int((primary[destinations[mentor]] >= t).sum()) for t in (800,1000,1200)},
        mentor_destination_sizes=distribution(primary[destinations[mentor]]),
        complement_destination_sizes=distribution(primary[destinations[complement]]),
        top10=[dict(posting=int(j), primary=int(primary[j]), incoming=int(incoming[j]), physical=int(physical[j])) for j in top])
    result['physical']['p99_target_pass'] = result['physical']['p99_over_mean'] < 2
    result['physical']['max_target_pass'] = result['physical']['max_over_mean'] < 4
    return result, incoming, physical


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    reference = json.loads((ROOT/'results/r2_part8/sweep_results.json').read_text())
    cfg = IndexConfig(dim=128, build_mode='r5', depth=1, branching_factor=16, seed=0,
                      target_capacity=800, lambda_balance=.25, alpha=1.5,
                      tiny_cluster_fraction=.1, max_iter=15, delta=None, rng_rule=False)
    assert asdict(cfg) == reference['provenance']['configuration']
    snapshot_path = CACHE/'segment.pkl'
    metadata_path = CACHE/'metadata.json'
    if snapshot_path.exists():
        with snapshot_path.open('rb') as f:
            segment = pickle.load(f)
        metadata = json.loads(metadata_path.read_text())
        assert metadata['configuration'] == asdict(cfg)
        assert dataset_checksum(segment.vectors) == metadata['dataset_sha256'] == reference['provenance']['dataset_checksum']
        metadata['this_run_build_count'] = 0
        metadata['status'] = 'reused'
    else:
        # Also accept the legacy cache only with exact config/data/snapshot identity.
        legacy = ROOT/'data/sift_r5_segment.pkl'
        segment = None
        if legacy.exists():
            with legacy.open('rb') as f:
                old = pickle.load(f)
            if asdict(old['config']) == asdict(cfg) and snapshot_digest(old['segment']) == reference['results'][0]['snapshot_sha256']:
                segment = old['segment']
        elapsed = 0.
        if segment is None:
            base = l2_normalize(read_fvecs(ROOT/'data/sift/sift_base.fvecs'))
            assert base.shape == (1_000_000, 128)
            assert dataset_checksum(base) == reference['provenance']['dataset_checksum']
            print('Starting the single authorized R5 build', flush=True)
            start = time.perf_counter()
            segment = Segment.build(np.arange(len(base), dtype=np.int64), base, cfg, Counters())
            elapsed = time.perf_counter()-start
        metadata = dict(configuration=asdict(cfg), build_seconds=elapsed,
            this_run_build_count=int(elapsed > 0), status='rebuilt' if elapsed else 'reused',
            dataset_sha256=dataset_checksum(segment.vectors), snapshot_sha256=snapshot_digest(segment),
            numpy_version=np.__version__, thread_count=1,
            configuration_sha256=hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest())
        with snapshot_path.open('wb') as f:
            pickle.dump(segment, f, protocol=pickle.HIGHEST_PROTOCOL)
        metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    assert snapshot_digest(segment) == metadata['snapshot_sha256'] == reference['results'][0]['snapshot_sha256'], 'Snapshot mismatch: STOP'
    assert len(segment.postings) == 1250
    primary = np.array([len(p.owned) for p in segment.postings])
    owners = np.full(len(segment.vectors), -1, dtype=np.int64)
    occurrences = np.zeros(len(owners), dtype=np.int64)
    for p in segment.postings:
        np.testing.assert_array_equal(p.members, p.owned)
        np.add.at(occurrences, p.owned, 1)
        owners[p.owned] = p.id
    assert primary.sum() == 1_000_000 and np.all(occurrences == 1)
    for name, a in [('owners', owners), ('centroids', segment.levels[-1].centroids), ('primary_sizes', primary)]:
        np.save(CACHE/(name+'.npy'), a)
    boundary_path = CACHE/'boundary.npz'
    if boundary_path.exists():
        with np.load(boundary_path) as z:
            d1,d2,r,c1,c2 = [z[k] for k in ('d1','d2','r','c1','c2')]
    else:
        print('Computing boundary arrays once', flush=True)
        d1,d2,r,c1,c2 = compute_boundary_statistics(segment.vectors, segment.levels[-1].centroids, geometry='l2')
        np.savez(boundary_path, d1=d1,d2=d2,r=r,c1=c1,c2=c2)
    counts = {b: int((r <= delta).sum()) for b,delta in DELTAS.items()}
    metadata.update(candidate_counts=counts, cache_files=[str(p.relative_to(ROOT)) for p in CACHE.iterdir() if p.is_file()],
                    ann_query_runs=0, recall_runs=0)
    metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    assert all(counts[b] == round(b*len(owners)) for b in DELTAS), f'Candidate count mismatch: STOP {counts}'
    report = dict(provenance=metadata, primary=distribution(primary),
        owner_is_c1_count=int((owners == c1).sum()), owner_not_c1_count=int((owners != c1).sum()),
        owner_is_c1_percent=100*float((owners == c1).mean()), owner_not_c1_percent=100*float((owners != c1).mean()), treatments=[])
    series = []
    for budget,delta in DELTAS.items():
        selected = r <= delta
        for mode in ('dynamic','rng'):
            accepted = selected if mode == 'dynamic' else rng_keep_candidates(segment.vectors,
                segment.levels[-1].centroids, owners, np.where(owners == c1,c2,c1), selected, rng_alpha=1.59375)
            result, incoming, physical = membership_stats(primary, owners, c1, c2, accepted)
            result.update(budget=budget, mode=mode, delta=delta, rng_alpha=1.59375 if mode=='rng' else None)
            report['treatments'].append(result)
            series.append((budget,mode,incoming,physical))
            print(json.dumps(dict(budget=budget,mode=mode,replicas=result['replicas'],physical=result['physical'])), flush=True)
    assert snapshot_digest(segment) == metadata['snapshot_sha256']
    (OUT/'physical_balance_diagnostic.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    write_csv(OUT/'physical_balance_summary.csv', [dict(budget=t['budget'],mode=t['mode'],replicas=t['replicas'],**t['physical']) for t in report['treatments']])
    write_csv(OUT/'mentor_scenario.csv', [{k:t[k] for k in ('budget','mode','replicas','mentor_count','mentor_percent_replicas','mentor_percent_dataset','complement_count','complement_percent_replicas')} for t in report['treatments']])
    write_csv(OUT/'destination_size_buckets.csv', [dict(budget=t['budget'],mode=t['mode'],population=pop,**b) for t in report['treatments'] for pop,key in [('all','destination_buckets'),('mentor','mentor_destination_buckets')] for b in t[key]])
    write_csv(OUT/'replica_concentration.csv', [dict(budget=t['budget'],mode=t['mode'],pearson=t['pearson'],**t['incoming']) for t in report['treatments']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(11,8))
    for ax,budget in zip(axes.flat, DELTAS):
        for label,values in [('R5 primary',primary)]+[(mode,physical) for b,mode,inc,physical in series if b==budget]:
            ax.plot(np.sort(values),np.arange(1,len(values)+1)/len(values),label=label)
        ax.axvline(1200,color='gray',linestyle=':'); ax.set(title=f'Budget {budget:.0%}',xlabel='Posting memberships',ylabel='Empirical CDF'); ax.legend()
    fig.tight_layout(); fig.savefig(OUT/'posting_size_distribution.png'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,key,target in zip(axes,('p99_over_mean','max_over_mean'),(2,4)):
        ax.axhline(report['primary'][key],label='R5 primary',color='black')
        ax.axhline(target,color='gray',linestyle=':',label='R5 reference target')
        for mode in ('dynamic','rng'):
            ts=[t for t in report['treatments'] if t['mode']==mode]
            ax.plot([t['budget']*100 for t in ts],[t['physical'][key] for t in ts],marker='o',label=mode)
        ax.set(xlabel='Budget (%)',ylabel=key); ax.legend()
    fig.tight_layout(); fig.savefig(OUT/'physical_p99_max_vs_budget.png'); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,budget in zip(axes.flat, DELTAS):
        ax.scatter(primary,np.zeros(len(primary)),s=4,label='R5 primary (zero replicas)',color='black')
        for b,mode,inc,physical in series:
            if b==budget: ax.scatter(primary,inc,s=5,alpha=.5,label=mode)
        ax.set(title=f'Budget {budget:.0%}',xlabel='Primary posting size',ylabel='Incoming replicas'); ax.legend()
    fig.tight_layout(); fig.savefig(OUT/'replicas_vs_primary_size.png'); plt.close(fig)
    print('Diagnostic saved; no queries executed', flush=True)

if __name__ == '__main__':
    main()
