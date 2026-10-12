"""Run only Part 7 SIFT1M recall/bloat comparison, with one fresh R5 build."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/vecstudy-matplotlib')

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from vecstudy.core.config import IndexConfig, QueryConfig
from vecstudy.core.metrics import Counters
from vecstudy.harness.datasets import read_fvecs, read_ivecs, l2_normalize, dataset_checksum
from vecstudy.harness.overlap_experiment import run_overlap_comparison, save_results, plot_results
from vecstudy.index.segment import Segment


def main(*, part8=False, rng_ablation=False, relaxed_rng=False, multibudget_rng=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / ('results/r2_relaxed_rng_multibudget' if multibudget_rng else 'results/r2_relaxed_rng' if relaxed_rng else 'results/r2_rng' if rng_ablation else ('results/r2_part8' if part8 else 'results/r2_part7')))
    args = parser.parse_args()
    start_utc = datetime.now(timezone.utc).isoformat()
    base = l2_normalize(read_fvecs(ROOT / 'data/sift/sift_base.fvecs'))
    queries = l2_normalize(read_fvecs(ROOT / 'data/sift/sift_query.fvecs'))
    gt = read_ivecs(ROOT / 'data/sift/sift_groundtruth.ivecs')
    assert base.shape == (1_000_000, 128)
    assert queries.shape == (10_000, 128) and gt.shape == (10_000, 100)
    cfg = IndexConfig(dim=128, build_mode='r5', depth=1, branching_factor=16,
                      seed=0, target_capacity=800, lambda_balance=.25, alpha=1.5,
                      tiny_cluster_fraction=.1, max_iter=15)
    query_cfg = QueryConfig(k=10, n_postings=8, beam_widths=(8,))
    vids = np.arange(len(base), dtype=np.int64)
    print('Building R5 ONCE with validated configuration', flush=True)
    start = time.perf_counter()
    seg = Segment.build(vids, base, cfg, Counters())
    elapsed = time.perf_counter() - start
    print(f'Built {len(seg.postings)} leaves in {elapsed:.6f}s', flush=True)
    provenance = dict(dataset='sift1m', base_shape=list(base.shape),
                      query_shape=list(queries.shape), ground_truth_shape=list(gt.shape),
                      dataset_checksum=dataset_checksum(base), query_checksum=dataset_checksum(queries),
                      ground_truth_checksum=dataset_checksum(gt), query_ids='all 0..9999 in file order',
                      configuration=asdict(cfg), query_configuration=query_cfg.to_dict(),
                      build_seconds=elapsed, build_count=1, boundary_computation_count=1,
                      boundary_geometry='Euclidean L2', build_geometry='capacity-aware squared L2',
                      search_geometry='inner product', ground_truth_geometry='supplied raw L2',
                      rng=False, numpy_version=np.__version__, thread_count=1,
                      timestamp_start_utc=start_utc,
                      git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      implementation_status='uncommitted R2 working tree',
                      source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [ROOT/'vecstudy/core/boundary.py', ROOT/'vecstudy/index/overlap.py',
                                  ROOT/'vecstudy/harness/overlap_experiment.py', Path(__file__).resolve()]})
    if multibudget_rng:
        from vecstudy.harness.multibudget_rng import run_multibudget
        run_multibudget(seg,vids,queries,gt,query_cfg,provenance,args.out,ROOT)
        return
    if relaxed_rng:
        from vecstudy.harness.relaxed_rng import run_relaxed
        run_relaxed(seg,vids,queries,gt,query_cfg,provenance,args.out,ROOT)
        return
    if rng_ablation:
        treatments = [(f'Dynamic {b*100:g}%'+(' + RNG' if mode=='dynamic_rng' else ''), mode, b)
                      for b in (.1,.2,.45) for mode in ('dynamic','dynamic_rng')]
    elif part8:
        from vecstudy.harness.knee import BUDGETS, analyze_ranges, plot_knee
        treatments = [('Dynamic 0%', 'none', None)] + [(f'Dynamic {b*100:g}%', 'dynamic', b) for b in BUDGETS if b]
    else:
        from vecstudy.harness.overlap_experiment import TREATMENTS
        treatments = TREATMENTS
    rows = run_overlap_comparison(seg, vids, queries, gt, query_cfg, treatments=treatments,
                                  progress=lambda message: print(message, flush=True))
    provenance['timestamp_end_utc'] = datetime.now(timezone.utc).isoformat()
    # Preserve numeric results even if optional plotting fails.
    if rng_ablation:
        import json
        from vecstudy.harness.rng_ablation import paired_effects, plot_rng
        previous = json.loads((ROOT/'results/r2_part8/sweep_results.json').read_text())
        assert previous['results'][0]['snapshot_sha256'] == rows[0]['snapshot_sha256']
        for key in ('dataset_checksum','query_checksum','ground_truth_checksum','configuration','query_configuration'):
            assert previous['provenance'][key] == provenance[key], f'Reference mismatch: {key}'
        for row in rows:
            if not row['rng']:
                ref = next(r for r in previous['results'] if r['target_bloat']==row['target_bloat'])
                for key in ('delta','selected_count','replicated_count','actual_bloat','recall_at_10','total_scanned','scans_per_query'):
                    assert row[key] == ref[key], f'No-RNG reproduction failed: {key}'
                row.update(candidate_replica_count=row['selected_count'], rng_accepted_count=None,
                           rng_rejected_count=None, rng_acceptance_fraction=None, rng_rejection_fraction=None)
        effects = paired_effects(rows)
        provenance['rng'] = 'paired disabled/enabled; base R5 RNG disabled'
        provenance['rng_geometry'] = 'Euclidean L2; reject candidate iff vector-candidate > owner-candidate'
        provenance['source_sha256']['vecstudy/harness/rng_ablation.py'] = hashlib.sha256((ROOT/'vecstudy/harness/rng_ablation.py').read_bytes()).hexdigest()
        save_results(rows, provenance, args.out, stem='rng_ablation')
        report_path = args.out/'rng_ablation.json'
        report = json.loads(report_path.read_text()); report['paired_effects'] = effects
        report['baseline_reference'] = previous['results'][0]
        report_path.write_text(json.dumps(report,indent=2)+'\n')
        plot_rng(rows, previous['results'][0], args.out/'rng_comparison.png')
        print(json.dumps(effects),flush=True)
    elif part8:
        import json
        provenance['source_sha256']['vecstudy/harness/knee.py'] = hashlib.sha256((ROOT/'vecstudy/harness/knee.py').read_bytes()).hexdigest()
        save_results(rows, provenance, args.out, stem='sweep_results')
        analysis = analyze_ranges(rows)
        (args.out/'knee_analysis.json').write_text(json.dumps(analysis, indent=2)+'\n')
        previous = json.loads((ROOT/'results/r2_part7/results.json').read_text())
        assert previous['results'][0]['snapshot_sha256'] == rows[0]['snapshot_sha256'], 'Part 7 reference snapshot differs'
        for key in ('dataset_checksum','query_checksum','ground_truth_checksum','configuration','query_configuration'):
            assert previous['provenance'][key] == provenance[key], f'Part 7 reference mismatch: {key}'
        fixed = [r for r in previous['results'] if r['threshold_mode'] == 'fixed']
        plot_knee(rows, analysis, fixed, args.out)
        print(json.dumps({key: a['knee'] for key,a in analysis.items()}), flush=True)
    else:
        save_results(rows, provenance, args.out)
        plot_results(rows, args.out / 'recall_vs_bloat.png')
    print(f'Saved numeric results and plot to {args.out}', flush=True)


if __name__ == '__main__':
    main()
