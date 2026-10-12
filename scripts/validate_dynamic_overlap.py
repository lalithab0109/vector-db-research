"""SIFT1M Part 6 calibration only; deliberately runs no query experiments."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from vecstudy.core.boundary import compute_boundary_statistics
from vecstudy.core.config import IndexConfig
from vecstudy.core.metrics import Counters
from vecstudy.harness.datasets import read_fvecs, l2_normalize, dataset_checksum
from vecstudy.index.segment import Segment
from vecstudy.index.overlap import apply_budget_overlap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/r2_part6/validation.json')
    args = parser.parse_args()
    base = l2_normalize(read_fvecs(ROOT / 'data/sift/sift_base.fvecs'))
    cfg = IndexConfig(dim=128, build_mode='r5', depth=1, branching_factor=16,
                      seed=0, target_capacity=800, lambda_balance=.25, alpha=1.5,
                      tiny_cluster_fraction=.1, max_iter=15)
    vids = np.arange(len(base), dtype=np.int64)
    print('Building frozen R5 configuration from local normalized SIFT1M', flush=True)
    start = time.perf_counter()
    seg = Segment.build(vids, base, cfg, Counters())
    build_seconds = time.perf_counter() - start
    centers = seg.levels[-1].centroids
    print(f'Built {len(centers)} leaves in {build_seconds:.6f}s; computing existing Part-2 L2 statistics', flush=True)
    start = time.perf_counter()
    _, _, r, c1, c2 = compute_boundary_statistics(base, centers, geometry='l2')
    boundary_seconds = time.perf_counter() - start
    owners_before = [p.owned.copy() for p in seg.postings]
    rows = []
    for budget in [.05, .10, .20, .30]:
        overlay, metrics = apply_budget_overlap(seg, vids, r, c1, c2, budget)
        for p, expected in zip(overlay.postings, owners_before):
            np.testing.assert_array_equal(p.owned, expected)
            np.testing.assert_array_equal(p.centroid, centers[p.id])
            assert len(p.members) == len(np.unique(p.members))
        for original, treatment in zip(seg.levels, overlay.levels):
            np.testing.assert_array_equal(original.centroids, treatment.centroids)
            for a, b in zip(original.children, treatment.children):
                np.testing.assert_array_equal(a, b)
        multiplicity = np.zeros(len(base), dtype=np.int8)
        for p in overlay.postings:
            np.add.at(multiplicity, p.members, 1)
        selected = r <= metrics['derived_delta']
        np.testing.assert_array_equal(multiplicity, 1 + selected.astype(np.int8))
        assert metrics['selected_count'] == metrics['replicated_count']
        assert abs(metrics['realized_bloat'] - metrics['replication_fraction']) < 1e-12
        for p, expected in zip(seg.postings, owners_before):
            np.testing.assert_array_equal(p.owned, expected)
            np.testing.assert_array_equal(p.members, expected)
        rows.append(metrics)
        print(json.dumps(metrics), flush=True)
        del overlay
    report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  dataset='sift1m', base_shape=list(base.shape), query_count=0,
                  dataset_checksum=dataset_checksum(base), configuration=asdict(cfg),
                  geometry='Euclidean L2', rng=False, provenance='fresh R5 build, no cache',
                  git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  implementation_status='uncommitted Part 6 working tree',
                  build_seconds=build_seconds, boundary_seconds=boundary_seconds,
                  leaf_count=len(centers),
                  centroid_sha256=hashlib.sha256(centers.tobytes()).hexdigest(),
                  ownership_sha256=hashlib.sha256(b''.join(p.owned.tobytes() for p in seg.postings)).hexdigest(),
                  numpy_version=np.__version__, thread_count=1,
                  finite_r_count=int(np.isfinite(r).sum()), positive_inf_count=int(np.isposinf(r).sum()),
                  results=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved {args.out}', flush=True)


if __name__ == '__main__':
    main()
