"""Small membership arithmetic checks; no dataset build or query evaluation."""
import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('physical_balance', Path(__file__).parents[1]/'scripts/diagnose_physical_balance.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def test_bucket_boundaries():
    rows = diag.destination_buckets([799,800,999,1000,1199,1200])
    assert [r['count'] for r in rows] == [1,2,2,1]
    assert sum(r['percent'] for r in rows) == 100


def test_membership_and_mentor_masks_are_deterministic_and_preserve_owners():
    primary = np.array([2,1,1])
    owners = np.array([0,0,1,2])
    c1 = np.array([0,1,0,2])
    c2 = np.array([1,0,2,1])
    selected = np.array([True,True,True,False])
    original = owners.copy()
    report,incoming,physical = diag.membership_stats(primary,owners,c1,c2,selected)
    np.testing.assert_array_equal(incoming,[1,2,0])
    np.testing.assert_array_equal(physical,primary+incoming)
    np.testing.assert_array_equal(owners,original)
    assert report['mentor_count'] == 2 and report['complement_count'] == 1
    assert report['incoming']['mean'] == 1
    assert report['incoming']['median'] == 1
    assert report['incoming']['p99'] == np.percentile(incoming,99)
    assert report['top10'][0] == dict(posting=1,primary=1,incoming=2,physical=3)
    assert report == diag.membership_stats(primary,owners,c1,c2,selected)[0]


def test_rng_uses_same_primary_snapshot():
    from vecstudy.index.overlap import rng_keep_candidates
    primary = np.array([2,1])
    owners = np.array([0,0,1])
    c1 = np.array([0,1,1]); c2 = 1-c1
    vectors = np.array([[0.],[5.],[1.]])
    centers = np.array([[0.],[1.]])
    original = owners.copy()
    selected = np.ones(3,dtype=bool)
    destinations = np.where(owners==c1,c2,c1)
    keep = rng_keep_candidates(vectors,centers,owners,destinations,selected,rng_alpha=1.59375)
    _,raw_incoming,raw = diag.membership_stats(primary,owners,c1,c2,selected)
    _,filtered_incoming,filtered = diag.membership_stats(primary,owners,c1,c2,keep)
    np.testing.assert_array_equal(filtered-filtered_incoming,raw-raw_incoming)
    np.testing.assert_array_equal(owners,original)
    assert np.all(filtered <= raw)
