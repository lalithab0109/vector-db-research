import numpy as np
import pytest
from vecstudy.index.overlap import rng_keep_candidates,apply_budget_overlap
from vecstudy.core.types import Posting,HierarchyLevel
from vecstudy.index.segment import Segment
from vecstudy.harness.overlap_experiment import validate_treatment,snapshot_digest


def test_pass_fail_equality_zero_and_unselected():
    C=np.array([[0.,0.],[2.,0.],[2.,0.]])
    X=np.array([[1.,0.],[5.,0.],[0.,0.],[2.,0.],[3.,0.],[2.,0.]])
    owners=np.array([0,0,0,0,1,1]);dest=np.array([1,1,1,1,2,2])
    selected=np.array([True,True,True,False,True,True])
    keep=rng_keep_candidates(X,C,owners,dest,selected)
    assert keep.tolist()==[True,False,True,False,False,True]
    np.testing.assert_array_equal(keep,rng_keep_candidates(X,C,owners,dest,selected))
    assert selected.tolist()==[True,True,True,False,True,True]


def test_owner_cases_overlay_and_no_accumulation():
    C=np.array([[0.,0.],[2.,0.],[0.,3.]])
    X=np.array([[1.,0.],[1.,0.],[1.,0.],[8.,0.],[1.,0.]])
    seg=Segment([HierarchyLevel(C.copy(),[])],
        [Posting(i,C[i].copy(),np.array(v),np.array(v)) for i,v in enumerate([[0,3],[1],[2,4]])],X)
    ids=np.arange(5);r=np.array([0.,0.,0.,0.,1.]);c1=np.zeros(5,dtype=int);c2=np.ones(5,dtype=int)
    original=snapshot_digest(seg)
    off,off_m=apply_budget_overlap(seg,ids,r,c1,c2,.8)
    on,m=apply_budget_overlap(seg,ids,r,c1,c2,.8,rng=True)
    validate_treatment(seg,on,ids,r,c1,c2,m['derived_delta'],rng=True)
    assert m['selected_count']==m['candidate_replica_count']==4
    assert m['rng_accepted_count']==m['replicated_count']==3
    assert m['rng_rejected_count']==1
    assert m['rng_acceptance_fraction']==.75
    assert m['rng_rejection_fraction']==.25
    assert m['realized_bloat']==pytest.approx(3/5)
    for a,b,c in zip(seg.postings,off.postings,on.postings):
        assert set(c.members)<=set(b.members)
        np.testing.assert_array_equal(a.owned,c.owned)
        np.testing.assert_array_equal(a.centroid,c.centroid)
    assert snapshot_digest(seg)==original
    assert 0 in on.postings[1].members # owner=c1 -> c2
    assert 1 in on.postings[0].members # owner=c2 -> c1
    assert 2 in on.postings[0].members # owner outside -> c1
    assert sum(4 in p.members for p in on.postings)==1 # unselected
    again,again_m=apply_budget_overlap(on,ids,r,c1,c2,.8,rng=True)
    assert again_m==m
    assert snapshot_digest(again)==snapshot_digest(on)
    assert off_m['replicated_count']==4


@pytest.mark.parametrize('case',['nan','inf','shape','ids','same_owner','mask'])
def test_bad_geometry(case):
    X=np.array([[1.,0.]]);C=np.array([[0.,0.],[2.,0.]])
    o=np.array([0]);d=np.array([1]);s=np.array([True])
    if case=='nan': X[0,0]=np.nan
    elif case=='inf': C[0,0]=np.inf
    elif case=='shape': X=X[:,0]
    elif case=='ids': d[0]=3
    elif case=='same_owner': d[0]=0
    else: s=np.array([1])
    with pytest.raises(ValueError): rng_keep_candidates(X,C,o,d,s)


def test_paired_harness_and_serialization(synthetic_200,tmp_path):
    import json,csv
    from vecstudy.core.config import IndexConfig,QueryConfig
    from vecstudy.core.metrics import Counters
    from vecstudy.harness.datasets import compute_ground_truth
    from vecstudy.harness.overlap_experiment import run_overlap_comparison,save_results
    from vecstudy.harness.rng_ablation import paired_effects
    X=synthetic_200
    seg=Segment.build(np.arange(200),X,IndexConfig(dim=32,build_mode='r5',target_capacity=20),Counters())
    queries=X[:2];gt=compute_ground_truth(queries,X,k=10,cache_dir=tmp_path)
    treatments=[(f'{b}-{mode}',mode,b) for b in (.1,.2,.45) for mode in ('dynamic','dynamic_rng')]
    rows=run_overlap_comparison(seg,np.arange(200),queries,gt,QueryConfig(n_postings=2),treatments=treatments)
    effects=paired_effects(rows)
    assert len(effects)==3
    assert all(r['rng'] for r in rows[1::2])
    save_results(rows,{},tmp_path,stem='rng')
    assert json.loads((tmp_path/'rng.json').read_text())['results']==rows
    with (tmp_path/'rng.csv').open() as f: c=list(csv.DictReader(f))
    assert len(c)==6
    assert int(c[1]['rng_accepted_count'])==rows[1]['replicated_count']
