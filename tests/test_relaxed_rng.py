import numpy as np
import pytest
from vecstudy.index.overlap import rng_keep_candidates,apply_budget_overlap
from vecstudy.harness.relaxed_rng import best_alpha,pareto_frontier,enrich
from vecstudy.harness.overlap_experiment import snapshot_digest,run_overlap_comparison,save_results
from vecstudy.core.config import IndexConfig,QueryConfig
from vecstudy.core.metrics import Counters
from vecstudy.core.boundary import compute_boundary_statistics,choose_delta
from vecstudy.index.segment import Segment


@pytest.mark.parametrize('alpha',[.5,np.nan,np.inf,-np.inf])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError,match='rng_alpha'):
        rng_keep_candidates([[0,0]],[[0,0],[1,0]],np.array([0]),np.array([1]),np.array([True]),rng_alpha=alpha)


def test_equality_and_nested_geometry():
    X=np.array([[0.,0.],[3.,0.],[4.,0.],[8.,0.]])
    C=np.array([[0.,0.],[1.,0.]])
    o=np.zeros(4,dtype=int);d=np.ones(4,dtype=int);sel=np.ones(4,dtype=bool)
    original=rng_keep_candidates(X,C,o,d,sel)
    np.testing.assert_array_equal(original,rng_keep_candidates(X,C,o,d,sel,rng_alpha=1))
    previous=original
    for a in [1.25,1.5,2,3,5]:
        keep=rng_keep_candidates(X,C,o,d,sel,rng_alpha=a)
        assert not np.any(previous & ~keep)
        previous=keep
    assert rng_keep_candidates(X,C,o,d,sel,rng_alpha=2)[1] # exact equality
    assert sel.all()


def test_overlay_backward_compatibility_and_nested_sets(synthetic_200,tmp_path):
    import json
    from vecstudy.harness.datasets import compute_ground_truth
    X=synthetic_200;ids=np.arange(len(X))
    seg=Segment.build(ids,X,IndexConfig(dim=32,build_mode='r5',target_capacity=20),Counters())
    fingerprint=snapshot_digest(seg)
    stats=compute_boundary_statistics(X,seg.levels[-1].centroids);_,_,r,c1,c2=stats
    delta=choose_delta(r,.2)
    old,metrics=apply_budget_overlap(seg,ids,r,c1,c2,.2,rng=True)
    explicit,explicit_metrics=apply_budget_overlap(seg,ids,r,c1,c2,.2,rng=True,rng_alpha=1)
    assert snapshot_digest(old)==snapshot_digest(explicit) and metrics==explicit_metrics
    off,_=apply_budget_overlap(seg,ids,r,c1,c2,.2)
    previous=[set(p.owned) for p in seg.postings];bloats=[]
    for a in [1,1.25,1.5,2,3,5]:
        out,m=apply_budget_overlap(seg,ids,r,c1,c2,.2,rng=True,rng_alpha=a)
        for i,p in enumerate(out.postings):
            assert previous[i]<=set(p.members)<=set(off.postings[i].members)
            np.testing.assert_array_equal(p.owned,seg.postings[i].owned)
            np.testing.assert_array_equal(p.centroid,seg.postings[i].centroid)
            previous[i]=set(p.members)
        bloats.append(m['realized_bloat'])
        repeat,again=apply_budget_overlap(out,ids,r,c1,c2,.2,rng=True,rng_alpha=a)
        assert again==m and snapshot_digest(repeat)==snapshot_digest(out)
    assert bloats==sorted(bloats) and snapshot_digest(seg)==fingerprint
    gt=compute_ground_truth(X[:2],X,k=10,cache_dir=tmp_path)
    rows=run_overlap_comparison(seg,ids,X[:2],gt,QueryConfig(n_postings=2),
        treatments=[('alpha1','dynamic_rng',.2,1),('no RNG','dynamic',.2,None)],
        boundary_statistics=stats,calibrated_delta=delta)
    save_results(rows,{},tmp_path)
    assert json.loads((tmp_path/'results.json').read_text())['results'][0]['rng_alpha']==1
    assert rows[0]['selected_count']==rows[1]['selected_count']


def test_best_selection_and_pareto():
    def point(a,b,s,ret,rec):
        return dict(rng_alpha=a,actual_bloat=b,scans_per_query=s,recall_gain_retention=ret,recall_at_10=rec)
    rows=[point(1,.01,10,.1,.76),point(2,.15,20,.95,.795),point(3,.15,19,.96,.796),point(4,.15,19,.96,.796),point(None,.2,25,1,.8)]
    assert best_alpha(rows)['rng_alpha']==3
    assert {r['rng_alpha'] for r in pareto_frontier(rows)}=={1,3,4,None}
    assert best_alpha(rows[:1]) is None
    assert best_alpha([rows[-1]]) is None


def test_negative_retention_not_clipped():
    row=dict(delta=.1,selected_count=10,selected_fraction=.1,replicated_count=1,actual_bloat=.01,scans_per_query=1,total_scanned=1,recall_at_10=.7)
    endpoint=dict(replicated_count=10,actual_bloat=.1,scans_per_query=2,recall_at_10=.8)
    assert enrich(row,endpoint,dict(recall_at_10=.75))['recall_gain_retention']<0
