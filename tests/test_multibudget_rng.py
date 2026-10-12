import numpy as np
import pytest
from vecstudy.harness.multibudget_rng import (
    QueryBudget, EvaluationLimit, bounded_search, stage_a, common_measured, BUDGETS)
from vecstudy.harness.relaxed_rng import best_alpha
from vecstudy.core.config import IndexConfig
from vecstudy.core.metrics import Counters
from vecstudy.core.boundary import compute_boundary_statistics, choose_delta
from vecstudy.index.segment import Segment
from vecstudy.harness.overlap_experiment import snapshot_digest


def test_counter_and_reuse_guard():
    counter = QueryBudget()
    with pytest.raises(EvaluationLimit, match='reuse-only'):
        counter.reserve(.2)
    for b in (.1,.3,.45):
        for _ in range(6): counter.reserve(b)
        with pytest.raises(EvaluationLimit): counter.reserve(b)
    assert counter.expensive_query_evaluations==18 and counter.counts[.2]==0


@pytest.mark.parametrize('threshold, expected', [(1.6,[1.5,1.75,1.625,1.5625,1.59375]),
    (1.4,[1.5,1.75,1.25,1.375,1.4375,1.40625]),
    (1.1,[1.5,1.75,1.25,1.,1.125,1.0625]),
    (2.7,[1.5,1.75,2.,3.,2.5,2.75]),
    (4.,[1.5,1.75,2.,3.])])
def test_bounded_policy(threshold, expected):
    calls=[]
    def evaluate(alpha, stage):
        calls.append(alpha)
        return dict(rng_alpha=alpha, stage=stage, recall_gain_retention=.95 if alpha>=threshold else .94)
    rows, analysis=bounded_search(evaluate)
    assert calls==expected and len(rows)<=6 and analysis['refinement_count']<=3


def test_stage_a_no_search_reuses_snapshot_and_stats(synthetic_200, monkeypatch):
    import vecstudy.harness.multibudget_rng as harness
    seg=Segment.build(np.arange(200), synthetic_200,
        IndexConfig(dim=32,build_mode='r5',target_capacity=20), Counters())
    stats=compute_boundary_statistics(synthetic_200,seg.levels[-1].centroids)
    refs={b:dict(delta=choose_delta(stats[2],b),selected_count=round(200*b)) for b in (.1,.3,.45)}
    def forbidden(*args, **kwargs): raise AssertionError('Stage A must not search or recompute boundary')
    monkeypatch.setattr(harness,'run_overlap_comparison',forbidden)
    monkeypatch.setattr(harness,'compute_boundary_statistics',forbidden)
    before=snapshot_digest(seg)
    rows,candidates,_,_=stage_a(seg,np.arange(200),stats,refs)
    assert len(rows)==27 and snapshot_digest(seg)==before
    assert [int(candidates[b].sum()) for b in (.1,.3,.45)]==[20,60,90]
    assert [(r['target_bloat'],r['rng_alpha']) for r in rows]==sorted((r['target_bloat'],r['rng_alpha']) for r in rows)


def test_selection_and_common_use_measured_points():
    rows=[dict(target_bloat=b,rng_alpha=a,recall_gain_retention=ret,actual_bloat=a/10,scans_per_query=100)
          for b in BUDGETS for a,ret in ((1.5,.949999),(1.75,.95),(2.,.99))]
    assert best_alpha(rows)['rng_alpha']==1.75
    common=common_measured(rows)
    assert common['rng_alpha']==1.75 and len(common['points'])==4
    assert all(r['rng_alpha']!=1.6 for r in rows)


def test_orchestration_reuses_20_and_boundary_once(synthetic_200, tmp_path, monkeypatch):
    import json
    import vecstudy.harness.multibudget_rng as harness
    from vecstudy.core.config import QueryConfig
    from vecstudy.harness.overlap_experiment import run_overlap_comparison
    from vecstudy.harness.relaxed_rng import enrich
    vids=np.arange(200)
    seg=Segment.build(vids, synthetic_200, IndexConfig(dim=32,build_mode='r5',target_capacity=20), Counters())
    q=synthetic_200[:1]; gt=np.arange(10)[None,:]; qc=QueryConfig(k=10,n_postings=2)
    refs=run_overlap_comparison(seg,vids,q,gt,qc,treatments=[('baseline','none',None)]+[(str(b),'dynamic',b) for b in BUDGETS])
    for row in refs: row['recall_at_10']=.4 if row['threshold_mode']=='none' else .5
    prov=dict(dataset_checksum='base',query_checksum='query',ground_truth_checksum='gt',
              query_configuration=qc.to_dict(), configuration={'alpha':1.5},source_sha256={})
    p=tmp_path/'results/r2_part8'; p.mkdir(parents=True)
    (p/'sweep_results.json').write_text(json.dumps(dict(results=refs,provenance=prov)))
    old=tmp_path/'results/r2_relaxed_rng';old.mkdir()
    measured=run_overlap_comparison(seg,vids,q,gt,qc,treatments=[('old','dynamic_rng',.2,1.75)])[0]
    measured['recall_at_10']=.499; measured['stage']='predefined'; enrich(measured,refs[2],refs[0])
    endpoint=dict(refs[2],rng_alpha=None,stage='endpoint'); enrich(endpoint,refs[2],refs[0])
    (old/'relaxed_rng_results.json').write_text(json.dumps(dict(results=[endpoint,measured],provenance=dict(prov,baseline_reference=refs[0]))))
    (old/'best_alpha_analysis.json').write_text(json.dumps(dict(best_measured_point=measured)))
    boundary_calls=[]; shared_stats=[]; snapshot_ids=[]; budgets=[]
    original_boundary=harness.compute_boundary_statistics
    def boundary(*args,**kwargs):
        boundary_calls.append(1)
        return original_boundary(*args,**kwargs)
    def evaluation(snapshot,*args,**kwargs):
        snapshot_ids.append(id(snapshot)); shared_stats.append(id(kwargs['boundary_statistics']))
        treatment=kwargs['treatments'][0]; budgets.append(treatment[2])
        row=run_overlap_comparison(snapshot,*args,**kwargs)[0]
        row['recall_at_10']=.499 if treatment[3]>=1.6 else .48
        return [row]
    monkeypatch.setattr(harness,'compute_boundary_statistics',boundary)
    monkeypatch.setattr(harness,'run_overlap_comparison',evaluation)
    monkeypatch.setattr(harness,'plot_saved',lambda path: None)
    rows,analysis=harness.run_multibudget(seg,vids,q,gt,qc,prov,tmp_path/'output',tmp_path)
    assert len(boundary_calls)==1 and len(set(shared_stats))==1 and set(snapshot_ids)=={id(seg)}
    assert .2 not in budgets and len(budgets)==15
    assert analysis['evaluations_by_budget']['0.2']==0
    reused=[r for r in rows if r['target_bloat']==.2]
    assert len(reused)==1 and reused[0]['reused'] and not reused[0]['new_query_evaluation']
    assert [(r['target_bloat'],r['rng_alpha']) for r in rows]==sorted((r['target_bloat'],r['rng_alpha']) for r in rows)
