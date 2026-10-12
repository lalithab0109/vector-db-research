import copy
import json
from pathlib import Path
import numpy as np
import pytest
from vecstudy.harness.knee import analyze_knee, analyze_ranges, BUDGETS


def rows(x, y):
    return [dict(actual_bloat=a, recall_at_10=b, target_bloat=a, delta=a,
                 scans_per_query=100+a) for a,b in zip(x,y)]


def test_clear_elbow_and_marginals():
    a=analyze_knee(rows([0,.2,.4,.6,1],[0,.5,.8,.9,1]))
    assert a['knee']['actual_bloat']==.4
    assert a['knee']['marginal_before']==pytest.approx(1.5)
    assert a['knee']['marginal_after']==pytest.approx(.5)
    assert a['knee']['normalized_distance']==pytest.approx(.4/np.sqrt(2))


@pytest.mark.parametrize('x,y', [([0,.5,1],[0,.5,1]), ([0,.5,1],[1,1,1]),
                                ([1,1,1],[0,.5,1]), ([0,1],[0,1]), ([],[])])
def test_no_knee(x,y):
    assert analyze_knee(rows(x,y))['knee'] is None


def test_unsorted_duplicates_and_purity():
    r=rows([1,.4,0,.4,.2],[1,.7,0,.8,.5]); original=copy.deepcopy(r)
    a=analyze_knee(r)
    assert a['duplicates_collapsed']==1
    assert a['knee']['actual_bloat']==.4
    assert r==original
    assert analyze_knee(r)==a
    assert analyze_knee(r[::-1])==a


def test_equal_distance_lower_bloat_and_tolerance():
    a=analyze_knee(rows([0,.2,.4,1],[0,.6,.8,1]))
    assert a['knee']['actual_bloat']==.2
    assert analyze_knee(rows([0,.5,1],[0,.5+1e-12,1]))['knee'] is None


def test_scale_normalization():
    x=np.array([0,.2,.4,.6,1]); y=np.array([0,.5,.8,.9,1])
    a=analyze_knee(rows(x,y)); b=analyze_knee(rows(x*100+4,y*3+7))
    assert a['selected_index']==b['selected_index']
    assert a['knee']['normalized_distance']==pytest.approx(b['knee']['normalized_distance'])


@pytest.mark.parametrize('value', [np.nan,np.inf,-np.inf])
@pytest.mark.parametrize('field', ['actual_bloat','recall_at_10'])
def test_nonfinite_rejected(value,field):
    r=rows([0,.5,1],[0,.8,1]); r[1][field]=value
    with pytest.raises(ValueError,match='finite'): analyze_knee(r)


def test_real_result_schema_and_ranges():
    r=rows([0,.05,.1,.2,.3,.758716,.92998],
           [.75820,.76850,.77862,.79753,.81416,.87281,.88498])
    a=analyze_ranges(r)
    assert a['full_range']['knee'] is not None
    assert a['operational_range']['input_count']==5
    assert json.loads(json.dumps(a))['full_range']['selected_index']==a['full_range']['selected_index']
    assert BUDGETS==tuple(i/100 for i in range(0,51,5))+(.6,.7,.8,.9)


def test_nonmonotone_retained():
    a=analyze_knee(rows([0,.2,.4,.6,1],[0,.5,.8,.7,1]))
    assert not a['recall_nondecreasing']
    assert a['points'][3]['marginal_gain']<0
