from itertools import product
import pandas as pd
import pytest
from article1.budget_analysis import budget_inventory, require_complete_curve, CE, FOCAL, SIZES
from article1 import SEEDS


def grid():
    rows=[dict(dataset='cifar',regime='shared_proxy',seed=s,method=CE,proxy_size=n,valid=True) for s,n in product(SEEDS,SIZES)]
    rows += [dict(dataset='cifar',regime=r,seed=s,method='expert_prob',proxy_size=n,valid=True) for r,s,n in product(FOCAL,SEEDS,SIZES)]
    return pd.DataFrame(rows)


def test_curve_complete_grid_and_ce_reuse():
    frame=grid()
    assert len(frame)==60 and sum(frame.method.eq(CE))==15
    require_complete_curve(budget_inventory(frame))


@pytest.mark.parametrize('problem',['missing','duplicate','invalid'])
def test_curve_definitive_rejects_incomplete_or_invalid(problem):
    frame=grid()
    if problem=='missing': frame=frame.iloc[:-1]
    elif problem=='duplicate': frame=pd.concat([frame,frame.iloc[:1]])
    else: frame.loc[0,'valid']=False
    inv=budget_inventory(frame)
    assert len(inv)==60 and not inv.status.eq('present').all()
    with pytest.raises(ValueError,match='60 unique'): require_complete_curve(inv)


def test_empty_curve_remains_executable_inventory():
    inv=budget_inventory(pd.DataFrame())
    assert len(inv)==60 and inv.status.eq('pending').all()
