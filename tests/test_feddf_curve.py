from itertools import product
import json
from pathlib import Path
import pandas as pd
import pytest
from article1.distillation import kd_config
from article1.progress import KD
from run_article1_feddf_curve import references, jobs, pending_jobs, REGIMES, SEEDS, SIZES


def fixture():
    records = []
    for regime, seed, size in product(REGIMES, SEEDS, (*SIZES, 10000)):
        recipe = kd_config(); recipe.pop('epochs')
        recipe.update(updates=1200, proxy_size=size, proxy_labels_sha256='labels')
        for method in (('expert_prob', 'feddf_prob') if size == 10000 else ('expert_prob',)):
            row = dict.fromkeys(KD, 'same')
            row.update(dataset='cifar', regime=regime, seed=seed, proxy_size=size,
                       method=method, valid=True, temperature=8.0, proxy_labels_sha256='labels',
                       cache_sha256=f'cache-{regime}-{seed}',
                       training_recipe_json=json.dumps(recipe, sort_keys=True),
                       run_id=f'{regime}-{seed}-{size}-{method}')
            records.append(row)
    frame = pd.DataFrame(records)
    return frame


def plan(frame):
    refs = references(frame)
    return refs, jobs(refs, Path('/sources'), Path('/data'), Path('/out/results.csv'), 'cpu')


def test_exact_remaining_grid():
    refs, planned = plan(fixture())
    assert len(planned) == 36
    assert len({j['run_id'] for j in planned}) == 36
    assert {j['proxy_size'] for j in planned} == set(SIZES)
    for job in planned:
        cmd = job['command']
        assert cmd[cmd.index('--method')+1] == 'feddf_prob'
        assert cmd[cmd.index('--updates')+1] == '1200'
        assert cmd[cmd.index('--temperature')+1] == '8'
    assert pending_jobs(planned, pd.DataFrame(), refs) == planned


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'invalid'])
def test_reject_bad_expert_references(problem):
    frame = fixture()
    if problem == 'missing': frame = frame.iloc[1:]
    elif problem == 'duplicate': frame = pd.concat([frame, frame.iloc[:1]])
    else: frame.loc[0, 'valid'] = False
    with pytest.raises(ValueError, match='reference'): references(frame)


def test_resume_requires_matching_crn():
    refs, planned = plan(fixture())
    job = planned[0]
    row = refs[job['regime'], job['seed'], job['proxy_size'], 'expert_prob'].copy()
    row['method'] = 'feddf_prob'; row['run_id'] = job['run_id']
    completed = pd.DataFrame([row])
    assert len(pending_jobs(planned, completed, refs)) == 35
    completed.loc[completed.index[0], 'consumed_batches_sha256'] = 'different'
    with pytest.raises(ValueError, match='consumed_batches'): pending_jobs(planned, completed, refs)


def test_recipe_must_match_expert():
    frame = fixture(); frame.loc[0, 'training_recipe_json'] = '{}'
    with pytest.raises(ValueError, match='recipe'): plan(frame)
