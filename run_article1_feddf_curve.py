"""Plan (default) or explicitly execute the 36 missing FedDF-prob proxy runs.

Use an audited snapshot for EXPERT references and read-only original teacher caches.
Writes only to a separate output directory; does not invoke the phase pipeline.
"""
from __future__ import annotations

import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

os.environ.setdefault('MPLCONFIGDIR', '/tmp/article1-matplotlib')

import argparse
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import shlex
import subprocess
import sys

import pandas as pd
from article1.distillation import kd_config, metadata_identity
from article1.hashes import file_sha256
from article1.progress import KD, read_csv, validate_rows

SIZES = (100, 500, 1000, 5000)
REGIMES = ('iid', 'alpha0p1', 'single')
SEEDS = (42, 43, 44)
ROOT = Path(__file__).resolve().parent


def references(frame):
    """Require unique, audited EXPERT references and full-size FedDF partners."""
    rows = {}
    for regime, seed, size in product(REGIMES, SEEDS, (*SIZES, 10000)):
        methods = ('expert_prob', 'feddf_prob') if size == 10000 else ('expert_prob',)
        for method in methods:
            selected = frame[(frame.dataset == 'cifar') & (frame.regime == regime)
                             & (frame.seed == seed) & (frame.proxy_size == size)
                             & (frame.method == method)]
            key = (regime, seed, size, method)
            if len(selected) != 1 or not selected.valid.all():
                raise ValueError(f'Missing, duplicate or invalid reference: {key}')
            rows[key] = selected.iloc[0]
    for regime, seed in product(REGIMES, SEEDS):
        check_pair(rows[regime, seed, 10000, 'feddf_prob'], rows[regime, seed, 10000, 'expert_prob'])
    return rows


def check_pair(actual, expert):
    for field in KD:
        if pd.isna(actual.get(field)) or pd.isna(expert.get(field)) or actual[field] != expert[field]:
            raise ValueError(f'Incompatible pair: {field}')


def jobs(refs, source_root, data_dir, results, device):
    planned = []
    for seed, size, regime in product(SEEDS, SIZES, REGIMES):
        expert = refs[regime, seed, size, 'expert_prob']
        config = kd_config()
        config.pop('epochs')
        config.update(updates=1200, proxy_size=size, proxy_labels_sha256=expert.proxy_labels_sha256)
        if json.loads(expert.training_recipe_json) != config or expert.temperature != 8:
            raise ValueError('EXPERT recipe differs from the planned FedDF recipe')
        run_id = metadata_identity(method='feddf_prob', temperature=8.0, config=config,
                                   source_hash=expert.cache_sha256, proxy_hash=expert.proxy_sha256,
                                   mask_hash=expert.M_sha256)
        command = [sys.executable, '-m', 'article1.runner', 'distill', '--dataset', 'cifar',
                   '--seed', str(seed), '--method', 'feddf_prob', '--temperature', '8',
                   '--proxy-size', str(size), '--updates', '1200', '--batch-size', '256',
                   '--cache', str(source_root/'sources'/f'cifar-seed{seed}-{regime}'/'teacher_cache.npz'),
                   '--data-dir', str(data_dir), '--results', str(results), '--device', device,
                   '--skip-existing']
        planned.append(dict(regime=regime, seed=seed, proxy_size=size, run_id=run_id,
                            expert_run_id=expert.run_id, command=command))
    if len({job['run_id'] for job in planned}) != 36:
        raise ValueError('Conflicting planned run identities')
    return planned


def pending_jobs(planned, existing, refs):
    """Never silently skip a conflicting identity or accept incompatible CRN."""
    expected = {job['run_id']: job for job in planned}
    if existing.empty:
        return planned
    if existing.run_id.duplicated().any() or not existing.valid.all():
        raise ValueError('Duplicate or invalid FedDF result')
    for _, row in existing.iterrows():
        job = expected.get(row.run_id)
        if job is None or row.method != 'feddf_prob' or row.dataset != 'cifar' or (
            row.regime, row.seed, row.proxy_size) != (job['regime'], job['seed'], job['proxy_size']):
            raise ValueError('Unexpected FedDF result identity')
        check_pair(row, refs[job['regime'], job['seed'], job['proxy_size'], 'expert_prob'])
    completed = set(existing.run_id)
    return [job for job in planned if job['run_id'] not in completed]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    snapshot, source, data, output = [p.resolve() for p in (args.snapshot, args.source_root, args.data_dir, args.output_root)]
    if ROOT == source.parent.parent or any(output.is_relative_to(p) for p in (source.parent.parent, snapshot, data)):
        raise ValueError('Use an independent checkout and output directory outside training, snapshot and data')
    manifest = json.loads((snapshot/'manifest.json').read_text())
    if manifest.get('closure', {}).get('proxy_curve') != 'cerrado':
        raise ValueError('Requires an audited complete EXPERT/CE curve snapshot')
    frames = []
    for name in ('results_baseline.csv', 'results_proxy_size_expert_prob.csv'):
        path = snapshot/'snapshots'/name
        records = [r for r in manifest['inputs'] if Path(r['original']).name == name]
        if len(records) != 1 or file_sha256(path) != records[0]['sha256']:
            raise ValueError(f'Snapshot hash mismatch: {name}')
        frames.append(read_csv(path))
    coverage = read_csv(snapshot/'tables/coverage.csv')
    subsets = read_csv(snapshot/'tables/proxy_subsets.csv')
    refs = references(validate_rows(pd.concat(frames, ignore_index=True), coverage, subsets, False))
    for regime, seed in product(REGIMES, SEEDS):
        cache = source/'sources'/f'cifar-seed{seed}-{regime}'/'teacher_cache.npz'
        if file_sha256(cache) != refs[regime, seed, 10000, 'expert_prob'].cache_sha256:
            raise ValueError(f'Original cache changed: {cache}')
    results = output/'results_proxy_size_feddf_prob.csv'
    planned = jobs(refs, source, data, results, args.device)
    def pending():
        existing = validate_rows(read_csv(results), coverage, subsets, False) if results.exists() else pd.DataFrame()
        return pending_jobs(planned, existing, refs)
    remaining = pending()
    print(f'EXPERT verified: 36 reduced + 9 full-size; FedDF reused from baseline: 9; pending: {len(remaining)}/36')
    for job in remaining:
        print(shlex.join(job['command']))
    if not args.execute:
        print('PLAN ONLY: no training and no output files written. Add --execute to launch sequentially.')
        return
    output.mkdir(parents=True, exist_ok=True)
    receipt = dict(date=datetime.now(timezone.utc).isoformat(), analysis_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), snapshot=str(snapshot),
        snapshot_manifest_sha256=file_sha256(snapshot/'manifest.json'), expected=36,
        reused_full_size=9, jobs=planned)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    (output/f'launch_{stamp}.json').write_text(json.dumps(receipt, indent=2)+'\n')
    for job in remaining:
        subprocess.run(job['command'], cwd=ROOT, check=True)
        # Fail immediately if a trained result cannot be paired with its EXPERT reference.
        if any(item['run_id'] == job['run_id'] for item in pending()):
            raise ValueError('Runner did not persist the expected run identity')
    print('Verified 36/36 reduced FedDF runs; ready for 45 paired comparisons with full-size references.')


if __name__ == '__main__':
    main()
