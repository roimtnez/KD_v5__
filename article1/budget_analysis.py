"""Shared-label CE comparison and explicitly provisional/complete proxy curves."""
from itertools import product
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from article1 import DATASETS, REGIMES, SEEDS
from article1.analysis import METRICS, summarize

SIZES = (100, 500, 1000, 5000, 10000)
FOCAL = ('iid', 'alpha0p1', 'single')
CE = 'supervised_proxy_ce'


def budget_inventory(frame):
    """60 unique CIFAR executions: 15 CE and 45 KD, including reused N=10000."""
    keys = ['dataset', 'regime', 'seed', 'method', 'proxy_size']
    expected = [('cifar', 'shared_proxy', seed, CE, size) for seed, size in product(SEEDS, SIZES)]
    expected += [('cifar', regime, seed, 'expert_prob', size) for regime, seed, size in product(FOCAL, SEEDS, SIZES)]
    records = []
    for values in expected:
        rows = frame
        for key, value in zip(keys, values):
            if not rows.empty:
                rows = rows[rows[key].eq(value)]
        status = 'pending' if rows.empty else 'present' if len(rows)==1 and rows.valid.all() else 'invalid'
        records.append(dict(zip(keys, values), status=status))
    return pd.DataFrame(records)


def require_complete_curve(inventory):
    if len(inventory)!=60 or not inventory.status.eq('present').all():
        raise ValueError('Definitive proxy curve requires all 60 unique CIFAR executions (including reused N=10000)')


def curve_plot(out, regime, pairs, column):
    """Seed trajectories break at missing N; means with different seeds are not joined."""
    from article1.progress import save_figure
    if pairs.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    for seed in SEEDS:
        rows = pairs[pairs.seed.eq(seed)].set_index('proxy_size')
        values = rows[column].reindex(SIZES)
        ax.plot(range(5), values, marker='o', alpha=.65, label=f'seed {seed}')
    labels = []
    for i, size in enumerate(SIZES):
        rows = pairs[pairs.proxy_size.eq(size)]
        seeds = ','.join(map(str, sorted(rows.seed.unique()))) or '—'
        labels.append(f'{size}\nn={len(rows)} [{seeds}]')
        if len(rows):
            ax.plot(i, rows[column].mean(), 'k_', ms=12)
        if len(rows)>1:
            ax.errorbar(i, rows[column].mean(), yerr=rows[column].std(ddof=1), color='black', capsize=4)
    if column.startswith('delta_'):
        ax.axhline(0, lw=.7, color='grey')
    ax.set(xticks=range(5), xticklabels=labels, ylabel=column,
           xlabel='N público; 1200 actualizaciones (no iguala ejemplos consumidos)',
           title=f'CIFAR · {regime} · solo pares completos; barras: SD muestral')
    ax.legend(fontsize=8)
    save_figure(out, 'curve_'+regime+'_'+column, fig)


def paired_summary(pairs, groups):
    tables = []
    for column in [metric+suffix for metric in METRICS[:2] for suffix in ('_left','_right')]+['delta_'+metric for metric in METRICS[:2]]:
        summary = summarize(pairs, column, groups=groups)
        summary['metric'] = column
        seed_map = pairs.groupby(groups).seed.agg(lambda values: ','.join(map(str, sorted(values))))
        summary = summary.merge(seed_map.rename('seeds').reset_index(), on=groups, validate='one_to_one')
        tables.append(summary)
    return pd.concat(tables, ignore_index=True)


def analyze_budget(out: Path, mode='provisional'):
    from article1.progress import (read_csv, deduplicate_files, validate_rows, save_table,
                                   compatible_pairs, SHARED, effect_plot)
    if mode not in ('provisional', 'definitive'):
        raise ValueError('mode must be provisional or definitive')
    paths = [out/'snapshots'/name for name in ('results_supervised_proxy.csv','results_baseline.csv',
                                               'results_expertise.csv','results_proxy_size_expert_prob.csv')]
    frames = [read_csv(path).assign(input_file=path.name) for path in paths if path.exists()]
    curve_input = next((frame for frame in frames if not frame.empty and frame.input_file.iloc[0]=='results_proxy_size_expert_prob.csv'), pd.DataFrame())
    unexpected = pd.DataFrame()
    if not curve_input.empty:
        expected = curve_input.dataset.eq('cifar') & curve_input.proxy_size.isin(SIZES) & ((curve_input.method.eq(CE) & curve_input.regime.eq('shared_proxy')) | (curve_input.method.eq('expert_prob') & curve_input.regime.isin(FOCAL)))
        unexpected = curve_input[~expected]
    save_table(out, 'unexpected_curve_rows', unexpected)
    combined = deduplicate_files(frames)
    if not combined.empty:
        combined = combined[combined.method.isin([CE,'expert_prob'])].copy()
    combined = validate_rows(combined, read_csv(out/'tables/coverage.csv'), read_csv(out/'tables/proxy_subsets.csv'), False)
    save_table(out, 'proxy_budget_validated', combined)
    inventory = budget_inventory(combined)
    save_table(out, 'proxy_budget_completeness', inventory)
    print(inventory.pivot(index=['proxy_size','seed'], columns=['regime','method'], values='status').to_string())
    # Required before any curve figure; header-only optional files cannot satisfy it.
    if mode=='definitive':
        require_complete_curve(inventory)
        if not unexpected.empty:
            raise ValueError('Unexpected curve identities; inspect unexpected_curve_rows.csv')
    full = combined[combined.proxy_size.eq(10000)] if not combined.empty else combined
    ce = full[full.method.eq(CE)] if not full.empty else full
    ce_inventory = []
    for dataset, seed in product(DATASETS, SEEDS):
        rows = ce[ce.dataset.eq(dataset) & ce.seed.eq(seed)] if not ce.empty else ce
        ce_inventory.append(dict(dataset=dataset, seed=seed, status='present' if len(rows)==1 and rows.valid.all() else 'pending_or_invalid'))
    save_table(out, 'supervised_completeness', pd.DataFrame(ce_inventory))
    save_table(out, 'supervised_observations', ce)
    ce_closed = all(row['status']=='present' for row in ce_inventory)
    full_pairs, full_excluded = [], []
    curve_pairs, curve_excluded = [], []
    for regime in REGIMES:
        if combined.empty:
            continue
        subset = combined[combined.method.eq(CE) | combined.regime.eq(regime)]
        pairs, excluded = compatible_pairs(subset, 'expert_prob', CE, fields=SHARED,
                                           keys=['dataset','seed','proxy_size'], metrics=METRICS[:2])
        if not pairs.empty:
            pairs['regime'] = regime
            full_pairs.append(pairs[pairs.proxy_size.eq(10000)])
        if not excluded.empty:
            excluded['regime'] = regime
            full_excluded.append(excluded[excluded.proxy_size.eq(10000)])
        if regime not in FOCAL:
            continue
        selected = pairs[pairs.dataset.eq('cifar')].copy() if not pairs.empty else pd.DataFrame()
        seen = set(map(tuple, selected[['seed','proxy_size']].to_numpy())) if not selected.empty else set()
        for seed, size in product(SEEDS, SIZES):
            if (seed,size) not in seen:
                reason = 'missing_or_invalid_partner'
                if not excluded.empty:
                    matching = excluded[excluded.dataset.eq('cifar') & excluded.seed.eq(seed) & excluded.proxy_size.eq(size)]
                    if len(matching):
                        reason = matching.iloc[0].reason
                curve_excluded.append(dict(dataset='cifar',regime=regime,seed=seed,proxy_size=size,reason=reason))
        if not selected.empty:
            curve_pairs.append(selected)

    full_pairs = pd.concat(full_pairs, ignore_index=True) if full_pairs else pd.DataFrame()
    full_excluded = pd.concat(full_excluded, ignore_index=True) if full_excluded else pd.DataFrame()
    curve_pairs = pd.concat(curve_pairs, ignore_index=True) if curve_pairs else pd.DataFrame()
    save_table(out, 'private_knowledge_paired', full_pairs)
    save_table(out, 'private_knowledge_excluded', full_excluded)
    save_table(out, 'proxy_curve_paired', curve_pairs)
    save_table(out, 'proxy_curve_excluded', pd.DataFrame(curve_excluded))
    if not full_pairs.empty:
        save_table(out, 'private_knowledge_summary', paired_summary(full_pairs, ['dataset','regime']))
        for metric in METRICS[:2]:
            effect_plot(out, 'private_knowledge_'+metric, full_pairs, 'delta_'+metric)
    if not curve_pairs.empty:
        save_table(out, 'proxy_curve_summary', paired_summary(curve_pairs, ['dataset','regime','proxy_size']))
    if not combined.empty:
        save_table(out, 'examples_consumed', combined[['dataset','regime','seed','method','proxy_size','batch_size','updates','examples_seen','run_id']])
    manifest = json.loads((out/'manifest.json').read_text())
    manifest['supervised_analysis'] = dict(status='closed' if ce_closed and len(full_pairs)==54 else 'provisional',
                                          expected_ce=9, present_ce=sum(row['status']=='present' for row in ce_inventory),
                                          complete_pairs=len(full_pairs))
    manifest['proxy_curve'] = dict(mode=mode, complete=bool(inventory.status.eq('present').all() and unexpected.empty), unexpected_rows=len(unexpected),
                                  expected_executions=60, present=int(inventory.status.eq('present').sum()),
                                  expected_pairs=45, complete_pairs=len(curve_pairs),
                                  exclusions=curve_excluded, completeness=inventory.to_dict('records'))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if mode=='definitive' and len(curve_pairs)!=45:
        raise ValueError('Definitive proxy curve has incompatible pairs; inspect exclusions')
    if not curve_pairs.empty:
        for regime, selected in curve_pairs.groupby('regime'):
            for metric in METRICS[:2]:
                for column in [metric+'_left',metric+'_right','delta_'+metric]:
                    curve_plot(out, regime, selected, column)
    return manifest['supervised_analysis'], manifest['proxy_curve']
