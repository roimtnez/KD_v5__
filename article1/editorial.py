"""Read-only experimental audit and editorial exports for article1_paper.ipynb.

The notebook supplies explicit sources; historical summaries never supply observations.
"""
from pathlib import Path
from itertools import product
import json
import hashlib
import platform
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from article1 import DATASETS, REGIMES, SEEDS
from article1.experiments import BASELINE_METHODS
from article1.analysis import KEY, METRICS, ROUTING, SUPPORT_MASS
from article1.distillation import build_target, kd_config, METHODS
from article1.hashes import file_sha256, array_sha256
from article1.presence import load_presence, compare_presence
from article1.progress import (sources, stable_copy, read_csv, validate_rows,
                               compatible_pairs, KD, SHARED, CE, now)
from article1.proxy import proxy_positions

FOCAL = ('iid', 'alpha0p1', 'single')
SIZES = (100, 500, 1000, 5000, 10000)
LABELS = dict(zip(REGIMES, ['IID', 'α=1', 'α=.5', 'α=.1', 'Multi', 'Single']))
NAMES = {m: m.replace('_', ' ') for m in (*METHODS, 'presence_prob')}
NAMES.update({'feddf_logit':'FedDF-logit', 'feddf_prob':'FedDF-prob', 'oracle_logit':'ORACLE-logit',
              'oracle_prob':'ORACLE-prob', 'expert_prob':'EXPERT-prob', 'expert_prob_sr':'EXPERT-prob-SR',
              'presence_prob':'Presence-prob', 'expert_logit':'EXPERT-logit'})
NAMES[CE] = 'CE (referencia reutilizada)'
METRIC_LABELS = {'student_test_accuracy':'Accuracy student', 'student_test_nll':'NLL student',
                 'target_accuracy':'Accuracy target', 'target_nll':'NLL target', 'target_entropy':'Entropía target',
                 'fallback_rate':'Tasa fallback', 'mean_selected_teachers':'Teachers seleccionados'}
COLORS = dict(zip([*BASELINE_METHODS, 'presence_prob', CE, 'expert_logit'],
                 ['#0072B2', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#D55E00', '#882255', '#444444', '#999933']))
CONTRASTS = {
    'selection': ('oracle_logit', 'feddf_logit'),
    'expertise': ('expert_prob', 'feddf_prob'),
    'oracle_expert': ('oracle_prob', 'expert_prob'),
    'competence_presence': ('expert_prob', 'presence_prob'),
    'pooling_feddf': ('feddf_prob', 'feddf_logit'),
    'pooling_oracle': ('oracle_prob', 'oracle_logit'),
    'pooling_expert': ('expert_prob', 'expert_logit'),
    'support': ('expert_prob_sr', 'expert_prob'),
}


def table(out, name, frame):
    frame.to_csv(out / 'tables' / f'{name}.csv', index=False)
    frame.to_latex(out / 'tables' / f'{name}.tex', index=False, float_format='%.4f', escape=True)
    return frame


def summary(frame, groups, metrics):
    records = []
    for key, rows in frame.groupby(groups, observed=True, dropna=False, sort=False):
        ident = dict(zip(groups, key if isinstance(key, tuple) else (key,)))
        if rows.seed.duplicated().any():
            raise ValueError(f'Duplicate experimental seed in summary: {ident}')
        for metric in metrics:
            values = rows[metric].dropna()
            scale = 100 if 'accuracy' in metric else 1
            records.append(dict(**ident, metric=metric, mean=values.mean()*scale,
                                sd=values.std(ddof=1)*scale, n=len(values),
                                seeds=','.join(map(str, sorted(rows.loc[values.index, 'seed'].unique())))))
    return pd.DataFrame(records)


def _recipe(row):
    if row.method == CE:
        return dict(loss='cross_entropy', optimizer='AdamW', lr=.001, weight_decay=.0001,
                    batch_size=min(256, row.proxy_size), updates=1200, scheduler=None,
                    proxy_view='deterministic_evaluation', proxy_labels_sha256=row.proxy_labels_sha256,
                    protocol_version='article1-v3')
    config = kd_config()
    if row.proxy_size != 10000:
        config.pop('epochs')
        config.update(updates=1200, proxy_size=int(row.proxy_size),
                      proxy_labels_sha256=row.proxy_labels_sha256)
    return config


def audit(root, source, out, selected_files, verify_states=True):
    root, source, out = map(lambda p: Path(p).resolve(), (root, source, out))
    if out == source or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('Analysis output overlaps original inputs')
    for folder in ('tables', 'figures', 'snapshots'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    manifest = dict(snapshot_date=now(), analysis_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        analysis_dirty=subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
        inputs=[], issues=[], source_root=str(source), output_root=str(out),
        source_policy='Explicit live files plus compatible optional backup; no historical aggregate inputs',
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__))
    for path in [*root.glob('article1/*.py'), root/'run_article1_pipeline.py', root/'run_article1_feddf_curve.py']:
        stable_copy(path, out/'snapshots'/'code'/path.name, manifest)
    frames = []
    for path in map(Path, selected_files):
        if not path.exists():
            manifest['issues'].append(dict(path=str(path), reason='missing selected input'))
            continue
        copied = stable_copy(path, out/'snapshots'/'results'/path.name, manifest)
        f = read_csv(copied)
        if not f.empty:
            f['input_file'] = str(path.resolve())
            f['input_row'] = np.arange(2, len(f)+2)
            frames.append(f)
    selected = {str(Path(p).resolve()) for p in selected_files}
    inventory = []
    for path in sorted((root/'OUTPUTS').rglob('*.csv')):
        if path.is_relative_to(out):
            continue
        inventory.append(dict(path=str(path.resolve()), sha256=file_sha256(path),
                              mtime_ns=path.stat().st_mtime_ns,
                              selected=str(path.resolve()) in selected,
                              reason='explicit individual results' if str(path.resolve()) in selected else
                              'historical/derived/redundant phase; excluded from observations'))
    table(out, 'input_inventory', pd.DataFrame(inventory))
    manifest['candidate_inputs'] = inventory
    for path in sorted(source.glob('launch_*.json')):
        stable_copy(path, out/'snapshots'/'launches'/path.name, manifest)
    print('Verificando 54 fuentes, particiones, etiquetas, máscaras y subconjuntos…', flush=True)
    coverage, subsets = sources(source, out, manifest, root/'data')
    states = []
    if verify_states:
        import torch
        from article1.local_training import _hash_state
        for row in coverage.itertuples():
            name = f'{row.dataset}-seed{row.seed}-{row.regime}'
            meta = json.loads((out/'snapshots'/'sources'/name/'metadata.json').read_text())
            for k in range(10):
                p = source/'sources'/name/'teachers'/f'teacher_{k:03d}.pt'
                try:
                    before = file_sha256(p)
                    checkpoint = torch.load(p, map_location='cpu', weights_only=True)
                    digest = _hash_state(checkpoint['state_dict'])
                    good = (digest == meta['teacher_state_sha256'][k] == checkpoint['state_sha256']
                            and before == file_sha256(p))
                    states.append(dict(dataset=row.dataset, regime=row.regime, seed=row.seed,
                                       client=k, file_sha256=before, state_sha256=digest,
                                       valid=good, reason='' if good else 'state mismatch'))
                    del checkpoint
                except Exception as exc:
                    states.append(dict(dataset=row.dataset, regime=row.regime, seed=row.seed,
                                       client=k, valid=False, reason=str(exc)))
        table(out, 'teacher_states', pd.DataFrame(states))
        bad = {tuple(r[k] for k in KEY) for r in states if not r['valid']}
        coverage = coverage[[tuple(r) not in bad for r in coverage[KEY].to_numpy()]].copy()
    table(out, 'source_coverage', coverage)
    manifest['checkpoint_verification'] = ('Deserialized weights_only=True on CPU; state hashes and file bytes checked separately. '
        'No inference rerun: cached predictions are linked by provenance, not regenerated.' if verify_states else
        'File bytes only; declared model states not independently verified.')
    raw = pd.concat(frames, ignore_index=True)
    # A run_id collision with differing fields invalidates every version, never select by score.
    duplicate_records, keep = [], []
    for run_id, rows in raw.groupby('run_id', sort=False):
        observed = rows.drop(columns=['input_file', 'input_row']).fillna('<NA>').drop_duplicates()
        conflict = len(observed) != 1
        if len(rows)>1:
            duplicate_records.append(dict(run_id=run_id, copies=len(rows), conflict=conflict,
                                          sources='|'.join(rows.input_file)))
        if conflict:
            keep.extend(rows.index)
        else:
            keep.append(rows.index[0])
    raw = raw.loc[keep].reset_index(drop=True)
    table(out, 'duplicates', pd.DataFrame(duplicate_records, columns=['run_id','copies','conflict','sources']))
    regular = raw[raw.method.ne('presence_prob')].copy()
    audited = validate_rows(regular, coverage, subsets, baseline=False)
    presence = raw[raw.method.eq('presence_prob')].copy()
    cells = read_csv(out/'tables'/'expertise_cells.csv')
    cells['A'] = (cells.train_count>0).astype(int)
    cells['A_minus_M'] = cells.A-cells.M
    cells['diagnostic'] = np.select([cells.train_count.eq(0), cells.expertise_count.eq(0), cells.M.eq(0)],
                                   ['absent_in_train', 'no_expertise_observations', 'below_threshold'], default='accredited')
    table(out, 'mask_cells', cells)
    # Presence requires a different effective mask but the same original expertise source.
    labels = {}
    from article1.progress import local_labels
    for d in DATASETS:
        labels[d] = local_labels(root/'data', d, out, manifest)
    presence_cov = coverage.copy()
    provenance_errors = {}
    for index, row in presence.iterrows():
        try:
            name = f'{row.dataset}-seed{row.seed}-{row.regime}'
            cache = out/'snapshots'/'sources'/name/'teacher_cache.npz'
            meta = json.loads(cache.with_name('metadata.json').read_text())
            A, info = load_presence(cache, meta, labels[row.dataset])
            c = (presence_cov[KEY] == row[KEY]).all(axis=1)
            original_M = presence_cov.loc[c, 'M_sha256'].iloc[0]
            assert row.expertise_M_sha256 == original_M, 'presence original M link'
            assert row.M_sha256 == array_sha256(A), 'presence effective mask'
            for key, value in info.items():
                assert row[key] == value, f'presence {key}'
            presence_cov.loc[c, 'M_sha256'] = array_sha256(A)
        except (OSError, ValueError, KeyError, AssertionError, IndexError) as exc:
            provenance_errors[index] = str(exc)
    if not presence.empty:
        presence = validate_rows(presence, presence_cov, subsets, baseline=False)
        for index, reason in provenance_errors.items():
            presence.loc[index, ['valid','invalid_reason']] = [False, reason]
        audited = pd.concat([audited, presence], ignore_index=True)
    # Validate the full recipe, not just the duplicated scalar columns.
    for index, row in audited.iterrows():
        if row.method not in (*METHODS, CE, 'presence_prob') or json.loads(row.training_recipe_json) != _recipe(row):
            audited.loc[index, 'valid'] = False
            audited.loc[index, 'invalid_reason'] += ';unknown_method_or_recipe'
    print('Reconstruyendo targets y routing desde caches verificados…', flush=True)
    target_checks = []
    for ident, rows in audited[audited.valid & audited.method.ne(CE)].groupby(KEY):
        d, r, s = ident
        cache = out/'snapshots'/'sources'/f'{d}-seed{s}-{r}'/'teacher_cache.npz'
        with np.load(cache, allow_pickle=False) as data:
            z, y, M, idx = [data[k] for k in ('logits','labels','M','proxy_idx')]
        A = cells[(cells[KEY] == pd.Series(ident, index=KEY)).all(axis=1)].pivot(index='client',columns='class',values='A').to_numpy(np.uint8)
        for index, row in rows.iterrows():
            positions = proxy_positions(idx, y, int(row.proxy_size), int(s))
            t = build_target(z[positions], y[positions], A if row.method=='presence_prob' else M,
                             method=row.method, temperature=row.temperature)
            errors = []
            for metric in METRICS[2:]+ROUTING+[SUPPORT_MASS]:
                actual, expected = row.get(metric), t.metrics[metric]
                # Missing derived mass may be recovered, but a conflicting recorded value fails.
                if metric == SUPPORT_MASS and pd.isna(actual) and expected is not None:
                    audited.loc[index, metric] = expected
                    audited.loc[index, 'mass_provenance'] = 'recomputed_verified_cache'
                elif expected is not None and (pd.isna(actual) or not np.isclose(actual, expected, atol=1e-9, rtol=1e-9)):
                    errors.append(metric)
            target_checks.append(dict(run_id=row.run_id, valid=not errors, reason=','.join(errors)))
            if errors:
                audited.loc[index, ['valid','invalid_reason']] = [False, 'target:'+','.join(errors)]
    table(out, 'target_checks', pd.DataFrame(target_checks))
    table(out, 'runs', audited)
    table(out, 'invalid_runs', audited[~audited.valid])
    # The launcher receipts explicitly bind the extension to its EXPERT references.
    receipt_rows = []
    for path in sorted((out/'snapshots'/'launches').glob('launch_*.json')):
        receipt = json.loads(path.read_text())
        for job in receipt.get('jobs', []):
            found = audited[audited.run_id.eq(job['run_id'])]
            reference = audited[audited.run_id.eq(job['expert_run_id'])]
            receipt_rows.append(dict(launch=path.name, run_id=job['run_id'],
                expert_run_id=job['expert_run_id'], declared_output=job['command'][job['command'].index('--results')+1],
                found=len(found), reference_found=len(reference),
                valid=len(found)==len(reference)==1 and bool(found.valid.all() and reference.valid.all())))
    table(out, 'feddf_launch_links', pd.DataFrame(receipt_rows))
    manifest['execution_commits'] = {c: sorted(audited[c].dropna().astype(str).unique())
                                    for c in audited if 'commit' in c}
    manifest['source_issues'] = manifest['issues']
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'{audited.valid.sum()}/{len(audited)} identidades válidas', flush=True)
    return audited, coverage, manifest


def analyses(out, audited, source_coverage, manifest):
    out = Path(out)
    valid = audited[audited.valid].copy()
    full = valid[valid.proxy_size.eq(10000)].copy()
    pairs, exclusions = {}, []
    for name, (left, right) in CONTRASTS.items():
        fields = KD.copy()
        if name.startswith('pooling') or name=='support':
            fields += ROUTING
        if name=='support':
            fields += [SUPPORT_MASS]
        if name=='competence_presence':
            fields.remove('M_sha256')
        p, exc = compatible_pairs(full, left, right, fields=fields, metrics=METRICS+ROUTING)
        if name=='competence_presence' and not p.empty:
            # Reuse the dedicated original-M/recipe comparison in addition to row provenance.
            ids = set(p.run_id_left) | set(p.run_id_right)
            selected = full[full.run_id.isin(ids)]
            compare_presence(selected[selected.method.eq(right)], selected[selected.method.eq(left)])
        pairs[name] = p
        table(out, name+'_pairs', p)
        exclusions.append(exc.assign(contrast=name))
    # CE observations are reused by condition, never counted again as independent runs.
    ce = full[full.method.eq(CE)]
    expanded = pd.concat([ce.assign(regime=r) for r in REGIMES], ignore_index=True)
    full_display = pd.concat([full[full.method.ne(CE)], expanded], ignore_index=True)
    absolute = summary(full_display, ['dataset','regime','method'], METRICS[:2])
    table(out, 'absolute', absolute)
    for name, method in [('expert_ce','expert_prob'), ('feddf_ce','feddf_prob')]:
        p, exc = compatible_pairs(full_display, method, CE, fields=SHARED, metrics=METRICS[:2])
        pairs[name] = p
        table(out, name+'_pairs', p)
        exclusions.append(exc.assign(contrast=name))
    curve = valid[valid.dataset.eq('cifar') & valid.method.isin(['expert_prob','feddf_prob',CE])]
    ce_curve = curve[curve.method.eq(CE)]
    table(out, 'ce_unique', ce_curve)
    curve = pd.concat([curve[curve.method.ne(CE) & curve.regime.isin(FOCAL)],
                       *[ce_curve.assign(regime=r) for r in FOCAL]], ignore_index=True)
    table(out, 'curve_runs', curve)
    curve_pairs = []
    for left, right in [('expert_prob',CE), ('feddf_prob',CE), ('expert_prob','feddf_prob')]:
        p, exc = compatible_pairs(curve, left, right, fields=SHARED if right==CE else KD,
                                  keys=KEY+['proxy_size'], metrics=METRICS[:2])
        p['contrast'] = left+' − '+right
        curve_pairs.append(p)
        exclusions.append(exc.assign(contrast='curve:'+left+'-'+right))
    cp = pd.concat(curve_pairs, ignore_index=True)
    table(out, 'curve_pairs', cp)
    table(out, 'curve_summary', summary(cp, ['dataset','regime','proxy_size','contrast'], ['delta_'+m for m in METRICS[:2]]))
    ps = []
    for name, p in pairs.items():
        if not p.empty:
            s = summary(p, ['dataset','regime'], [c for c in p if c.startswith('delta_')])
            s['contrast'] = name
            ps.append(s)
    contrasts = pd.concat(ps, ignore_index=True)
    table(out, 'contrasts', contrasts)
    table(out, 'pair_exclusions', pd.concat(exclusions, ignore_index=True))
    # Coverage by real expected identity, including paired memberships.
    design = []
    for d,r,s in product(DATASETS, REGIMES, SEEDS):
        for m in BASELINE_METHODS:
            design.append(('baseline',d,r,s,m,10000))
        design.append(('presence',d,r,s,'presence_prob',10000))
    for d,s in product(DATASETS, SEEDS):
        design.append(('CE_10000',d,'shared',s,CE,10000))
    for s,n in product(SEEDS,SIZES[:-1]):
        design.append(('curve_CE','cifar','shared',s,CE,n))
        for r in FOCAL:
            for block,m in [('curve_EXPERT','expert_prob'),('curve_FedDF','feddf_prob')]:
                design.append((block,'cifar',r,s,m,n))
    memberships = set()
    for p in [*pairs.values(), cp]:
        if not p.empty:
            memberships.update(p.run_id_left)
            memberships.update(p.run_id_right)
    allrows = audited.copy()
    allrows.loc[allrows.method.eq(CE), 'regime'] = 'shared'
    exact = []
    for block,d,r,s,m,n in design:
        rows = allrows[(allrows.dataset==d)&(allrows.regime==r)&(allrows.seed==s)&(allrows.method==m)&(allrows.proxy_size==n)]
        good = rows[rows.valid]
        paired = len(good)==1 and good.iloc[0].run_id in memberships
        exact.append(dict(block=block,dataset=d,regime=r,seed=s,method=m,proxy_size=n,
                          found=len(rows),valid=int(len(good)==1),paired=int(paired),
                          status='complete' if paired else 'pending',
                          reason=';'.join(rows.invalid_reason.dropna()) if len(rows) else 'missing'))
    conditions = pd.DataFrame(exact)
    cov = conditions.groupby('block',sort=False).agg(expected=('status','size'), found=('found','sum'),
                valid=('valid','sum'), paired=('paired','sum')).reset_index()
    cov['pending'] = cov.expected-cov.paired
    teacher = pd.DataFrame([dict(block='teachers_masks',expected=54,
                        found=len(manifest['source_inventory']),valid=len(source_coverage),
                        paired=len(source_coverage),pending=54-len(source_coverage))])
    cov = pd.concat([teacher,cov],ignore_index=True)
    table(out, 'coverage', cov)
    table(out, 'expected_conditions', conditions)
    table(out, 'pending_conditions', conditions[conditions.status.ne('complete')])
    optional = []
    for m in ('expert_logit','confidence_logit','consensus_logit','energy_logit'):
        for d,r,s in product(DATASETS,REGIMES,SEEDS):
            rows=full[(full.method==m)&(full.dataset==d)&(full.regime==r)&(full.seed==s)]
            optional.append(dict(method=m,dataset=d,regime=r,seed=s,valid=len(rows)==1))
    table(out, 'optional_coverage', pd.DataFrame(optional))
    table(out, 'temperatures', audited[audited.method.ne(CE)].groupby(['method','temperature','valid']).size().rename('n').reset_index())
    main_ids = allrows.merge(conditions[['dataset','regime','seed','method','proxy_size']],
                             on=['dataset','regime','seed','method','proxy_size'])
    manifest['coverage'] = cov.to_dict('records')
    manifest['unique_main_students'] = int(main_ids[main_ids.valid].run_id.nunique())
    required = [name for name in CONTRASTS if name!='pooling_expert']+['expert_ce','feddf_ce']
    manifest['required_pair_counts'] = {name: len(pairs[name]) for name in required}
    manifest['curve_pairs'] = len(cp)
    manifest['closed'] = bool(cov.pending.eq(0).all() and all(len(pairs[n])==54 for n in required) and len(cp)==135)
    manifest['scope'] = 'Six-method baseline, CE, presence and agreed CIFAR curves; optional controls excluded from closure criterion'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return dict(valid=valid, full=full_display, pairs=pairs, contrasts=contrasts,
                curve=curve, curve_pairs=cp, ce=ce_curve, coverage=cov, manifest=manifest)


def figure(out, name, fig, caption):
    for ext in ('png','pdf'):
        fig.savefig(Path(out)/'figures'/f'{name}.{ext}', dpi=160, bbox_inches='tight')
    plt.close(fig)
    return dict(figure=name, caption=caption)


def effect_figure(out, contrasts, names, metrics, name):
    fig, axes = plt.subplots(len(metrics)*3, len(names), figsize=(5*len(names),3.1*len(metrics)*3),
                             squeeze=False, layout='constrained')
    for j, contrast in enumerate(names):
        for i,(metric,dataset) in enumerate(product(metrics,DATASETS)):
            ax=axes[i,j]
            rows=contrasts[(contrasts.contrast==contrast)&(contrasts.dataset==dataset)&(contrasts.metric==metric)]
            rows=rows.set_index('regime').reindex(REGIMES)
            ax.errorbar(range(6),rows['mean'],yerr=rows.sd,fmt='o',capsize=3,color='#0072B2')
            ax.axhline(0,color='gray',lw=.7)
            counts=rows.n.fillna(0).astype(int)
            ax.set(xticks=range(6),xticklabels=[LABELS[r]+f'\nn={n}' for r,n in zip(REGIMES,counts)],
                   ylabel=('Δ accuracy (pp)' if 'accuracy' in metric else 'Δ '+METRIC_LABELS.get(metric.removeprefix('delta_'),metric.removeprefix('delta_'))),
                   title=dataset+' · '+NAMES[CONTRASTS[contrast][0]]+' − '+NAMES[CONTRASTS[contrast][1]])
            ax.tick_params(axis='x', labelsize=8)
    return figure(out,name,fig,
        'Diferencias dentro de seed, después media ± SD. Signo: primer método menos segundo; accuracy en pp, NLL menor es mejor. '
        'n indica seeds válidas; eje categórico sin distancias experimentales iguales. Tres seeds limitan la inferencia.')


def figures(out, analysis):
    out=Path(out)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42, 'savefig.facecolor':'white'})
    captions=[]
    full, contrasts=analysis['full'],analysis['contrasts']
    # Two readable overview figures; complete numbers are always exported.
    for name, methods in [('overview_references',['feddf_logit','feddf_prob','oracle_logit','oracle_prob',CE]),
                          ('overview_ablations',['feddf_prob','expert_prob','expert_prob_sr','presence_prob',CE])]:
        fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
        for i,metric in enumerate(METRICS[:2]):
            for j,d in enumerate(DATASETS):
                ax=axes[i,j]
                for offset,m in enumerate(methods):
                    rows=full[(full.dataset==d)&(full.method==m)]
                    s=summary(rows,['regime'],[metric]).set_index('regime').reindex(REGIMES)
                    ax.errorbar(np.arange(6)+(offset-2)*.13,s['mean'],yerr=s.sd,fmt='o',markersize=4,
                                capsize=2,color=COLORS[m],label=NAMES[m])
                ax.set(title=d,xticks=range(6),xticklabels=[LABELS[r] for r in REGIMES],
                       ylabel='Accuracy (%)' if i==0 else 'NLL')
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=5,fontsize=9)
        captions.append(figure(out,name,fig,'Student en test oficial T=1, N=10000: media ± SD entre seeds; cobertura y seeds exactas en absolute.csv. '
                               'CE tiene nueve ejecuciones únicas reutilizadas entre regímenes. No se selecciona la mejor variante retrospectivamente.'))
    for name,names in [('selection_competence',['selection','expertise','oracle_expert','competence_presence']),
                       ('aggregation_space',['pooling_feddf','pooling_oracle','pooling_expert'])]:
        captions.append(effect_figure(out,contrasts,names,['delta_student_test_accuracy','delta_student_test_nll'],name))
    captions.append(effect_figure(out,contrasts,['competence_presence','expertise'],
                                  ['delta_fallback_rate','delta_mean_selected_teachers'],'routing'))
    # Five metrics together, retaining scale separation between target and student.
    sr=analysis['pairs']['support']
    fig,axes=plt.subplots(3,5,figsize=(22,10),layout='constrained')
    for i,d in enumerate(DATASETS):
        for j,metric in enumerate(['target_accuracy','target_nll','target_entropy',*METRICS[:2]]):
            ax=axes[i,j]
            rows=contrasts[(contrasts.contrast=='support')&(contrasts.dataset==d)&(contrasts.metric=='delta_'+metric)]
            s=rows.set_index('regime').reindex(REGIMES)
            ax.errorbar(range(6),s['mean'],yerr=s.sd,fmt='o',capsize=3,color='#CC79A7')
            ax.axhline(0,color='gray',lw=.7)
            ax.set(title=d+' · '+METRIC_LABELS[metric],xticks=range(6),xticklabels=[LABELS[r] for r in REGIMES],
                   ylabel='SR − EXPERT (pp)' if 'accuracy' in metric else 'SR − EXPERT')
            ax.tick_params(axis='x',rotation=25,labelsize=9)
    captions.append(figure(out,'support_target_student',fig,'SR − EXPERT completo; media ± SD de diferencias emparejadas. '
        'Target: proxy a T=8; student: test oficial a T=1. Sus niveles absolutos no definen un gap de generalización. '
        'Routing y masa previa compartidos; target NLL no aumenta por renormalización sobre soporte que contiene la etiqueta verdadera.'))
    mass=analysis['valid'][analysis['valid'].method.eq('expert_prob') & analysis['valid'].proxy_size.eq(10000)][KEY+[SUPPORT_MASS]]
    joined=sr.merge(mass,on=KEY,validate='one_to_one')
    table(out,'support_mass_pairs',joined)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,metric in zip(axes,['target_nll','student_test_accuracy','student_test_nll']):
        for d,g in joined.groupby('dataset'):
            ax.scatter(g[SUPPORT_MASS],g['delta_'+metric]*(100 if 'accuracy' in metric else 1),label=d,alpha=.75)
        ax.axhline(0,color='gray',lw=.7)
        ax.set(xlabel='Masa real fuera de expertise, antes de SR',ylabel='Δ '+METRIC_LABELS[metric]+(' (pp)' if 'accuracy' in metric else ''))
        ax.legend()
    captions.append(figure(out,'support_mass',fig,'Cada punto es una condición/seed. Masa promedio sobre eventos teacher/muestra seleccionados, excluyendo fallback; '
        'no necesariamente masa eliminada del ensemble por muestra. Asociación descriptiva; no demuestra destrucción de dark knowledge.'))
    ce=analysis['ce']
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,metric in zip(axes,METRICS[:2]):
        scale=100 if 'accuracy' in metric else 1
        for s,g in ce.groupby('seed'):
            g=g.sort_values('proxy_size')
            ax.plot(g.proxy_size,g[metric]*scale,'o--',alpha=.6,label=f'seed {s}')
        means=ce.groupby('proxy_size')[metric].mean()*scale
        ax.plot(means.index,means,'k-o',lw=2,label='Media')
        ax.set(xscale='log',xticks=SIZES,xticklabels=SIZES,xlabel='N (escala log)',
               ylabel='Accuracy (%)' if 'accuracy' in metric else 'NLL')
        ax.legend()
    captions.append(figure(out,'ce_only',fig,'Solo CE, CIFAR-10: tres seeds reales y su media, sin replicar regímenes. '
        '1200 actualizaciones en todos los puntos; N=100 usa batch efectivo 100 y cambia ejemplos consumidos.'))
    curve=analysis['curve']
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for i,metric in enumerate(METRICS[:2]):
        for j,r in enumerate(FOCAL):
            ax=axes[i,j]
            for m in ['expert_prob','feddf_prob',CE]:
                g=curve[(curve.regime==r)&(curve.method==m)]
                s=summary(g,['proxy_size'],[metric]).sort_values('proxy_size')
                ax.errorbar(s.proxy_size,s['mean'],yerr=s.sd,fmt='o-',capsize=3,label=NAMES[m],color=COLORS[m])
            ax.set(title=LABELS[r],xscale='log',xticks=SIZES,xticklabels=SIZES,xlabel='N',
                   ylabel='Accuracy (%)' if i==0 else 'NLL')
            ax.legend(fontsize=8)
    captions.append(figure(out,'proxy_absolute',fig,'CIFAR-10, tres seeds: media ± SD; anclas N=10000 reutilizadas del baseline. '
        'CE compartido entre regímenes. Subconjuntos anidados y trazas emparejadas verificadas; presupuesto de actualizaciones, no de ejemplos.'))
    cps=summary(analysis['curve_pairs'],['regime','proxy_size','contrast'],['delta_'+m for m in METRICS[:2]])
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for i,metric in enumerate(METRICS[:2]):
        for j,r in enumerate(FOCAL):
            ax=axes[i,j]
            for contrast,g in cps[(cps.regime==r)&(cps.metric=='delta_'+metric)].groupby('contrast'):
                g=g.sort_values('proxy_size')
                ax.errorbar(g.proxy_size,g['mean'],yerr=g.sd,fmt='o-',capsize=3,label=contrast.replace(CE,'CE'))
            ax.axhline(0,color='gray',lw=.7)
            ax.set(title=LABELS[r],xscale='log',xticks=SIZES,xticklabels=SIZES,xlabel='N',
                   ylabel='Δ accuracy (pp)' if i==0 else 'Δ NLL')
            ax.legend(fontsize=8)
    captions.append(figure(out,'proxy_differences',fig,'EXPERT − CE, FedDF − CE y EXPERT − FedDF. Media ± SD calculadas después de restar dentro de cada seed. '
        'Un cruce entre N evaluados no identifica N óptimo ni umbral universal. Solo CIFAR-10.'))
    cells=read_csv(out/'tables'/'mask_cells.csv')
    mask_stats=cells.groupby(KEY).agg(presence_density=('A','mean'), expertise_density=('M','mean'),
                   mask_disagreements=('A_minus_M',lambda x: int(x.ne(0).sum())),
                   absent_train=('train_count',lambda x:int(x.eq(0).sum())),
                   no_expertise=('expertise_count',lambda x:int(x.eq(0).sum()))).reset_index()
    support=cells[cells.M.eq(1)].groupby(KEY).expertise_count.agg(['min','median','max']).reset_index()
    mask_stats=mask_stats.merge(support,on=KEY,how='left')
    table(out,'mask_statistics',mask_stats)
    table(out,'experts_per_class',cells.groupby(KEY+['class']).agg(presence_teachers=('A','sum'),expert_teachers=('M','sum')).reset_index())
    # Representative rule fixed independently of the observed treatment effects.
    representatives=[]
    for d in DATASETS:
        for r in FOCAL:
            g=mask_stats[(mask_stats.dataset==d)&(mask_stats.regime==r)].sort_values(['mask_disagreements','seed'])
            if len(g): representatives.append(g.iloc[len(g)//2][KEY].to_dict())
    table(out,'representative_masks',pd.DataFrame(representatives))
    for ident,g in cells.groupby(KEY):
        d,r,s=ident
        fig,axes=plt.subplots(1,5,figsize=(17,3.4),layout='constrained')
        for ax,col in zip(axes,['train_count','expertise_count','A','M','A_minus_M']):
            matrix=g.pivot(index='client',columns='class',values=col)
            im=ax.imshow(matrix,cmap='coolwarm' if col=='A_minus_M' else 'viridis',
                         vmin=-1 if col=='A_minus_M' else 0,
                         vmax=1 if col in ['A','M','A_minus_M'] else None)
            ax.set(title=col,xlabel='Clase',ylabel='Cliente',xticks=range(10),yticks=range(10))
            fig.colorbar(im,ax=ax,shrink=.7)
        fig.suptitle(f'{d} · {LABELS[r]} · seed {s}')
        figure(out,f'masks_{d}_{r}_{s}',fig,'')
    captions.append(dict(figure='masks_*',caption='Suplemento completo: 54 condiciones, train/expertise counts y A/M/A−M. '
        'Ejemplos principales: seed mediana por número de discrepancias, desempate por seed, dentro de dataset/régimen focal; '
        'no seleccionados por ventajas del student. Celdas acreditadas no constituyen validación independiente de su máscara.'))
    relation=analysis['pairs']['competence_presence'].merge(mask_stats,on=KEY,validate='one_to_one')
    table(out,'mask_advantage',relation)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,metric in zip(axes,METRICS[:2]):
        for d,g in relation.groupby('dataset'):
            ax.scatter(g.mask_disagreements,g['delta_'+metric]*(100 if 'accuracy' in metric else 1),label=d,alpha=.7)
        ax.axhline(0,color='gray',lw=.7)
        ax.set(xlabel='Celdas donde A ≠ M',ylabel='EXPERT − presencia: '+METRIC_LABELS[metric]+(' (pp)' if 'accuracy' in metric else ''))
        ax.legend()
    captions.append(figure(out,'mask_advantage',fig,'Relación descriptiva de discrepancias A/M con ventaja de EXPERT. '
        'Puede cambiar el número de participantes; no aísla calidad respecto a cantidad. Clientes y muestras no son réplicas estadísticas.'))
    table(out,'captions',pd.DataFrame(captions))
    return pd.DataFrame(captions)


def narrative(out, analysis):
    out=Path(out)
    contrasts=analysis['contrasts']
    findings=[]
    for name,g in contrasts[contrasts.metric.isin(['delta_student_test_accuracy','delta_student_test_nll'])].groupby('contrast'):
        for d,rows in g.groupby('dataset'):
            acc=rows[rows.metric.eq('delta_student_test_accuracy')]
            nll=rows[rows.metric.eq('delta_student_test_nll')]
            findings.append(dict(contrast=name,dataset=d,
                positive_accuracy_regimes=','.join(acc.loc[acc['mean']>0,'regime']),
                negative_accuracy_regimes=','.join(acc.loc[acc['mean']<0,'regime']),
                negative_nll_regimes=','.join(nll.loc[nll['mean']<0,'regime']),
                accuracy_min_pp=acc['mean'].min(),accuracy_max_pp=acc['mean'].max(),
                complete_regimes=int(acc.n.eq(3).sum())))
    table(out,'findings',pd.DataFrame(findings))
    sections=[
        ('1. Introducción','¿Aporta medir competencia de clase al transferir ensembles privados?',
         'selection, expertise, competence_presence','selection_competence; contrasts',
         'No superioridad universal, ni comparación exhaustiva SOTA.'),
        ('2. Related Work','Posicionar selección y pooling respecto a ensemble distillation y one-shot FL.',
         'Definiciones e implementaciones de FedDF, routing y SR; búsqueda bibliográfica pendiente.',
         'Método y protocolo; sin tabla SOTA nueva','Estos controles internos no sustituyen benchmark externo; no inventar citas.'),
        ('3. Método','A desde train; M desde expertise; pooling uniforme seleccionado y fallback logit.',
         'mask_cells, target_checks y código de distillation/presence','masks_*; routing',
         'ORACLE usa etiquetas proxy y no es cota garantizada; EXPERT tampoco es label-free.'),
        ('4. Protocolo experimental','Tres seeds, splits disjuntos, 10 teachers, 1200 actualizaciones; auditoría de compatibilidad.',
         'manifest; teacher_states; proxy_subsets; runs; expected_conditions','coverage; input_inventory',
         'Hashes registrados del student no reevalúan test; caches no regenerados mediante inferencia. N=100 cambia batch y ejemplos.'),
        ('5. Resultados','Selección → competencia/presencia → pooling → target/student → presupuesto proxy.',
         'Contrastes emparejados individuales; conclusiones desagregadas por dataset/régimen.',
         'overview_references; overview_ablations; selection_competence; aggregation_space; support_target_student; ce_only; proxy_differences',
         'No equivalencia por diferencias pequeñas; SR cambia información y concentración; curva solo CIFAR-10.'),
        ('6. Discusión y limitaciones','Separar observaciones de mecanismos y alcance.',
         'Excepciones en findings; opcionales y condiciones pendientes explícitas.',
         'findings; optional_coverage; pending_conditions; support_mass',
         'n=3 limita inferencia; no generalizar entre datasets ni atribuir todo a calidad del teacher.'),
        ('7. Conclusiones y futuro','Concluir únicamente los contrastes respaldados; formular controles adicionales.',
         'Mapa hipótesis–evidencia y estado de cierre','claim_evidence',
         'Entropía/concentración, unlabeled proxy, inversión y personalización: futuro, no evidencia actual.'),
    ]
    structure=pd.DataFrame(sections,columns=['section','question_message','evidence','figure_table','cannot_claim'])
    table(out,'article_structure',structure)
    claims=[]
    for name,(left,right) in CONTRASTS.items():
        p=analysis['pairs'][name]
        rows=pd.DataFrame(findings)
        result=rows[rows.contrast.eq(name)].to_dict('records')
        claims.append(dict(hypothesis=name,contrast=f'{left} − {right}',result=json.dumps(result,ensure_ascii=False),
                           defensible_claim='Efecto observado del procedimiento por dataset/régimen; medias y SD en contrasts.csv.',
                           limitation=('SR cambia concentración e información interclase; NLL target tiene explicación algebraica.' if name=='support' else
                                       'Calidad y número de teachers pueden cambiar.' if name=='competence_presence' else
                                       'Tres seeds; no equivalencia, causalidad de mecanismo ni superioridad universal.'),
                           status=f'{len(p)}/54 pares; '+('completo' if len(p)==54 else 'alcance parcial')))
    claims.append(dict(hypothesis='public_labels_proxy_budget',contrast='EXPERT−CE / FedDF−CE / EXPERT−FedDF',
                        result=f"{len(analysis['curve_pairs'])}/135 pares en CIFAR-10",defensible_claim='Curvas para cinco N evaluados.',
                        limitation='CE reutilizado; 1200 updates; batch 100 para N=100; cruces no son óptimos.',status='descriptivo'))
    table(out,'claim_evidence',pd.DataFrame(claims))
    return structure,pd.DataFrame(claims),pd.DataFrame(findings)


def interpretation(out, analysis):
    """Generate restrained, numerical prose from the audited paired observations."""
    out=Path(out)
    s=analysis['contrasts']
    lines=['## Lectura de los contrastes observados',
           'Los números siguientes son medias ± SD de diferencias dentro de seed; no son intervalos de confianza.']
    for name in ['competence_presence','selection','expertise','oracle_expert','pooling_feddf','pooling_oracle','pooling_expert']:
        left,right=CONTRASTS[name]
        lines.append(f'### {NAMES[left]} − {NAMES[right]}')
        for d in DATASETS:
            rows=s[(s.contrast==name)&(s.dataset==d)&(s.metric=='delta_student_test_accuracy')]
            parts=[]
            for r in REGIMES:
                g=rows[rows.regime==r]
                if g.empty:
                    parts.append(f'{LABELS[r]}: sin pares')
                    continue
                v=g.iloc[0]
                sd=f'{v.sd:.2f}' if pd.notna(v.sd) else 'no estimable'
                parts.append(f'{LABELS[r]}: {v["mean"]:+.2f} ± {sd} pp (n={v.n})')
            lines.append(f'**{d}**. '+'; '.join(parts)+'.')
    lines += ['### Presencia: alcance de la acreditación',
              'En Multi y Single las máscaras A y M coinciden en las condiciones verificadas y los resultados de ambos brazos coinciden. '
              'Este resultado describe esas máscaras concretas; no constituye una prueba estadística general de equivalencia.',
              'La ventaja de accuracy en Dirichlet debe leerse junto al NLL y al cambio en el número de participantes. '
              'En CIFAR-10 y Fashion-MNIST el NLL de EXPERT frente a presencia empeora en α=1 y α=.1, pese a la ventaja media de accuracy. '
              'Por tanto, acreditar competencia no proporciona una mejora uniforme de ambas métricas.']
    lines.append('### SR − EXPERT: excepciones y mecanismo abierto')
    for d in DATASETS:
        parts=[]
        for metric in ['target_nll','student_test_accuracy','student_test_nll']:
            g=s[(s.contrast=='support')&(s.dataset==d)&(s.metric=='delta_'+metric)]
            pos=[LABELS[r] for r in REGIMES if r in set(g.loc[g['mean']>1e-9,'regime'])]
            neg=[LABELS[r] for r in REGIMES if r in set(g.loc[g['mean']< -1e-9,'regime'])]
            parts.append(f'{METRIC_LABELS[metric]}: sube en {", ".join(pos) or "ninguno"}; baja en {", ".join(neg) or "ninguno"}')
        lines.append(f'**{d}**. '+'. '.join(parts)+'.')
    lines.append('El target tiene una garantía algebraica sobre la etiqueta proxy que no se transfiere al riesgo del student. '
                 'Las excepciones en MNIST y Fashion-MNIST impiden afirmar daño universal por SR; falta un control de concentración para distinguir mecanismos.')
    lines.append('### Curva CIFAR-10')
    cp=summary(analysis['curve_pairs'],['regime','proxy_size','contrast'],['delta_student_test_accuracy','delta_student_test_nll'])
    for r in FOCAL:
        for contrast,g in cp[(cp.regime==r)&(cp.metric=='delta_student_test_accuracy')].groupby('contrast'):
            g=g.sort_values('proxy_size')
            endpoints=[f'N={int(v.proxy_size)}: {v["mean"]:+.2f} ± {v.sd:.2f} pp' for _,v in g[g.proxy_size.isin([100,10000])].iterrows()]
            crosses=[]
            records=g.to_dict('records')
            for a,b in zip(records,records[1:]):
                if a['mean']*b['mean']<0:
                    crosses.append(f'{a["proxy_size"]}–{b["proxy_size"]}')
            lines.append(f'**{LABELS[r]}, {contrast.replace(CE,"CE")}**: '+ '; '.join(endpoints)+
                         '. Cambios de signo de la media entre puntos: '+(', '.join(crosses) or 'ninguno')+'.')
    lines.append('Los cambios de signo son interpolaciones descriptivas entre puntos evaluados, no umbrales identificados. '
                 'En IID, EXPERT−FedDF tiene accuracy media negativa en los cinco tamaños; en α=.1 y Single es positiva. '
                 'La ventaja frente a CE depende de N y no se deduce de la ventaja frente a FedDF.')
    result='\n\n'.join(lines)+'\n'
    (out/'observed_results.md').write_text(result)
    return result
