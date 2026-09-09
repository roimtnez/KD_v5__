"""Read-only, CPU-only provisional analysis of immutable snapshots.

No checkpoint deserialization, training, downloads or writes to the source tree.
"""
from pathlib import Path
from itertools import product
from datetime import datetime, timezone
import csv
import io
import json
import os
import shutil
import subprocess
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from article1 import DATASETS, REGIMES, SEEDS, THRESHOLDS
from article1.experiments import BASELINE_METHODS
from article1.hashes import file_sha256, array_sha256
from article1.partitioning import load_partitions
from article1.distillation import authority_from_expertise, metadata_identity
from article1.analysis import KEY, METRICS, ROUTING, summarize
from article1.proxy import proxy_positions

SHARED = ['proxy_sha256','proxy_labels_sha256','proxy_master_sha256','student_init_sha256',
          'batch_order_sha256','consumed_batches_sha256','updates','batch_size','optimizer',
          'learning_rate','weight_decay','scheduler','proxy_view','examples_seen','protocol_version']
KD = SHARED + ['cache_sha256','M_sha256','temperature','training_recipe_json']
CONTRASTS = [('oracle_logit','feddf_logit'),('feddf_prob','feddf_logit'),
 ('oracle_prob','oracle_logit'),('expert_prob','feddf_prob'),('oracle_prob','expert_prob'),
 ('expert_prob_sr','expert_prob'),('expert_prob','expert_logit'),
 ('confidence_logit','feddf_logit'),('consensus_logit','feddf_logit'),('energy_logit','feddf_logit')]
CE = 'supervised_proxy_ce'

def now():
    return datetime.now(timezone.utc).isoformat()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def stable_copy(source, destination, manifest, retries=3):
    """Compare inode/stat and two hashes; never combine multiple CSV versions."""
    source, destination = Path(source), Path(destination)
    for _ in range(retries):
        before = source.stat()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        digest = file_sha256(destination)
        after = source.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns) and digest == file_sha256(source):
            record = dict(original=str(source.resolve()), copy=str(destination.resolve()), sha256=digest,
                          bytes=after.st_size, source_mtime_ns=after.st_mtime_ns, snapshot_date=now())
            if source.suffix == '.csv':
                with destination.open(newline='') as f:
                    reader = csv.reader(f, strict=True)
                    header = next(reader)
                    rows = list(reader)
                require(len(header) == len(set(header)) and all(len(r)==len(header) for r in rows), 'Malformed CSV')
                record['rows'] = len(rows)
            manifest['inputs'].append(record)
            return destination
    destination.unlink(missing_ok=True)
    raise ValueError(f'File changed during copy: {source}')

def read_csv(path):
    try:
        return pd.read_csv(path, keep_default_na=False, na_values=[''])
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

def inventory(frame):
    records = []
    for d,r,s,m in product(DATASETS,REGIMES,SEEDS,BASELINE_METHODS):
        rows = frame if frame.empty else frame[(frame.dataset==d)&(frame.regime==r)&(frame.seed==s)&(frame.method==m)]
        status = 'pending' if len(rows)==0 else ('invalid' if len(rows)!=1 or not rows.get('valid',pd.Series(True,index=rows.index)).all() else 'present')
        records.append(dict(dataset=d,regime=r,seed=s,method=m,status=status,rows=len(rows)))
    return pd.DataFrame(records)

def compatible_pairs(frame, left, right, fields=KD, keys=KEY, metrics=METRICS):
    output, excluded = [], []
    subset = frame[frame.method.isin([left,right])] if not frame.empty else frame
    if subset.empty:
        return pd.DataFrame(), pd.DataFrame([dict(reason='both_methods_absent',left=left,right=right)])
    for identity, group in subset.groupby(keys,dropna=False):
        a,b = group[group.method==left],group[group.method==right]
        reason = ''
        if len(a)!=1 or len(b)!=1:
            reason = 'missing_partner_or_duplicate'
        else:
            a,b = a.iloc[0],b.iloc[0]
            bad = [f for f in fields if f not in a or pd.isna(a[f]) or pd.isna(b[f]) or a[f]!=b[f]]
            if not a.get('valid',True) or not b.get('valid',True): bad.append('invalid_row')
            for row in (a,b):
                if row.method=='expert_prob_sr' and row.get('target_revision')!=2: bad.append('old_SR')
            if bad: reason = ','.join(bad)
        ident = dict(zip(keys, identity if isinstance(identity,tuple) else (identity,)))
        if reason:
            excluded.append(dict(**ident,left=left,right=right,reason=reason)); continue
        values = dict(**ident,left=left,right=right)
        for metric in metrics:
            if not np.isfinite([a[metric],b[metric]]).all():
                reason='nonfinite_'+metric; break
            values.update({metric+'_left':a[metric],metric+'_right':b[metric], 'delta_'+metric:a[metric]-b[metric]})
        if reason: excluded.append(dict(**ident,reason=reason))
        else: output.append(values)
    return pd.DataFrame(output), pd.DataFrame(excluded)

def deduplicate_files(frames):
    """Only identical executions across different files may be reused."""
    if not frames: return pd.DataFrame()
    for frame in frames:
        require(not frame.run_id.duplicated().any(), 'Duplicate run_id within file')
    combined = pd.concat(frames,ignore_index=True)
    for _, rows in combined.groupby('run_id'):
        if len(rows)>1:
            require(len(rows.drop(columns=['input_file'],errors='ignore').fillna('<NA>').drop_duplicates())==1,
                    'Conflicting duplicate execution across files')
    return combined.drop_duplicates('run_id').copy()

def save_table(out,name,frame):
    frame.to_csv(out/'tables'/f'{name}.csv',index=False)

def save_figure(out,name,fig):
    for ext in ('png','pdf'): fig.savefig(out/'figures'/f'{name}.{ext}',dpi=120,bbox_inches='tight')
    plt.close(fig)

def effect_plot(out,name,frame,column, x='regime'):
    if frame.empty: return
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,d in zip(axes,DATASETS):
        rows=frame[frame.dataset==d]
        categories=list(REGIMES) if x=='regime' else [100,500,1000,5000,10000]
        for seed,g in rows.groupby('seed'):
            ax.scatter([categories.index(v) for v in g[x]],g[column],label=str(seed),alpha=.65)
        for value,g in rows.groupby(x):
            pos=categories.index(value)
            ax.plot(pos,g[column].mean(),'k_')
            if len(g)>1: ax.errorbar(pos,g[column].mean(),yerr=g[column].std(ddof=1),color='black',capsize=3)
        ax.axhline(0,color='grey',lw=.5)
        ax.set(title=d+' · provisional; n variable',xticks=range(len(categories)),xticklabels=categories,ylabel=column)
        ax.tick_params(axis='x',rotation=30)
        if len(rows): ax.legend(title='seed')
    save_figure(out,name,fig)

def local_labels(data_dir,dataset,out,manifest):
    """Read local training labels without torchvision, downloads or test data."""
    if dataset!='cifar':
        directory='MNIST' if dataset=='mnist' else 'FashionMNIST'
        p=data_dir/directory/'raw'/'train-labels-idx1-ubyte'
        dest=stable_copy(p,out/'snapshots'/'labels'/dataset/p.name,manifest)
        raw=dest.read_bytes()
        require(int.from_bytes(raw[:4],'big')==2049,'Invalid IDX labels')
        y=np.frombuffer(raw[8:],dtype=np.uint8).astype(np.int64)
        require(len(y)==int.from_bytes(raw[4:8],'big'),'Truncated IDX')
        return y
    # CIFAR's official local batches are pickle archives, not model checkpoints.
    import pickle
    labels=[]
    for i in range(1,6):
        p=data_dir/'cifar-10-batches-py'/f'data_batch_{i}'
        dest=stable_copy(p,out/'snapshots'/'labels'/'cifar'/p.name,manifest)
        with dest.open('rb') as f: batch=pickle.load(f,encoding='bytes')
        labels.extend(batch[b'labels'])
        del batch
    return np.asarray(labels,dtype=np.int64)

def sources(original,out,manifest,data_dir):
    cells,selection,coverage,states,subsets=[],[],[],[],[]
    labels={}
    for d in DATASETS:
        try: labels[d]=local_labels(data_dir,d,out,manifest)
        except (OSError,ValueError) as e: manifest['issues'].append(dict(kind='labels_pending',dataset=d,reason=str(e)))
    for d,r,s in product(DATASETS,REGIMES,SEEDS):
        name=f'{d}-seed{s}-{r}'; ident=dict(dataset=d,regime=r,seed=s)
        try:
            src=original/'sources'/name; part=original/'partitions'/name
            dst=out/'snapshots'/'sources'/name; pdst=out/'snapshots'/'partitions'/name
            for p in [src/'metadata.json',src/'teacher_cache.npz',part/'metadata.json',part/'proxy.npz',*[part/f'client_{k:03d}.npz' for k in range(10)]]:
                stable_copy(p,(dst if p.parent==src else pdst)/p.name,manifest)
            meta=json.loads((dst/'metadata.json').read_text())
            require((meta['dataset'],meta['regime'],meta['seed'])==(d,r,s),'Source identity')
            require(meta['protocol_version']=='article1-v3' and meta['threshold']==THRESHOLDS[d],'Protocol/threshold')
            require(file_sha256(dst/'teacher_cache.npz')==meta['cache_sha256'],'Cache hash mismatch')
            require(file_sha256(pdst/'metadata.json')==meta['partition_metadata_sha256'],'Partition link mismatch')
            hashes=meta['teacher_state_sha256']
            require(len(hashes)==10 and hashlib.sha256(''.join(hashes).encode()).hexdigest()==meta['teacher_fingerprint'],'Teacher fingerprint')
            # Hash bytes only: state hashes cannot be recomputed without loading checkpoints.
            for k in range(10):
                p=src/'teachers'/f'teacher_{k:03d}.pt'
                before=p.stat(); digest=file_sha256(p); after=p.stat()
                require(before.st_size>0 and (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Unstable checkpoint')
                manifest['inputs'].append(dict(original=str(p),sha256=digest,bytes=after.st_size,snapshot_date=now(),usage='presence and byte fingerprint only; state not deserialized'))
            proxy,clients,pm=load_partitions(pdst,labels.get(d))
            require((pm['dataset'],pm['regime'],pm['seed'])==(d,r,s),'Partition identity')
            with np.load(dst/'teacher_cache.npz',allow_pickle=False) as cache:
                require({'proxy_idx','labels','logits','M','expertise_accuracy','expertise_counts'}<=set(cache.files),'Incomplete cache')
                M,a,n,y,idx=[cache[k] for k in ('M','expertise_accuracy','expertise_counts','labels','proxy_idx')]
                z=cache['logits']; require(z.shape==(10000,10,10) and np.isfinite(z).all(),'Invalid logits'); del z
            require(np.array_equal(proxy,idx),'Proxy/partition mismatch')
            require(M.shape==a.shape==n.shape==(10,10) and np.isfinite(a).all() and ((a>=0)&(a<=1)).all() and n.dtype.kind in 'iu' and (n>=0).all(),'Invalid expertise')
            require(np.array_equal(M,authority_from_expertise(a,n,THRESHOLDS[d])),'M formula mismatch')
            if d in labels: require(np.array_equal(y,labels[d][idx]),'Proxy labels mismatch')
            records=meta['selection_records']; require(sorted(v['client'] for v in records)==list(range(10)),'Incomplete selection records')
            for v in records:
                require(1<=v['selected_epoch']<=v['epochs_run']<=50 and 0<=v['validation_accuracy']<=1,'Invalid selection record')
            pc=M.sum(0); fallback=int((pc[y]==0).sum())
            coverage.append(dict(**ident,M_density=float(M.mean()),classes_without_expert=json.dumps(np.flatnonzero(pc==0).tolist()),proxy_coverage=1-fallback/len(y),fallback_count=fallback,mean_selected_teachers=float(pc[y].mean()),cache_sha256=meta['cache_sha256'],M_sha256=array_sha256(M),proxy_sha256=array_sha256(idx),proxy_labels_sha256=array_sha256(y),teachers_sha256=meta['teacher_fingerprint'],artifact_creation_commit=meta['cache_creation_commit']))
            for v in records: selection.append(dict(**ident,**v))
            for k,c in product(range(10),repeat=2):
                row=dict(**ident,client=k,**{'class':c},expertise_count=int(n[k,c]),expertise_accuracy=float(a[k,c]),M=int(M[k,c]),experts_for_class=int(pc[c]),classes_for_teacher=int(M[k].sum()),evidence=bool(n[k,c]>0))
                if d in labels:
                    for role in ('train','validation','expertise'):
                        actual_count=int((labels[d][clients[k][role+'_idx']]==c).sum())
                        if role=='expertise': require(actual_count==n[k,c],'Expertise count/partition mismatch')
                        row[role+'_count']=actual_count
                    require(row['expertise_count']==n[k,c],'Expertise count/partition mismatch')
                cells.append(row)
            previous=set()
            for size in (100,500,1000,5000,10000):
                positions=proxy_positions(idx,y,size,s)
                require(previous<=set(idx[positions]),'Non-nested subsets'); previous=set(idx[positions])
                subsets.append(dict(**ident,proxy_size=size,proxy_sha256=array_sha256(idx[positions]),proxy_labels_sha256=array_sha256(y[positions]),proxy_master_sha256=array_sha256(idx)))
            states.append(dict(**ident,status='present',checkpoint_state_verification='unavailable: metadata stores state hashes, not file hashes'))
        except (OSError,ValueError,KeyError) as e:
            states.append(dict(**ident,status='invalid_or_pending',reason=str(e)))
    for name,rows in [('expertise_cells',cells),('selection_records',selection),('coverage',coverage),('source_inventory',states),('proxy_subsets',subsets)]: save_table(out,name,pd.DataFrame(rows))
    manifest['source_inventory']=states
    return pd.DataFrame(coverage),pd.DataFrame(subsets)

def validate_rows(frame,coverage,subsets,baseline=True):
    if frame.empty: return frame
    frame=frame.copy(); errors=[]
    duplicate=frame.duplicated(['dataset','regime','seed','method','proxy_size'],keep=False)|frame.run_id.duplicated(keep=False)
    for i,row in frame.iterrows():
        reasons=[]
        if duplicate.loc[i]: reasons.append('duplicate_identity')
        if row.get('protocol_version')!='article1-v3': reasons.append('protocol')
        if row.get('updates')!=1200: reasons.append('budget')
        if row.method=='expert_prob_sr' and row.get('target_revision')!=2: reasons.append('old_SR')
        needed=SHARED+['student_final_sha256','student_test_accuracy','student_test_nll']
        if row.method!=CE: needed+=KD+METRICS+ROUTING
        for f in set(needed):
            if f not in row or pd.isna(row[f]): reasons.append('missing_'+f)
        for f in ['student_test_accuracy','student_test_nll']+(METRICS+ROUTING if row.method!=CE else []):
            try: require(np.isfinite(float(row[f])),'nonfinite')
            except (ValueError,KeyError,TypeError): reasons.append('nonfinite_'+f)
        if baseline and (row.method not in BASELINE_METHODS or row.get('temperature')!=8 or row.get('proxy_size')!=10000): reasons.append('unexpected_baseline_identity')
        if row.method!=CE:
            cov=coverage[(coverage.dataset==row.dataset)&(coverage.regime==row.regime)&(coverage.seed==row.seed)] if not coverage.empty else coverage
            if len(cov)!=1: reasons.append('unverified_source')
            else:
                for f in ['cache_sha256','M_sha256']:
                    if row.get(f)!=cov.iloc[0][f]: reasons.append(f+'_mismatch')
                if str(row.method).startswith('expert') and row.proxy_size==10000:
                    for f in ['fallback_count','mean_selected_teachers']:
                        if not np.isclose(row[f],cov.iloc[0][f]): reasons.append(f+'_mismatch')
            try:
                expected=metadata_identity(method=row.method,temperature=float(row.temperature),config=json.loads(row.training_recipe_json),source_hash=row.cache_sha256,proxy_hash=row.proxy_sha256,mask_hash=row.M_sha256)
                if expected!=row.run_id: reasons.append('run_id_mismatch')
            except (ValueError,TypeError): reasons.append('invalid_recipe')
        regime='iid' if row.method==CE else row.regime
        sub=subsets[(subsets.dataset==row.dataset)&(subsets.regime==regime)&(subsets.seed==row.seed)&(subsets.proxy_size==row.proxy_size)] if not subsets.empty else subsets
        if len(sub)!=1: reasons.append('proxy_unverified')
        else:
            for f in ['proxy_sha256','proxy_labels_sha256','proxy_master_sha256']:
                if row.get(f)!=sub.iloc[0][f]: reasons.append(f+'_mismatch')
        errors.append(';'.join(sorted(set(reasons))))
    frame['invalid_reason']=errors; frame['valid']=frame.invalid_reason.eq('')
    return frame

def prepare(original,analysis_root,data_dir):
    original=Path(original).resolve(); out=Path(analysis_root).resolve()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    require(not out.is_relative_to(original.parent.parent),'Analysis must be outside active checkout')
    for directory in ('snapshots','tables','figures'): (out/directory).mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    git=lambda *args: subprocess.check_output(['git',*args],cwd=root,text=True).strip()
    manifest=dict(snapshot_date=now(),analysis_commit=git('rev-parse','HEAD'),analysis_dirty=git('status','--porcelain'),original_root=str(original),inputs=[],issues=[],expected_baseline_rows=540)
    for f in ['results_baseline.csv','conditions.csv','results_supervised_proxy.csv','results_proxy_size_expert_prob.csv','results_expertise.csv','results_selection.csv']:
        try: stable_copy(original/f,out/'snapshots'/f,manifest)
        except (OSError,ValueError) as e: manifest['issues'].append(dict(kind='input_pending',path=str(original/f),reason=str(e)))
    for f in ('article1/experiments.py','run_article1_pipeline.py'):
        stable_copy(original.parent.parent/f,out/'snapshots'/'active_code'/f,manifest)
    manifest['active_checkout_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=original.parent.parent,text=True).strip()
    manifest['active_checkout_diff']=subprocess.check_output(['git','diff','--','article1/experiments.py'],cwd=original.parent.parent,text=True)
    coverage,subsets=sources(original,out,manifest,Path(data_dir))
    path=out/'snapshots'/'results_baseline.csv'
    frame=validate_rows(read_csv(path) if path.exists() else pd.DataFrame(),coverage,subsets)
    save_table(out,'baseline_validated',frame); inv=inventory(frame); save_table(out,'completeness',inv)
    manifest['completeness']=inv.to_dict('records')
    manifest['artifact_commits']=sorted(coverage.artifact_creation_commit.unique()) if not coverage.empty else []
    manifest['kd_execution_commits']=sorted(frame.kd_execution_commit.dropna().unique()) if not frame.empty else []
    manifest['checkpoint_limitation']='Checkpoint bytes fingerprinted and all ten files present; metadata only declares state hashes. These cannot be independently matched without deserializing checkpoints, forbidden here. Cache and partition hashes are verified.'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(out,flush=True)
    return out

def completeness(out):
    table=read_csv(Path(out)/'tables'/'completeness.csv')
    matrix=table.pivot(index=KEY,columns='method',values='status')
    print(matrix.to_string())
    return matrix

def progress(out):
    out=Path(out); completeness(out)
    frame=read_csv(out/'tables'/'baseline_validated.csv')
    if (out/'snapshots'/'conditions.csv').exists():
        audit_conditions(out)
    manifest=json.loads((out/'manifest.json').read_text()); manifest['pairs']={}
    all_summaries=[]
    for a,b in CONTRASTS:
        name=a+'__minus__'+b
        fields=KD
        if (a,b) in [('feddf_prob','feddf_logit'),('oracle_prob','oracle_logit'),('expert_prob','expert_logit'),('expert_prob_sr','expert_prob')]:
            fields=KD+ROUTING
        pairs,excluded=compatible_pairs(frame,a,b,fields=fields)
        save_table(out,name+'_paired',pairs); save_table(out,name+'_excluded',excluded)
        manifest['pairs'][name]=dict(complete=len(pairs),excluded=len(excluded),seeds=sorted(pairs.seed.unique().tolist()) if not pairs.empty else [])
        if pairs.empty: continue
        for metric in METRICS:
            summary=summarize(pairs,column='delta_'+metric)
            summary['seeds']=summary.apply(lambda row: ','.join(map(str,sorted(pairs[(pairs.dataset==row.dataset)&(pairs.regime==row.regime)].seed))),axis=1)
            summary['contrast']=name; summary['metric']=metric; all_summaries.append(summary)
        for metric in METRICS[:2]: effect_plot(out,name+'_'+metric,pairs,'delta_'+metric)
    if all_summaries: save_table(out,'contrast_summary',pd.concat(all_summaries,ignore_index=True))
    save_table(out,'target_and_routing_diagnostics',frame[[c for c in KEY+['method','valid','invalid_reason']+METRICS+ROUTING+['pre_restriction_outside_support_mass'] if c in frame]])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest['pairs']

def expertise(out):
    out=Path(out); completeness(out)
    cells=read_csv(out/'tables'/'expertise_cells.csv')
    if cells.empty: print('Aplazado: no hay caches verificados'); return
    selected=cells[cells.M==1]
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for dataset,g in selected.groupby('dataset'):
        values=np.sort(g.expertise_count.to_numpy())
        ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=dataset)
    ax.set(xlabel='Soporte de celdas M=1 (recuento exacto)',ylabel='Fracción acumulada descriptiva')
    ax.legend(); save_figure(out,'selected_support_ecdf',fig)
    save_table(out,'selected_support_exact',selected.groupby(KEY+['expertise_count']).size().rename('cells').reset_index())
    save_table(out,'selected_support_summary',selected.groupby(KEY).expertise_count.describe().reset_index())
    save_table(out,'selected_support_sparse_descriptive',selected.assign(le1=selected.expertise_count<=1,le5=selected.expertise_count<=5,le10=selected.expertise_count<=10).groupby(KEY)[['le1','le5','le10']].sum().reset_index())
    save_table(out,'client_split_totals',cells.groupby(KEY+['client'])[[c for c in ['train_count','validation_count','expertise_count'] if c in cells]].sum().reset_index())
    selection=read_csv(out/'tables'/'selection_records.csv')
    save_table(out,'selection_summary',selection.groupby(KEY)[['selected_epoch','epochs_run','validation_accuracy']].agg(['mean','std','min','max']).reset_index())
    for ident,rows in cells.groupby(KEY,sort=False):
        fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
        for ax,column in zip(axes,['expertise_count','expertise_accuracy','M']):
            values=rows.pivot(index='client',columns='class',values=column).reindex(index=range(10),columns=range(10)).to_numpy(float)
            if column=='expertise_accuracy': values[rows.pivot(index='client',columns='class',values='expertise_count').to_numpy()==0]=np.nan
            cmap=plt.get_cmap('viridis').copy(); cmap.set_bad('#dddddd')
            im=ax.imshow(values,cmap=cmap,vmin=0,vmax=1 if column!='expertise_count' else None)
            ax.set(title=column,xlabel='Clase',ylabel='Cliente',xticks=range(10),yticks=range(10)); fig.colorbar(im,ax=ax)
        fig.suptitle(str(ident)+' · gris: sin evidencia; cero observado: color mínimo')
        save_figure(out,'expertise_'+'_'.join(map(str,ident)),fig)
    return read_csv(out/'tables'/'coverage.csv')

def proxy_budget(out):
    out=Path(out); completeness(out)
    paths=[out/'snapshots'/f for f in ('results_supervised_proxy.csv','results_proxy_size_expert_prob.csv','results_baseline.csv','results_expertise.csv')]
    frames=[read_csv(p).assign(input_file=p.name) for p in paths if p.exists()]
    combined=deduplicate_files(frames)
    if not combined.empty: combined=combined[combined.method.isin([CE,'expert_prob'])]
    combined=validate_rows(combined,read_csv(out/'tables'/'coverage.csv'),read_csv(out/'tables'/'proxy_subsets.csv'),False)
    expected=[(d,'shared_proxy',s,CE,10000) for d,s in product(DATASETS,SEEDS)]
    expected += [('cifar','shared_proxy',s,CE,n) for s,n in product(SEEDS,[100,500,1000,5000])]
    expected += [('cifar',r,s,'expert_prob',n) for r,s,n in product(['iid','alpha0p1','single'],SEEDS,[100,500,1000,5000,10000])]
    rows=[]
    for d,r,s,m,n in expected:
        g=combined[(combined.dataset==d)&(combined.regime==r)&(combined.seed==s)&(combined.method==m)&(combined.proxy_size==n)] if not combined.empty else combined
        rows.append(dict(dataset=d,regime=r,seed=s,method=m,proxy_size=n,status='pending' if len(g)==0 else 'present' if len(g)==1 and g.valid.all() else 'invalid'))
    inv=pd.DataFrame(rows); save_table(out,'proxy_budget_completeness',inv); print(inv.to_string(index=False))
    save_table(out,'proxy_budget_validated',combined)
    for r in ['iid','alpha0p1','single']:
        if combined.empty: continue
        subset=combined[(combined.dataset=='cifar')&((combined.method==CE)|(combined.regime==r))].copy()
        pairs,excluded=compatible_pairs(subset,'expert_prob',CE,fields=SHARED,keys=['dataset','seed','proxy_size'],metrics=METRICS[:2])
        save_table(out,'proxy_'+r+'_paired',pairs); save_table(out,'proxy_'+r+'_excluded',excluded)
        if pairs.empty: continue
        for metric in METRICS[:2]:
            for column in [metric+'_left',metric+'_right','delta_'+metric]: effect_plot(out,'proxy_'+r+'_'+column,pairs,column,x='proxy_size')
            save_table(out,'proxy_'+r+'_'+metric+'_summary',summarize(pairs,'delta_'+metric,groups=['dataset','proxy_size']))
    manifest=json.loads((out/'manifest.json').read_text())
    manifest['proxy_budget_completeness']=inv.to_dict('records')
    manifest['proxy_pairs']={}
    for regime in ['iid','alpha0p1','single']:
        path=out/'tables'/('proxy_'+regime+'_paired.csv')
        pairs=read_csv(path) if path.exists() else pd.DataFrame()
        manifest['proxy_pairs'][regime]={'complete':len(pairs),'seeds':sorted(pairs.seed.unique().tolist()) if not pairs.empty else []}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return inv

def report(out):
    out=Path(out); m=json.loads((out/'manifest.json').read_text()); inv=read_csv(out/'tables'/'completeness.csv')
    counts=inv.groupby(['dataset','status']).size().to_string()
    text=f'''# Artículo 1: instantánea provisional\n\nFecha UTC: {m['snapshot_date']}. Código de análisis: {m['analysis_commit']}.\n\n## Observado\n\n```\n{counts}\n```\n\nFuentes de cache: {len([s for s in m['source_inventory'] if s['status']=='present'])}/54 verificadas por cache, particiones y regla M.\nProcedencia declarada de caches: {m['artifact_commits']}. KD: {m['kd_execution_commits']}.\nEl checkout activo tiene cambios locales en BASELINE_METHODS (seis métodos); no puede atribuirse esa configuración al proceso ya iniciado. No se ha modificado.\n\n## Interpretación\n\nContrastes y seeds exactos en contrast_summary.csv y tablas paired/excluded. Media y SD muestral por condición; SD con una seed queda indefinida. Coberturas diferentes no constituyen comparaciones entre regímenes. Clientes no son réplicas.\nValidation selecciona checkpoint; expertise construye M. No hay test local independiente v3. Accuracy de celdas acreditadas no valida generalización independiente ni rechazo OOD. Sin curvas por época: solo selection_records.\nNo se infiere equivalencia de ausencia de significación, superioridad universal de probabilidades ni pérdida causal de dark knowledge. No se seleccionaron variantes, temperaturas o thresholds con test.\n\n## Pendiente y limitaciones\n\n{m['checkpoint_limitation']}\nNo se cargaron checkpoints ni se hizo inferencia. Fuentes aceptadas para diagnósticos del cache, no como auditoría independiente de los estados serializados.\nCSV supervisado y curva ausentes se enumeran en manifest.json y proxy_budget_completeness.csv. CE se reutiliza entre regímenes sin nuevas réplicas. Un cruce queda acotado entre N evaluados; no define óptimo. 1200 actualizaciones no igualan ejemplos consumidos.\nNo se ejecutó exportador baseline ni análisis definitivo sobre la cuadrícula incompleta. No se lanzó ninguna fase.\nproxy-curve exige results_expertise.csv completo; supervised exige results_selection.csv completo. Exportar bloques de una copia baseline de 540 filas validada antes de preparar esas fases; no relanzar expertise. El bloque selection existente puede ser solo el piloto y entrar en conflicto: usar directorio de exportación limpio.\n'''
    summary_path=out/'tables'/'contrast_summary.csv'
    if summary_path.exists():
        summary=read_csv(summary_path)
        observed=summary[summary.metric.eq('student_test_accuracy') & summary.contrast.eq('expert_prob__minus__feddf_prob') & summary.regime.isin(['iid','single'])]
        text+='\nLa tabla breve muestra los extremos categóricos IID/Single del contraste principal; todos los regímenes, contrastes y NLL están en contrast_summary.csv. No implica orden causal ni superioridad universal.\n'
        text+='\n## Diferencias observadas de accuracy (fracción; SD muestral)\n\n```\n'+observed.to_string(index=False)+'\n```\n'
    (out/'report.md').write_text(text)
    return text


def audit_conditions(out):
    """Compare the frozen conditions table with independently read cache facts."""
    path=out/'snapshots'/'conditions.csv'
    require(path.exists(),'conditions.csv missing: provenance comparisons pending')
    conditions=read_csv(path)
    require(not conditions.duplicated(KEY).any(),'Duplicate conditions')
    expected=set(product(DATASETS,REGIMES,SEEDS))
    require(set(map(tuple,conditions[KEY].to_numpy()))==expected,'Incomplete conditions identities')
    require(conditions.protocol_version.eq('article1-v3').all(),'Conditions protocol')
    require(conditions.expertise_threshold.eq(conditions.dataset.map(THRESHOLDS)).all(),'Conditions thresholds')
    coverage=read_csv(out/'tables'/'coverage.csv')
    merged=conditions.merge(coverage,on=KEY,suffixes=('_declared','_observed'),validate='one_to_one')
    for field in ('proxy_sha256','teachers_sha256','artifact_creation_commit'):
        require(merged[field+'_declared'].eq(merged[field+'_observed']).all(),'Conditions mismatch: '+field)
    require(np.allclose(merged.M_density_declared,merged.M_density_observed),'Conditions M density')
    save_table(out,'conditions_audit',merged)
