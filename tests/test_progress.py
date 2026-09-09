import pandas as pd
import pytest
from article1.progress import compatible_pairs, inventory, deduplicate_files, CE

def frame():
 return pd.DataFrame([dict(dataset='mnist',regime='iid',seed=42,method=m,crn='same',score=.8,run_id=m,target_revision=2) for m in ['expert_prob_sr','expert_prob']])

def test_absent_dataset_and_incomplete():
 f=frame(); inv=inventory(f)
 assert (inv[inv.dataset=='cifar'].status=='pending').all()
 p,e=compatible_pairs(f.iloc[:1],'expert_prob_sr','expert_prob',fields=['crn'],metrics=['score'])
 assert p.empty and len(e)==54

@pytest.mark.parametrize('change',['duplicate','crn','old_sr','nan'])
def test_reject(change):
 f=frame()
 if change=='duplicate': f=pd.concat([f,f.iloc[:1]])
 if change=='crn': f.loc[0,'crn']='other'
 if change=='old_sr': f.loc[0,'target_revision']=1
 if change=='nan': f.loc[0,'score']=float('nan')
 p,e=compatible_pairs(f,'expert_prob_sr','expert_prob',fields=['crn'],metrics=['score'])
 assert p.empty and len(e)==54

def test_ce_kd_coexist_and_reuse():
 f=frame(); f['method']=['expert_prob',CE]; f['cache_sha256']=['private','irrelevant']
 p,e=compatible_pairs(f,'expert_prob',CE,fields=['crn'],metrics=['score'])
 assert len(p)==1 and len(e)==53
 assert len(deduplicate_files([f.assign(input_file='a'),f.assign(input_file='b')]))==2
 other=f.copy(); other.loc[0,'score']=.2
 with pytest.raises(ValueError): deduplicate_files([f,other])
 with pytest.raises(ValueError): deduplicate_files([pd.concat([f,f])])

def test_snapshot_csv_hash_and_rows(tmp_path):
 from article1.progress import stable_copy
 from article1.hashes import file_sha256
 source=tmp_path/'source.csv'; source.write_text('a,b\n1,2\n')
 manifest={'inputs':[]}
 copy=stable_copy(source,tmp_path/'copy.csv',manifest)
 assert manifest['inputs'][0]['rows']==1
 assert manifest['inputs'][0]['sha256']==file_sha256(copy)
 source.write_text('a,b\n1,2,3\n')
 with pytest.raises(ValueError): stable_copy(source,tmp_path/'bad.csv',manifest)

def test_sd_undefined_for_single_seed():
 from article1.analysis import summarize
 f=frame().iloc[:1].assign(delta_accuracy_pp=1.)
 stats=summarize(f)
 assert stats.n.iloc[0]==1 and pd.isna(stats.sd.iloc[0])


def test_original_grid_independent_of_six_method_queue():
 from article1.progress import CONFIGURED_BASELINE_METHODS
 inv=inventory(pd.DataFrame())
 assert len(inv)==540 and inv.method.nunique()==10
 assert len(inventory(pd.DataFrame(), methods=CONFIGURED_BASELINE_METHODS))==54*len(CONFIGURED_BASELINE_METHODS)


def test_missing_both_arms_are_enumerated():
 p,e=compatible_pairs(pd.DataFrame(),'expert_prob','feddf_prob')
 assert p.empty and len(e)==54 and e.reason.eq('both_methods_absent').all()


def test_duplicate_file_provenance_preserved():
 f=frame()
 result=deduplicate_files([f.assign(input_file='baseline'),f.assign(input_file='backup')])
 assert len(result)==2 and result.input_file.eq('backup|baseline').all()


def test_export_only_complete_blocks(tmp_path):
 from itertools import product
 from article1 import DATASETS, REGIMES, SEEDS
 from article1.progress import export_ready_blocks
 rows=[dict(dataset=d,regime=r,seed=s,method='expert_prob',run_id=f'{d}-{r}-{s}') for d,r,s in product(DATASETS,REGIMES,SEEDS)]
 raw=pd.DataFrame(rows)
 (tmp_path/'snapshots').mkdir(); (tmp_path/'tables').mkdir()
 raw.to_csv(tmp_path/'snapshots'/'results_baseline.csv',index=False)
 statuses=export_ready_blocks(tmp_path,raw.assign(valid=True))
 assert (tmp_path/'exports'/'results_expertise.csv').read_text()==raw.to_csv(index=False)
 assert not (tmp_path/'exports'/'results_controls.csv').exists()
 assert {r['filename'] for r in statuses if r['status']=='ready'}=={'results_expertise.csv'}
 # Repeated rendering is safe and does not overwrite a conflicting export.
 export_ready_blocks(tmp_path,raw.assign(valid=True))
 (tmp_path/'exports'/'results_expertise.csv').write_text('conflict')
 with pytest.raises(ValueError): export_ready_blocks(tmp_path,raw.assign(valid=True))


def test_snapshot_retries_a_changed_source(tmp_path,monkeypatch):
 import shutil
 from article1.progress import stable_copy
 source=tmp_path/'source.csv'; source.write_text('a,b\n1,2\n')
 original=shutil.copyfile
 calls=[]
 def copying(src,dst):
  result=original(src,dst)
  if not calls: source.write_text('a,b\n3,4\n5,6\n')
  calls.append(1)
  return result
 monkeypatch.setattr(shutil,'copyfile',copying)
 manifest={'inputs':[]}
 target=stable_copy(source,tmp_path/'copy.csv',manifest)
 assert len(calls)==2 and target.read_bytes()==source.read_bytes()
 assert len(manifest['inputs'])==1 and manifest['inputs'][0]['rows']==2


def test_empty_optional_blocks_are_not_runs():
 f=frame()
 result=deduplicate_files([f, pd.DataFrame(columns=f.columns)])
 assert len(result)==len(f)
