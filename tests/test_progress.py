import pandas as pd
import pytest
from article1.progress import compatible_pairs, inventory, deduplicate_files, CE

def frame():
 return pd.DataFrame([dict(dataset='mnist',regime='iid',seed=42,method=m,crn='same',score=.8,run_id=m,target_revision=2) for m in ['expert_prob_sr','expert_prob']])

def test_absent_dataset_and_incomplete():
 f=frame(); inv=inventory(f)
 assert (inv[inv.dataset=='cifar'].status=='pending').all()
 p,e=compatible_pairs(f.iloc[:1],'expert_prob_sr','expert_prob',fields=['crn'],metrics=['score'])
 assert p.empty and len(e)==1

@pytest.mark.parametrize('change',['duplicate','crn','old_sr','nan'])
def test_reject(change):
 f=frame()
 if change=='duplicate': f=pd.concat([f,f.iloc[:1]])
 if change=='crn': f.loc[0,'crn']='other'
 if change=='old_sr': f.loc[0,'target_revision']=1
 if change=='nan': f.loc[0,'score']=float('nan')
 p,e=compatible_pairs(f,'expert_prob_sr','expert_prob',fields=['crn'],metrics=['score'])
 assert p.empty and len(e)==1

def test_ce_kd_coexist_and_reuse():
 f=frame(); f['method']=['expert_prob',CE]; f['cache_sha256']=['private','irrelevant']
 p,e=compatible_pairs(f,'expert_prob',CE,fields=['crn'],metrics=['score'])
 assert len(p)==1 and e.empty
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
