"""Prepare a stable snapshot and execute CPU analysis notebooks, never training."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MPLBACKEND']='Agg'
os.environ.setdefault('MPLCONFIGDIR','/tmp/article1-matplotlib')
import argparse
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from article1.progress import prepare, report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--analysis-root',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,help='Reuse an already prepared immutable snapshot')
    args=parser.parse_args()
    out=args.snapshot or prepare(args.source_root,args.analysis_root,args.data_dir)
    root=Path(__file__).resolve().parent
    if args.snapshot:
        import json, subprocess
        manifest=json.loads((out/'manifest.json').read_text())
        manifest.setdefault('snapshot_preparation_commit', manifest['analysis_commit'])
        manifest['analysis_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        manifest['analysis_dirty']=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    os.environ['ARTICLE1_SNAPSHOT']=str(out)
    (out/'notebooks').mkdir(exist_ok=True)
    # Explicit kernel executable: use the current interpreter without installing anything.
    kernel_dir=out/'jupyter'/'kernels'/'article1-cpu'; kernel_dir.mkdir(parents=True,exist_ok=True)
    (kernel_dir/'kernel.json').write_text(__import__('json').dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='Article1 CPU',language='python')))
    os.environ['JUPYTER_PATH']=str(out/'jupyter')
    os.environ['JUPYTER_RUNTIME_DIR']=str(out/'jupyter'/'runtime')
    os.environ['IPYTHONDIR']=str(out/'ipython')
    for name in ('article1_progress_analysis','article1_expertise_diagnostics','article1_proxy_budget_analysis'):
        notebook=nbformat.read(root/'notebooks'/f'{name}.ipynb',as_version=4)
        NotebookClient(notebook,timeout=900,kernel_name='article1-cpu',resources={'metadata':{'path':str(root)}}).execute()
        nbformat.write(notebook,out/'notebooks'/f'{name}.ipynb')
        print('Executed',name,flush=True)
    print(report(out))
