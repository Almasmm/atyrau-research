"""Reproduce from declared raw inputs in a new folder; promote checked outputs.

No derived features, split, models, predictions or notebook are copied in.
This is an offline raw-input replay, not a claim of re-downloading every source.
"""
from pathlib import Path
import tempfile,shutil,subprocess,sys,os,json,hashlib,time,re
ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    expected_metrics=json.loads((ROOT/'results/metrics.json').read_text(encoding='utf-8'))
    expected_predictions=sha(ROOT/'results/test_predictions.csv')
    expected_split=sha(ROOT/'results/split_manifest.csv')
    expected_dataset=sha(ROOT/'dataset.csv')
    env=os.environ.copy();env['PYTHONNOUSERSITE']='1';env['OPENBLAS_NUM_THREADS']='4'
    start=time.perf_counter();steps=[]
    with tempfile.TemporaryDirectory(prefix='vehicle_raw_replay_',dir=ROOT/'results') as folder:
        target=Path(folder).resolve()
        # TemporaryDirectory only owns this newly created, checked descendant.
        assert target.is_relative_to((ROOT/'results').resolve()) and target.name.startswith('vehicle_raw_replay_')
        shutil.copyfile(ROOT/'pipeline.py',target/'pipeline.py')
        shutil.copytree(ROOT/'scripts',target/'scripts',ignore=shutil.ignore_patterns('__pycache__'))
        pins=json.loads((ROOT/'raw/source_integrity.json').read_text(encoding='utf-8'))
        for rel in ['raw/source_integrity.json',*pins['files']]:
            dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
        absent=['dataset.csv','processed','models','figures','Notebook.ipynb','Notebook.html','results']
        assert all(not (target/name).exists() for name in absent)
        with (ROOT/'results/cold_replay_20260926.log').open('w',encoding='utf-8') as log:
            for script in ['scripts/prepare_data.py','scripts/make_notebook.py','scripts/make_artifacts.py']:
                t=time.perf_counter()
                # An unrelated cwd verifies that scripts derive their own paths.
                result=subprocess.run([sys.executable,str(target/script)],cwd=ROOT.parent,env=env,stdout=log,stderr=subprocess.STDOUT)
                steps.append({'script':script,'returncode':result.returncode,'seconds':time.perf_counter()-t})
                if result.returncode:raise RuntimeError(f'Cold raw-input replay failed at {script}; see cold_replay_20260926.log')
        checks={'metrics_identical_to_original_experiment':json.loads((target/'results/metrics.json').read_text(encoding='utf-8'))==expected_metrics,'test_predictions_identical':sha(target/'results/test_predictions.csv')==expected_predictions,'split_identical':sha(target/'results/split_manifest.csv')==expected_split,'dataset_identical':sha(target/'dataset.csv')==expected_dataset,'notebook_errors_zero':json.loads((target/'results/notebook_execution.json').read_text(encoding='utf-8'))['errors']==0}
        checks['no_author_absolute_paths_in_notebook']=all(not re.search(r'[A-Za-z]:[\\/]+Users[\\/]+[^\\/ "\r\n]+',(target/name).read_text(encoding='utf-8')) for name in ['Notebook.ipynb','Notebook.html'])
        assert all(checks.values()),checks
        # Publish the actual freshly computed outputs after validating invariant results.
        for name in ['results','models','processed','figures']:
            shutil.copytree(target/name,ROOT/name,dirs_exist_ok=True)
        for src in target.iterdir():
            if src.is_file() and src.name!='pipeline.py':shutil.copyfile(src,ROOT/src.name)
        visual=ROOT/'results/notebook_visual_review.json'
        if visual.exists():
            v=json.loads(visual.read_text(encoding='utf-8'));v['html_bytes']=(ROOT/'Notebook.html').stat().st_size;v['revision_date']='2026-09-26';v['raw_only_replay_verified']=True
            visual.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
    out={'status':'PASS','date':'2026-09-26','declared_inputs':'339 pinned raw files, source_integrity.json, source code; same installed dependency environment','initial_derived_artifacts':0,'network_redownload_performed':False,'steps':steps,'checks':checks,'elapsed_seconds':time.perf_counter()-start,'fresh_outputs_promoted':True,'temporary_workspace_removed':True}
    (ROOT/'results/cold_replay_20260926.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
