"""Portable entry point. 'verify' does not train; 'science' trains; 'full' exports Office.
All paths derive from this file. Original course documents are never needed or changed.
"""
from pathlib import Path
import argparse,subprocess,sys,os,json,time,hashlib,shutil,datetime
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['verify','science','full'],default='verify');p.add_argument('--case',choices=['all','01','02','03','04'],default='all');p.add_argument('--download-water-sources',action='store_true');a=p.parse_args()
 env=os.environ.copy();env['PYTHONNOUSERSITE']='1';env['PYTHONIOENCODING']='utf-8';env['PYTHONUTF8']='1'
 log_path=ROOT/('reproduction_run.json' if a.mode=='full' else f'reproduction_{a.mode}_run.json')
 log=[]
 def cmd(args,cwd=ROOT,label=None):
  start=time.time();res=subprocess.run([str(x) for x in args],cwd=cwd,env=env)
  log.append({'step':label or ' '.join(str(x) for x in args[1:]),'returncode':res.returncode,'seconds':round(time.time()-start,3),'cwd_relative':str(cwd.relative_to(ROOT))})
  log_path.write_text(json.dumps(log,indent=2,ensure_ascii=False),encoding='utf-8')
  if res.returncode:raise SystemExit(res.returncode)
 def py(rel,*args):cmd([sys.executable,ROOT/rel,*args],label='python '+rel+' '+' '.join(map(str,args)))
 selected=['01','02','03','04'] if a.case=='all' else [a.case]
 if a.mode!='verify':
  if '01' in selected or '02' in selected:
   # Notebook code cells execute the full air experiment in a new kernel.
   # Case10 shares the source dataset prepared by Case1.
   if not (ROOT/'01_Air_Pollution/dataset.csv').exists():py('01_Air_Pollution/pipeline.py','--case','1')
   spec='both' if '01' in selected and '02' in selected else ('1' if '01' in selected else '10')
   py('01_Air_Pollution/build_notebooks.py','--case',spec)
   sensitivity=ROOT/'01_Air_Pollution/sensitivity235.py'
   if sensitivity.exists():py('01_Air_Pollution/sensitivity235.py')
   py('01_Air_Pollution/build_content.py')
   py('01_Air_Pollution/build_slides_v2.py')
  if '03' in selected:
   if a.download_water_sources:
    py('03_Water_Monitoring/code/prepare_data.py','--download','--cache',str(ROOT.parent/'_water_source_cache'))
   else:
    raw=ROOT/'03_Water_Monitoring/data/raw/atyrau_level_observations.csv'
    quality=json.loads((ROOT/'03_Water_Monitoring/results/data_quality.json').read_text())
    if hashlib.sha256(raw.read_bytes()).hexdigest()!=quality['raw_extracted_sha256']:raise ValueError('Water raw cache hash mismatch')
    shutil.copyfile(raw,ROOT/'03_Water_Monitoring/dataset.csv')
   # The fresh notebook runs training, evaluation and historical replay.
   py('03_Water_Monitoring/code/build_notebook.py');py('03_Water_Monitoring/code/build_materials.py')
   py('03_Water_Monitoring/code/build_slides_v2.py')
  if '04' in selected:
   py('04_Vehicle_Classification/scripts/prepare_data.py')
   # The notebook trains every candidate and emits genuine outputs.
   py('04_Vehicle_Classification/scripts/make_notebook.py')
   py('04_Vehicle_Classification/scripts/make_artifacts.py')
  py('scripts/assemble_overview.py');py('scripts/build_registry.py');py('scripts/build_defense.py')
 if a.mode=='full':
  for key in selected:py('scripts/build_documents.py',key)
  runtime=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies'
  env.setdefault('ARTIFACT_TOOL_ENTRY',str(runtime/'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs'))
  env.setdefault('RUNTIME_NODE_MODULES',str(runtime/'node/node_modules'))
  env.setdefault('RUNTIME_PYTHON',str(runtime/'python/python.exe') if sys.platform=='win32' else sys.executable)
  skills=sorted((Path.home()/'.codex/plugins/cache/openai-primary-runtime/presentations').glob('*/skills/presentations'))
  if skills:env.setdefault('PRESENTATION_SKILL_DIR',str(skills[-1]))
  node=env.get('NODE_EXE') or (str(runtime/'node/bin/node.exe') if sys.platform=='win32' else shutil.which('node'))
  if not node:raise RuntimeError('Set NODE_EXE to the bundled Node executable')
  cmd([node,ROOT/'scripts/build_presentations.mjs',','.join(selected)],label='build editable presentations')
  if sys.platform!='win32':raise RuntimeError('PDF conversion requires the documented Windows Office environment; scientific outputs and editable artifacts are already built')
  shell=env.get('POWERSHELL_EXE') or str(runtime/'native/powershell/pwsh.exe')
  if not Path(shell).is_file():shell=shutil.which('pwsh')
  if not shell:raise RuntimeError('Set POWERSHELL_EXE to PowerShell 7; legacy Windows PowerShell export stalled in the tested environment')
  cmd([shell,'-NoProfile','-File',ROOT/'scripts/export_office.ps1','-Case',','.join(selected)],label='Office PDF export (PowerShell 7)')
 if a.case=='all':py('scripts/verify_package.py')
 else:
  if a.case in ['01','02']:py('01_Air_Pollution/pipeline.py','--verify')
  elif a.case=='03':py('03_Water_Monitoring/code/pipeline.py','--verify-only')
  else:py('04_Vehicle_Classification/pipeline.py','--verify-only')
 print('Completed:',a.mode,a.case,'All commands recorded in',log_path.name)
if __name__=='__main__':main()
