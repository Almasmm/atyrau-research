"""Manifest, ZIP and unpacked integrity verification; does not alter source data."""
from pathlib import Path
import csv,hashlib,json,zipfile,datetime,subprocess,sys,re,argparse,os
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def allowed(p):
 rel=p.relative_to(ROOT)
 if any(x in {'research_notes','__pycache__','.git','.venv','_build'} for x in rel.parts):return False
 if p.suffix in {'.pyc','.part','.tmp'}:return False
 if p.name in {'SUBMISSION_MANIFEST.csv','demo_source_page.html'}:return False
 return p.is_file()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--retrain-check',action='store_true',help='Recompute science in the unpacked copy, then compare frozen metric tables');args=parser.parse_args()
 files=sorted(p for p in ROOT.rglob('*') if allowed(p))
 rows=[];absolute_paths=[];placeholders=[]
 for p in files:
  rel=p.relative_to(ROOT).as_posix();case=rel.split('/')[0] if re.match(r'0[1-4]_',rel) else 'shared'
  purpose='code' if p.suffix in {'.py','.sh','.ps1','.mjs'} else 'trained model' if '/models/' in rel else 'research artifact/data/evidence'
  rows.append({'relative_path':rel,'purpose':purpose,'case':case,'bytes':p.stat().st_size,'sha256':sha(p)})
  if p.suffix in {'.py','.mjs','.sh','.ps1','.ipynb','.json','.md','.csv'} and p.stat().st_size<4_000_000:
   txt=p.read_text(encoding='utf-8-sig',errors='replace')
   if re.search(r'[A-Za-z]:[\\/]+Users[\\/]+[A-Za-z][^\\/ "\r\n]*|/mnt/[a-z]/Users/[A-Za-z][^/ "\r\n]*',txt,re.I):absolute_paths.append(rel)
   if re.search(r'\bTODO\b|\bFIXME\b|lorem ipsum',txt,re.I) and p.name!='package_submission.py':placeholders.append(rel)
 if absolute_paths:raise ValueError('Author-specific absolute paths: '+str(absolute_paths))
 if placeholders:raise ValueError('Unresolved placeholders require review: '+str(placeholders))
 # A later editorial-only repack may retain evidence of the independent run.
 # Bind that evidence to exact code, inputs, fitted models and prediction files.
 computational=[(r['relative_path'],r['sha256']) for r in rows if
  r['purpose']=='code' or '/raw/' in r['relative_path'] or '/models/' in r['relative_path'] or
  Path(r['relative_path']).name in {'dataset.csv','metrics.csv','metrics.json','predictions.csv','test_predictions.csv','split_manifest.csv'}]
 science_fingerprint=hashlib.sha256(json.dumps(computational,ensure_ascii=True).encode()).hexdigest()
 manifest=ROOT/'SUBMISSION_MANIFEST.csv'
 with manifest.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
 build=ROOT.parent/'_build'/('archive_'+stamp);build.mkdir(parents=True)
 archive=build/'Case_Studies_Atyrau_SUBMISSION.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
  for p in files+[manifest]:z.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
 # Unique staging directory; never recursively delete or overwrite prior extraction.
 dest=ROOT.parent/'_build'/('unpacked_'+stamp)
 dest.mkdir(parents=True)
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for info in z.infolist():
   target=(dest/info.filename).resolve()
   assert target.is_relative_to(dest.resolve()),info.filename
  z.extractall(dest)
 restored=dest/ROOT.name
 for r in rows:
  p=restored/r['relative_path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],r['relative_path']
 res=subprocess.run([sys.executable,str(restored/'scripts/verify_package.py')],cwd=restored)
 if res.returncode:raise RuntimeError('Unpacked artifact verification failed')
 retraining='NOT_RUN'
 if args.retrain_check:
  res=subprocess.run([sys.executable,str(restored/'scripts/reproduce.py'),'--mode','science','--case','all'],cwd=restored)
  if res.returncode:raise RuntimeError('Independent unpacked science reproduction failed')
  import pandas as pd
  import numpy as np
  for case in ['01_Air_Pollution','02_Air_Quality_Prediction','03_Water_Monitoring']:
   for name in ['metrics.csv','predictions.csv']:
    rel=case+'/results/'+name
    pd.testing.assert_frame_equal(pd.read_csv(ROOT/rel),pd.read_csv(restored/rel),check_exact=False,rtol=1e-10,atol=1e-10)
  rel='04_Vehicle_Classification/results/metrics.json'
  before=json.loads((ROOT/rel).read_text())['test'];after=json.loads((restored/rel).read_text())['test']
  for model,values in before.items():
   for metric in ['macro_f1','accuracy','n']:
    if not np.isclose(values[metric],after[model][metric],rtol=1e-12,atol=1e-12):raise ValueError(f'Replay metric mismatch: {model}/{metric}')
  retraining='PASS: fresh kernels, all four science pipelines, frozen metrics reproduced'
  rel='04_Vehicle_Classification/results/test_predictions.csv'
  pd.testing.assert_frame_equal(pd.read_csv(ROOT/rel),pd.read_csv(restored/rel))
 else:
  proof_path=ROOT.parent/'UNPACKED_REPRODUCTION_CHECK.json'
  if proof_path.exists():
   previous=json.loads(proof_path.read_text(encoding='utf-8'))
   if previous.get('science_fingerprint')==science_fingerprint and previous.get('unpacked_science_retraining','').startswith('PASS'):
    retraining='PASS: identical computational files independently retrained in UNPACKED_REPRODUCTION_CHECK.json; editorial repack'
 final_archive=ROOT.parent/'Case_Studies_Atyrau_SUBMISSION.zip'
 os.replace(archive,final_archive);archive=final_archive
 receipt={'archive':archive.name,'sha256':sha(archive),'bytes':archive.stat().st_size,'file_count':len(files)+1,'crc':'PASS','all_manifest_hashes':'PASS','unpacked_verification':'PASS','unpacked_science_retraining':retraining,'science_fingerprint':science_fingerprint,'excluded':['research_notes and original hydro PDFs','__pycache__','.venv','.git','raw/demo_source_page.html'],'manifest_self_hash':'intentionally omitted'}
 if args.retrain_check:(ROOT.parent/'UNPACKED_REPRODUCTION_CHECK.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf-8')
 (ROOT.parent/'Case_Studies_Atyrau_SUBMISSION.sha256').write_text(receipt['sha256']+'  '+archive.name+'\n',encoding='utf-8')
 (ROOT.parent/'SUBMISSION_ARCHIVE_CHECK.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf-8')
 print(json.dumps(receipt,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
