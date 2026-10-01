"""Render finished PDFs for manual QA; renders are outside the submission folder."""
from pathlib import Path
import argparse,json,hashlib
import pymupdf
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--case',default='01');a=p.parse_args()
for case in sorted(ROOT.glob('0[1-4]_*')):
 if a.case!='all' and not case.name.startswith(a.case):continue
 out=ROOT.parent/'_build'/case.name/'final_root_review';out.mkdir(parents=True,exist_ok=True)
 data={}
 for stem in ['Report','Presentation']:
  pdf=case/(stem+'.pdf');doc=pymupdf.open(pdf)
  for i,page in enumerate(doc):page.get_pixmap(matrix=pymupdf.Matrix(1.4,1.4)).save(out/f'{stem}-{i+1}.png')
  data[stem]={'pages':len(doc),'sha256':hashlib.sha256(pdf.read_bytes()).hexdigest()}
 (out/'render_manifest.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
 print(case.name,json.dumps(data))
