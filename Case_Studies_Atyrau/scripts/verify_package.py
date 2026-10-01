"""Independent saved-artifact checks. Does not train or tune any model."""
from pathlib import Path
import json,csv,hashlib,sys,zipfile,re,base64,io
from xml.etree import ElementTree as ET
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,root_mean_squared_error,r2_score,f1_score,accuracy_score
from pypdf import PdfReader
import nbformat

ROOT=Path(__file__).resolve().parents[1]
checks=[]
def check(name,ok,detail=''):
    checks.append({'check':name,'status':'PASS' if bool(ok) else 'FAIL','detail':str(detail)})
def close(x,y):return np.isclose(float(x),float(y),rtol=1e-7,atol=1e-7)

def verify():
  for d in sorted(ROOT.glob('0[1-4]_*')):
    k=d.name
    required=['README.md','Report.pdf','Report.docx','Presentation.pptx','Presentation.pdf','Notebook.ipynb','Notebook.html','dataset.csv','DATA_CARD.md','data_dictionary.csv','requirements.txt','models/metadata.json','slides_v2.json','results/presentation_structure.json']
    required+=['results/metrics.json','results/test_predictions.csv','results/split_manifest.csv','demo.mp4'] if k.startswith('04') else ['results/metrics.csv','results/predictions.csv']
    for n in required:
      p=d/n;check(k+'/'+n+' exists',p.is_file() and p.stat().st_size>0)
    models=list((d/'models').glob('*.joblib'));check(k+' saved fitted models',len(models)>=2)
    if (d/'models/metadata.json').is_file():
      metadata=json.loads((d/'models/metadata.json').read_text(encoding='utf-8'))
      check(k+' model dataset hash matches',metadata.get('dataset_sha256')==hashlib.sha256((d/metadata.get('dataset_file','dataset.csv')).read_bytes()).hexdigest())
      check(k+' recorded seed and versions','seed' in metadata and bool(metadata.get('versions')))
    if (d/'dataset.csv').exists():
      df=pd.read_csv(d/'dataset.csv');check(k+' nonempty dataset',len(df)>0,f'{len(df)} rows, {len(df.columns)} columns')
    if (d/'Report.pdf').exists():
      pdf=PdfReader(d/'Report.pdf');n=len(pdf.pages)
      check(k+' report 8-15 pages',8<=n<=15,n)
      check(k+' report pages have content',all(len(p.extract_text() or '')>25 for p in pdf.pages))
      report_text='\n'.join(p.extract_text() or '' for p in pdf.pages)
      check(k+' no duplicate figure prefixes',not re.search(r'Рисунок\s+\d+[.]?\s+Рисунок',report_text))
      check(k+' readable Cyrillic text','данн' in report_text.lower() and '\ufffd' not in report_text)
    if (d/'Presentation.pptx').exists():
      with zipfile.ZipFile(d/'Presentation.pptx') as z:
        slides=[n for n in z.namelist() if re.match(r'ppt/slides/slide\d+\.xml$',n)]
        notes=[n for n in z.namelist() if re.match(r'ppt/notesSlides/notesSlide\d+\.xml$',n)]
        check(k+' slides 10-15',10<=len(slides)<=15,len(slides))
        check(k+' notes for each slide',len(notes)==len(slides),len(notes))
        check(k+' editable slide text',all(b'<a:t>' in z.read(n) for n in slides))
        ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
        notes_text=[' '.join(x.text or '' for x in ET.fromstring(z.read(n)).findall('.//a:t',ns)) for n in notes]
        check(k+' substantive speaker notes',all(len(x)>120 for x in notes_text))
        charts=[n for n in z.namelist() if re.match(r'ppt/(?:slides/)?charts/chart\d+\.xml$',n)]
        check(k+' native evidence chart',len(charts)>0)
        books=[n for n in z.namelist() if n.startswith('ppt/embeddings/') and n.endswith('.xlsx')]
        check(k+' embedded chart workbooks',len(books)>=len(charts))
        if (d/'slides_v2.json').exists():
          spec=json.loads((d/'slides_v2.json').read_text(encoding='utf-8'));spec=spec.get('slides') if isinstance(spec,dict) else spec
          check(k+' deck count matches narrative',len(spec)==len(slides))
          check(k+' sources cited on slides',all(bool(s.get('source')) for s in spec))
          visible_metrics=True
          for index,s in enumerate(spec,1):
            xml=ET.fromstring(z.read(f'ppt/slides/slide{index}.xml'))
            slide_text=' '.join(x.text or '' for x in xml.findall('.//a:t',ns))
            visible_metrics=visible_metrics and all(str(m['value']) in slide_text for m in s.get('metrics',[]))
          check(k+' every headline metric present in PPTX',visible_metrics)
          chart_specs=[s['chart'] for s in spec if s.get('layout')=='chart']
          chart_ok=len(chart_specs)==len(charts)
          chart_ns={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
          for part,expected in zip(sorted(charts),chart_specs):
            series=ET.fromstring(z.read(part)).findall('.//c:ser',chart_ns)
            chart_ok=chart_ok and len(series)==len(expected['series'])
            for node,expected_series in zip(series,expected['series']):
              values=[float(x.text) for x in node.findall('./c:val//c:pt/c:v',chart_ns)]
              expected_values=[round(float(x),3) for x in expected_series['values']]
              chart_ok=chart_ok and len(values)==len(expected_values) and np.allclose(values,expected_values,rtol=0,atol=1e-12)
          check(k+' native chart values match result narrative',chart_ok)
      if (d/'Presentation.pdf').exists():check(k+' PDF/PPT slide counts',len(PdfReader(d/'Presentation.pdf').pages)==len(slides))
    if (d/'Notebook.ipynb').exists():
      nb=nbformat.read(d/'Notebook.ipynb',as_version=4)
      cells=[c for c in nb.cells if c.cell_type=='code' and c.source.strip()]
      errors=[o for c in cells for o in c.get('outputs',[]) if o.output_type=='error']
      check(k+' notebook executed',bool(cells) and all(c.execution_count is not None for c in cells),len(cells))
      check(k+' notebook no errors',not errors,len(errors))
      check(k+' notebook outputs saved',any(c.get('outputs') for c in cells))
      check(k+' sequential clean-kernel execution',[c.execution_count for c in cells]==list(range(1,len(cells)+1)))
      html=(d/'Notebook.html').read_text(encoding='utf-8')
      check(k+' HTML document structure','<html' in html.lower() and '</html>' in html.lower())
      pngs=re.findall(r'data:image/png;base64,([^"\s]+)',html)
      valid=0
      from PIL import Image
      for data in pngs:
        try:
          with Image.open(io.BytesIO(base64.b64decode(data))) as im:im.verify()
          valid+=1
        except Exception:pass
      check(k+' HTML embedded PNG decode',len(pngs)>0 and valid==len(pngs),f'{valid}/{len(pngs)}')
    for p in [d/'Report.docx',d/'Presentation.pptx']:
      if p.exists():
        with zipfile.ZipFile(p) as z:check(k+'/'+p.name+' OOXML CRC',z.testzip() is None)
  for name in ['01_Air_Pollution','02_Air_Quality_Prediction']:
    d=ROOT/name
    if (d/'results/metrics.csv').exists():
      m=pd.read_csv(d/'results/metrics.csv');p=pd.read_csv(d/'results/predictions.csv')
      for _,r in m.iterrows():
        for metric,fun in [('MAE',mean_absolute_error),('RMSE',root_mean_squared_error),('R2',r2_score)]:
          check(name+' '+r['model']+' '+metric,close(fun(p.actual,p[r['model']]),r[metric]))
        check(name+' '+r['model']+' n',len(p)==r['n'])
      check(name+' unique target dates',not p.target_date.duplicated().any())
      if 'latest_feature_date' in p:
        check(name+' latest measurement prior to target date',(pd.to_datetime(p.latest_feature_date)<pd.to_datetime(p.target_date)).all())
      split=json.loads((d/'results/splits.json').read_text())
      check(name+' ordered temporal splits',split['train']['end']<split['validation']['start'] and split['validation']['end']<split['test']['start'])
  d=ROOT/'03_Water_Monitoring'
  secondary=ROOT/'02_Air_Quality_Prediction/results/sensitivity235'
  if (secondary/'metrics.csv').exists():
    sm=pd.read_csv(secondary/'metrics.csv');sp=pd.read_csv(secondary/'predictions.csv')
    for _,r in sm.iterrows():
      for metric,fun in [('MAE',mean_absolute_error),('RMSE',root_mean_squared_error),('R2',r2_score)]:
        check('air secondary235 '+r['model']+' '+metric,close(fun(sp.actual,sp[r['model']]),r[metric]))
      check('air secondary235 '+r['model']+' n',len(sp)==r['n'])
    meta=json.loads((secondary/'metadata.json').read_text())
    check('air secondary235 fitted before test',meta['fit_end']<meta['test_start'])
    check('air secondary235 frozen dataset hash',hashlib.sha256((secondary/'dataset.csv').read_bytes()).hexdigest()==meta['dataset_sha256'])
  if (d/'results/metrics.csv').exists():
    m=pd.read_csv(d/'results/metrics.csv');p=pd.read_csv(d/'results/predictions.csv')
    for _,r in m.iterrows():
      v=p[p.split==r['split']];model=r['model']
      for metric,fun in [('MAE_cm',mean_absolute_error),('RMSE_cm',root_mean_squared_error),('R2',r2_score)]:
        check('water '+r['split']+' '+model+' '+metric,close(fun(v.actual,v[model]),r[metric]))
      check('water '+r['split']+' '+model+' n',len(v)==r['n'])
    origins=pd.to_datetime(p.forecast_origin_date);targets=pd.to_datetime(p.target_time)
    check('water fixed next-day horizon',((targets-origins).dt.days==1).all())
    check('water validation before test',targets[p.split=='validation'].max()<targets[p.split=='test'].min())
  d=ROOT/'04_Vehicle_Classification'
  if (d/'results/metrics.json').exists():
    m=json.loads((d/'results/metrics.json').read_text());p=pd.read_csv(d/'results/test_predictions.csv')
    for model,r in m['test'].items():
      check('vehicle '+model+' macroF1',close(f1_score(p.actual,p[model],average='macro',zero_division=0),r['macro_f1']))
      check('vehicle '+model+' accuracy',close(accuracy_score(p.actual,p[model]),r['accuracy']))
    if (d/'results/split_manifest.csv').exists():
      s=pd.read_csv(d/'results/split_manifest.csv')
      for col in ['object_id','image_id','scene_group']:
        if col in s:check('vehicle no '+col+' split overlap',s.groupby(col)['split'].nunique().max()==1)
    if (d/'demo.mp4').exists():
      import cv2
      cap=cv2.VideoCapture(str(d/'demo.mp4'));fps=cap.get(cv2.CAP_PROP_FPS);n=cap.get(cv2.CAP_PROP_FRAME_COUNT);ok,frame=cap.read();cap.release()
      check('vehicle demo opens',ok and fps>0);check('vehicle demo 20-60 seconds',fps>0 and 20<=n/fps<=60,{'fps':fps,'frames':n,'seconds':n/fps if fps else 0})
  register=ROOT/'SOURCE_REGISTER.csv'
  if register.exists():
    for row in csv.DictReader(register.open(encoding='utf-8-sig',newline='')):
      if row.get('local_file'):
        file=ROOT/row['local_file'];present=file.is_file() or (file.is_dir() and any(p.is_file() for p in file.iterdir()))
        check('source local path '+row['source_id'],present,row['local_file'])
        if file.is_file() and row.get('sha256'):
          check('source hash '+row['source_id'],hashlib.sha256(file.read_bytes()).hexdigest()==row['sha256'])
  out=ROOT/'verification_results.json';out.write_text(json.dumps(checks,indent=2,ensure_ascii=False),encoding='utf-8')
  print(json.dumps({'PASS':sum(x['status']=='PASS' for x in checks),'FAIL':sum(x['status']=='FAIL' for x in checks),'failed':[x for x in checks if x['status']=='FAIL']},ensure_ascii=False,indent=2))
  return not any(x['status']=='FAIL' for x in checks)
if __name__=='__main__':sys.exit(0 if verify() else 1)
