"""Aggregate case source registers; retain each actual URL and licensing caveat."""
from pathlib import Path
import csv,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 rows=[];seen=set()
 for name in ['01_Air_Pollution/sources.csv','03_Water_Monitoring/SOURCE_REGISTER.csv','04_Vehicle_Classification/SOURCES.csv']:
  p=ROOT/name;case=p.parent.name
  for r in csv.DictReader(p.open(encoding='utf-8-sig',newline='')):
   key=(r.get('source_id'),r.get('url'))
   if key in seen:continue
   seen.add(key)
   row={'source_id':r.get('source_id'),'purpose':r.get('purpose',r.get('type','')),'title':r.get('title',''),'organization_author':r.get('organization_author',r.get('organization','')),'url':r.get('url',''),'access_date':r.get('access_date',r.get('accessed','2026-09-25')),'license_status':r.get('license_status','not established'),'geography':r.get('geography',''),'period':r.get('period',''),'local_file':r.get('local_file',''),'sha256':r.get('sha256',''),'source_pdf_sha256':r.get('source_pdf_sha256',''),'used_in':case}
   loc=row['local_file']
   if loc and not (ROOT/loc).exists() and (p.parent/loc).exists():row['local_file']=(p.parent/loc).relative_to(ROOT).as_posix()
   file=ROOT/row['local_file']
   if row['local_file'] and file.is_file():row['sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
   row['source_archive_sha256']=''
   if row['local_file'].replace('\\','/').startswith('research_notes/'):
    row['source_archive_sha256']=row['sha256']
    row['local_file']='';row['sha256']=''
   if case=='01_Air_Pollution':row['used_in']='01_Air_Pollution;02_Air_Quality_Prediction'
   rows.append(row)
 with (ROOT/'SOURCE_REGISTER.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 parts=['# Проверка выполнимости данных','\nРеальные скачанные образцы, периоды, единицы и признаки проверены до написания итоговых выводов. Размеры, пропуски и хеши приведены в карточках и manifests. Наличие публичной страницы не объявлялось доказательством наличия ряда.\n']
 for case in sorted(ROOT.glob('0[1-4]_*')):
  p=case/'DATA_FEASIBILITY.md'
  parts.append('## '+case.name)
  if p.exists():parts.append(p.read_text(encoding='utf-8'))
  else:parts.append((case/'DATA_CARD.md').read_text(encoding='utf-8'))
 (ROOT/'DATA_FEASIBILITY.md').write_text('\n\n'.join(parts),encoding='utf-8')
 print(len(rows),'source records')
if __name__=='__main__':main()
