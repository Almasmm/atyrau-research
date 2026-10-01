"""Re-extract public numeric observations; full yearbooks remain outside submission."""
from pathlib import Path
import argparse, calendar, hashlib, json, re, urllib.request
from datetime import datetime, timezone
import pandas as pd
import pdfplumber

CASE = Path(__file__).resolve().parents[1]
SOURCES = {
    2018: {'url':'https://www.kazhydromet.kz/uploads/files/495/file/630840dd1094fbasseyn-rek-ural-srednee-i-nizhnee-techenie-i-emba-i-ustevaya-chast-reki-volga-vypusk-4-2018-god.pdf','page':35,'sha256':'48f6e58ed06631b92b04b84e64dfc771911b415df2eaae364b178687bdd444c9'},
    2019: {'url':'https://kazhydromet.kz/uploads/files/496/file/630840f50875ebasseyn-rek-ural-srednee-i-nizhnee-techenie-i-emba-i-ustevaya-chast-reki-volga-vypusk-4-2019-god.pdf','page':35,'sha256':'754a2b2a1c7768529f040c06c0a41e82346cafe461e535bbc8636a1377d7e476'},
    2022: {'url':'https://www.kazhydromet.kz/uploads/files/1885/file/66e7e0b8ce51deds-vypusk-4-2022-ural-1.pdf','page':37,'sha256':'b67004d491322437a88aa250c050d896afb168bcff504e29416e462d504831fd'},
}

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class SourceIntegrityError(ValueError):
    """The source is not the frozen PDF used by this experiment."""

def verify_source(path, year, info):
    actual=sha256(path)
    expected=info['sha256']
    if actual != expected:
        raise SourceIntegrityError(
            f'W{year}: source PDF SHA-256 mismatch before extraction. '
            f'Expected {expected}; got {actual}; URL: {info["url"]}. '
            'The cached file may be incomplete or the publisher may have changed it. '
            'No dataset or source manifest has been replaced. Obtain the frozen original '
            'or investigate and document a new source version; do not silently update the expected hash.'
        )
    return actual

def prepare(cache=None, download=False):
    cache = Path(cache or CASE.parent/'research_notes'/'water')
    cache.mkdir(parents=True,exist_ok=True)
    rows,checks,manifest = [],[],[]
    for year,info in SOURCES.items():
        path=cache/f'yearbook_{year}.pdf'
        if not path.exists():
            if not download: raise FileNotFoundError(f'{path.name}: use --download --cache <local-cache> to fetch source')
            req=urllib.request.Request(info['url'],headers={'User-Agent':'Academic reproducibility study/1.0'})
            with urllib.request.urlopen(req,timeout=120) as response: path.write_bytes(response.read())
        # Check both cached and freshly downloaded bytes before interpreting the table.
        verify_source(path, year, info)
        with pdfplumber.open(path) as pdf:
            page=pdf.pages[info['page']-1]
            txt=page.extract_text()
            assert '19802' in txt and 'Атырау' in txt and '1.2.' in txt
            lines=txt.splitlines()
            start=next(i for i,line in enumerate(lines) if line.strip()=='1 2 3 4 5 6 7 8 9 10 11 12')+1
            day_lines=lines[start:start+31]
            for day,line in enumerate(day_lines,1):
                # Level values at this station/year are integer 3-digit cm;
                # assert expected cell count instead of silently swallowing missing cells.
                matches=list(re.finditer(r'(?<![\d.])\d{3}(?![\d.])',line))
                months=[m for m in range(1,13) if day<=calendar.monthrange(year,m)[1]]
                assert len(matches)==len(months),(year,day,line)
                for j,(month,m) in enumerate(zip(months,matches)):
                    mark=line[m.end():matches[j+1].start() if j+1<len(matches) else len(line)].strip()
                    rows.append({'date':f'{year}-{month:02d}-{day:02d}','station_id':19802,'level_cm':int(m.group()),'source_marks':mark,'source_year':year,'source_page':info['page'],'source_id':f'W{year}'})
            mean_values=[int(x) for x in re.findall(r'(?<![\d.])\d{3}(?![\d.])',lines[start+31])]
            assert len(mean_values)==12,(year,lines[start+31])
            year_df=pd.DataFrame([r for r in rows if r['source_year']==year])
            for month,published in enumerate(mean_values,1):
                actual=year_df.loc[pd.to_datetime(year_df.date).dt.month==month,'level_cm'].mean()
                checks.append({'year':year,'month':month,'published_mean_cm':published,'computed_mean_cm':actual,'absolute_difference_cm':abs(actual-published),'pass':abs(actual-published)<=0.51})
        manifest.append({'source_id':f'W{year}','year':year,'url':info['url'],'pdf_page':info['page'],'bytes':path.stat().st_size,'sha256':sha256(path),'download_date':'2026-09-25','method':'public HTTPS GET; pdfplumber page extraction','redistributed':False})
    df=pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    assert not df.date.duplicated().any()
    assert len(df)==1095
    assert df.level_cm.notna().all()
    check_df=pd.DataFrame(checks)
    assert check_df['pass'].all(),check_df.loc[~check_df['pass']]
    for out in ['data/raw','results']: (CASE/out).mkdir(parents=True,exist_ok=True)
    df.to_csv(CASE/'data/raw/atyrau_level_observations.csv',index=False)
    df.to_csv(CASE/'dataset.csv',index=False)
    check_df.to_csv(CASE/'results/extraction_validation.csv',index=False)
    (CASE/'data/source_download_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    (CASE/'results/data_quality.json').write_text(json.dumps({'rows':len(df),'unique_dates':df.date.nunique(),'missing_level':int(df.level_cm.isna().sum()),'duplicate_dates':int(df.date.duplicated().sum()),'years':[2018,2019,2022],'omitted_years':[2020,2021],'monthly_mean_checks_passed':int(check_df['pass'].sum()),'monthly_mean_max_abs_difference_cm':float(check_df.absolute_difference_cm.max()),'dataset_sha256':sha256(CASE/'dataset.csv'),'raw_extracted_sha256':sha256(CASE/'data/raw/atyrau_level_observations.csv')},indent=2),encoding='utf-8')
    return df

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cache');parser.add_argument('--download',action='store_true');args=parser.parse_args()
    print(prepare(args.cache,args.download).groupby('source_year').agg(n=('level_cm','size'),minimum=('level_cm','min'),maximum=('level_cm','max')))
