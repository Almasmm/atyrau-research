"""Read-only scientific audit of saved fits, features, provenance and v2 content."""
from pathlib import Path,PureWindowsPath
import hashlib,json,re
from urllib.parse import unquote,urlsplit
import numpy as np
import pandas as pd
import joblib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
CASE2=ROOT/'02_Air_Quality_Prediction'
def read(p,**kwargs):return pd.read_csv(p,float_precision='round_trip',**kwargs)
def verify():
    counts={}
    from report_revision import polish
    sample='![Рисунок: residuals](figures/residuals.png) и [CSV](results/test2025.csv)'
    polished=polish(sample)
    assert '](figures/residuals.png)' in polished and '](results/test2025.csv)' in polished
    for case in [HERE,CASE2]:
        meta=json.loads((case/'models/metadata.json').read_text(encoding='utf8'))
        path=case/meta['dataset_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['dataset_sha256']
        d=read(path,parse_dates=['date']);p=read(case/'results/predictions.csv',parse_dates=['target_date']);m=read(case/'results/metrics.csv')
        assert d[d.split=='train'].date.max()<d[d.split=='validation'].date.min()<d[d.split=='test'].date.min()
        test=d[d.split=='test'];np.testing.assert_array_equal(test.date.to_numpy(),p.target_date.to_numpy());np.testing.assert_allclose(test.value_ugm3,p.actual)
        for row in m.itertuples():
            err=p[row.model]-p.actual
            np.testing.assert_allclose([abs(err).mean(),np.sqrt((err**2).mean()),1-(err**2).sum()/((p.actual-p.actual.mean())**2).sum()],[row.MAE,row.RMSE,row.R2],atol=1e-12)
        for name in ['ridge','forest']:
            fitted=joblib.load(case/'models'/f'{name}.joblib')
            np.testing.assert_allclose(np.maximum(0,fitted.predict(test[meta['features']])),p[name],atol=1e-10)
        slides=json.loads((case/'slides_v2.json').read_text(encoding='utf8'))
        assert 13<=len(slides)<=15
        for slide in slides:
            assert len(slide['title'])<=65 and 60<=len(slide['notes'].split())<=110
            assert slide['source']
            if 'figure' in slide:assert (case/slide['figure']).is_file()
            if 'chart' in slide:
                np.testing.assert_allclose(slide['chart']['series'][0]['values'],m.MAE)
        sources=read(case/'sources.csv')
        for relative in sources.local_file.dropna():
            assert 'research_notes' not in relative
            assert (ROOT/relative).exists(),relative
        markdown_images=0
        for markdown in case.rglob('*.md'):
            content=markdown.read_text(encoding='utf8')
            for match in re.finditer(r'!\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))(?:\s+["\'][^\n]*["\'])?\s*\)',content):
                destination=match.group(1) or match.group(2)
                if urlsplit(destination).scheme in {'http','https','data'}:continue
                destination=unquote(urlsplit(destination).path)
                assert not PureWindowsPath(destination).is_absolute() and not Path(destination).is_absolute(),(markdown,destination,'absolute image path')
                image=(markdown.parent/destination).resolve()
                assert image.is_relative_to(ROOT.resolve()),(markdown,destination,'outside package')
                assert image.is_file(),(markdown,destination,'missing image')
                markdown_images+=1
        counts[case.name]={'predictions':len(p),'slides':len(slides),'sources':len(sources),'markdown_images':markdown_images}
    all_daily=read(HERE/'dataset.csv',parse_dates=['date'],dtype={'station_id':str})
    for station,path,modelpath in [('32',CASE2/'dataset.csv',CASE2/'models'),('235',CASE2/'results/sensitivity235/dataset.csv',CASE2/'models/sensitivity235')]:
        d=read(path,parse_dates=['date'])
        series=all_daily[(all_daily.city=='Atyrau')&(all_daily.station_id==station)&(all_daily.parameter=='pm2_5')].set_index('date').value_ugm3
        for lag in [1,2,3,7,14]:
            expected=series.reindex(d.date-pd.Timedelta(days=lag)).to_numpy()
            np.testing.assert_allclose(expected,d[f'lag{lag}'],equal_nan=True,atol=1e-12)
        for window in [3,7,14]:
            expected=[]
            for day in d.date:
                values=series.reindex(pd.date_range(day-pd.Timedelta(days=window),day-pd.Timedelta(days=1))).dropna()
                expected.append(values.mean() if len(values)>=max(2,window//2) else np.nan)
            np.testing.assert_allclose(expected,d[f'rolling_mean{window}'],equal_nan=True,atol=1e-12)
        expected=[]
        for day in d.date:
            values=series.reindex(pd.date_range(day-pd.Timedelta(days=7),day-pd.Timedelta(days=1))).dropna()
            expected.append(values.std(ddof=1) if len(values)>=3 else np.nan)
        np.testing.assert_allclose(expected,d.rolling_std7,equal_nan=True,atol=1e-10)
        meta=json.loads((modelpath/'metadata.json').read_text(encoding='utf8'))
        assert hashlib.sha256((CASE2/meta['dataset_file']).read_bytes()).hexdigest()==meta['dataset_sha256']
        for name in ['ridge','forest']:
            own=joblib.load(modelpath/f'{name}.joblib');primary=joblib.load(CASE2/'models'/f'{name}.joblib')
            assert {k:repr(v) for k,v in own.get_params().items()}=={k:repr(v) for k,v in primary.get_params().items()}
        counts['feature_rows_'+station]=len(d)
    boot=read(CASE2/'results/paired_error_block_sensitivity.csv')
    assert len(boot)==18 and set(boot.block_days)=={7,14,28}
    assert (boot.lower95<boot.upper95).all()
    print(json.dumps({'status':'PASS','checked':counts},ensure_ascii=False,indent=2))
if __name__=='__main__':verify()
