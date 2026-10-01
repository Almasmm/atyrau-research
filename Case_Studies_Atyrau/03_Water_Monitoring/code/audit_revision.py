"""Independent 2026-09-26 checks; fault fixtures never enter empirical datasets."""
from pathlib import Path
import argparse, calendar, hashlib, json, os, re, shutil, subprocess, sys, time
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from pipeline import CASE, FEATURES
from replay import replay_observations

def run(sources_dir=None, offline=False):
    checks=[]
    def check(name, ok, detail=''):
        checks.append({'check':name,'status':'PASS' if bool(ok) else 'FAIL','detail':detail})
        assert ok, name
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    data=pd.read_csv(CASE/'dataset.csv',parse_dates=['date'])
    meta=json.loads((CASE/'models/metadata.json').read_text())
    feat=pd.read_csv(CASE/'results/features_and_split.csv',parse_dates=['target_time','forecast_origin_date'])
    predictions=pd.read_csv(CASE/'results/predictions.csv',parse_dates=['target_time','forecast_origin_date'])
    check('raw_equals_dataset', (CASE/'dataset.csv').read_bytes()==(CASE/'data/raw/atyrau_level_observations.csv').read_bytes())
    check('dataset_and_plan_hashes',sha(CASE/'dataset.csv')==meta['dataset_sha256'] and sha(CASE/'EXPERIMENT_PLAN.md')==meta['experiment_plan_sha256'])
    check('dates_units_schema',data.date.is_unique and data.date.is_monotonic_increasing and len(data)==1095 and data.station_id.eq(19802).all() and np.isfinite(data.level_cm).all() and meta['datum_m_BS']==-30)
    levels=data.set_index('date').level_cm
    independent=[]
    for row in feat.itertuples():
        dates=pd.date_range(row.target_time-pd.Timedelta(days=7),row.target_time-pd.Timedelta(days=1))
        x=levels.reindex(dates).to_numpy()
        assert len(x)==7 and np.isfinite(x).all()
        doy=row.target_time.dayofyear
        independent.append([x[-1],x[-1]-x[-2],x[-2]-x[-3],x[-3]-x[-4],x[-1]-np.mean(x[-3:]),x[-1]-np.mean(x),(x[-1]-x[0])/6,np.sin(2*np.pi*doy/365.25),np.cos(2*np.pi*doy/365.25)])
        assert row.actual==levels.loc[row.target_time]
    check('all_nine_features_reconstructed_from_past',np.allclose(np.asarray(independent),feat[FEATURES],atol=1e-10,rtol=0),f'{len(feat)} target rows')
    check('calendar_horizon_and_split',((feat.target_time-feat.forecast_origin_date).dt.days==1).all() and feat.groupby('split').size().to_dict()=={'train':358,'validation':365,'test':358})
    check('gap_and_boundary',feat[feat.split=='test'].target_time.min()==pd.Timestamp('2022-01-08') and feat[feat.split=='validation'].forecast_origin_date.min()==pd.Timestamp('2018-12-31'))
    fit=feat[feat.split.isin(['train','validation'])]; test=feat[feat.split=='test'];testp=predictions[predictions.split=='test']
    ridge=joblib.load(CASE/'models/ridge.joblib');forest=joblib.load(CASE/'models/forest.joblib')
    check('scaler_fit_only_on_723_development_targets',ridge[0].n_samples_seen_==723 and np.allclose(ridge[0].mean_,fit[FEATURES].mean(),atol=1e-10))
    check('model_parameters_match_metadata',ridge[1].alpha==meta['ridge_alpha'] and all(forest.get_params()[k]==v for k,v in meta['hyperparameters']['forest'].items()))
    for name,model in [('ridge',ridge),('forest',forest)]:
        check(name+'_saved_model_reproduces_test',np.allclose(test.level_lag1.to_numpy()+model.predict(test[FEATURES]),testp[name],atol=1e-10))
    for row in pd.read_csv(CASE/'results/metrics.csv').itertuples():
        p=predictions[predictions.split==row.split]
        actual=[mean_absolute_error(p.actual,p[row.model]),root_mean_squared_error(p.actual,p[row.model]),r2_score(p.actual,p[row.model])]
        check(f'metrics_{row.split}_{row.model}',np.allclose(actual,[row.MAE_cm,row.RMSE_cm,row.R2],atol=1e-10) and len(p)==row.n)
    search=pd.read_csv(CASE/'results/validation_search.csv')
    winner=search.sort_values('MAE_cm',kind='stable').iloc[0]
    check('selection_matches_validation_only',winner.model==meta['selected_model']=='ridge' and float(winner.parameter)==meta['ridge_alpha'])
    history=data[data.date.dt.year.isin([2018,2019])]
    threshold=float(history.level_cm.diff().where(history.date.diff().dt.days.eq(1)).abs().quantile(.99))
    replay=replay_observations(data[data.date.dt.year==2022],threshold,'ridge',ridge)
    existing=pd.read_csv(CASE/'results/replay_actions.csv')
    check('replay_original_predictions_unchanged',np.allclose(replay.prediction_next_day_cm,existing.prediction_next_day_cm,equal_nan=True,atol=1e-10))
    check('replay_flags_and_threshold',threshold==31 and replay.statistical_flag.sum()==8 and replay.prediction_next_day_cm.notna().sum()==359)
    # Artificial fault fixtures check control flow only, not predictive quality.
    fixture=pd.DataFrame({'date':pd.date_range('2022-01-01',periods=24),'level_cm':np.arange(24,dtype=float)+200})
    gap=fixture.drop(index=8).reset_index(drop=True)
    gaps=replay_observations(gap,31,'persistence')
    block=gaps[pd.to_datetime(gaps.observation_date).between('2022-01-10','2022-01-15')]
    check('fault_gap_requires_new_seven_day_window',block.prediction_next_day_cm.isna().all() and gaps.loc[pd.to_datetime(gaps.observation_date).eq('2022-01-16'),'prediction_next_day_cm'].notna().all())
    missing=fixture.copy();missing.loc[8,'level_cm']=np.nan
    miss=replay_observations(missing,31,'persistence')
    check('fault_nonfinite_requires_new_seven_day_window',miss.iloc[8:15].prediction_next_day_cm.isna().all() and pd.notna(miss.iloc[15].prediction_next_day_cm) and not miss.iloc[8:10].statistical_flag.any())
    malformed=fixture.astype({'level_cm':'object','date':'object'});malformed.loc[8,'level_cm']='bad-value';malformed.loc[18,'date']='bad-date'
    bad=replay_observations(malformed,31,'persistence')
    check('fault_text_and_date_logged_without_forecast',bad.iloc[8].quality=='missing_or_invalid' and bad.iloc[18].quality=='invalid_date' and bad.iloc[8:15].prediction_next_day_cm.isna().all() and bad.iloc[18:].prediction_next_day_cm.isna().all())
    boundary=fixture.copy();boundary.loc[1,'level_cm']=231
    b=replay_observations(boundary,31,'persistence')
    check('statistical_flag_strict_greater_than',not bool(b.iloc[1].statistical_flag))
    if sources_dir:
        import pymupdf
        from prepare_data import SOURCES,verify_source
        count=0
        for year,info in SOURCES.items():
            path=Path(sources_dir)/f'yearbook_{year}.pdf';verify_source(path,year,info)
            with pymupdf.open(path) as doc:text=doc[info['page']-1].get_text()
            assert '19802' in text and '-30.00' in text and 'СМ' in text
            # Different PDF engine and extraction method from pdfplumber production code.
            body=text.split('12\n',1)[1]
            values=[int(v) for v in re.findall(r'(?<![\d.])\d{3}(?![\d.])',body)[:365]]
            expected=[int(levels.loc[pd.Timestamp(year,month,day)]) for day in range(1,32) for month in range(1,13) if day<=calendar.monthrange(year,month)[1]]
            check(f'PDF_{year}_independent_365_cells',values==expected)
            count+=len(values)
        check('independent_source_total',count==1095)
    else:checks.append({'check':'source_PDF_independent_extraction','status':'NOT_RUN','detail':'Supply --sources-dir; originals deliberately excluded from archive'})
    if offline:
        # No PDF cache, no previous models/predictions, and an unrelated working directory.
        stage=CASE.parents[1]/'_build/03_Water_Monitoring'/('offline_revision_'+str(time.time_ns()))
        for sub in ['code','results']:(stage/sub).mkdir(parents=True,exist_ok=True)
        for name in ['pipeline.py','replay.py']:shutil.copyfile(CASE/'code'/name,stage/'code'/name)
        for name in ['dataset.csv','EXPERIMENT_PLAN.md']:shutil.copyfile(CASE/name,stage/name)
        shutil.copyfile(CASE/'results/extraction_validation.csv',stage/'results/extraction_validation.csv')
        env=os.environ.copy();env['PYTHONNOUSERSITE']='1'
        completed=subprocess.run([sys.executable,str(stage/'code/pipeline.py')],cwd=stage.parent,env=env,capture_output=True,text=True)
        check('isolated_pipeline_exit',completed.returncode==0,completed.stderr[-1000:] if completed.returncode else 'Fresh case: no models, cache or PDF; completed from unrelated cwd')
        fresh=pd.read_csv(stage/'results/predictions.csv')
        check('isolated_predictions_identical',np.allclose(fresh[['actual','persistence','drift','ridge','forest','selected']],predictions[['actual','persistence','drift','ridge','forest','selected']],atol=1e-10))
    out={'date':'2026-09-26','empirical_data_unchanged':True,'checks':checks,'PASS':sum(c['status']=='PASS' for c in checks),'FAIL':sum(c['status']=='FAIL' for c in checks)}
    (CASE/'results/revision_checks_20260926.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sources-dir');p.add_argument('--offline',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.sources_dir,a.offline),ensure_ascii=True,indent=2))
