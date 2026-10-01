"""Reproducible observational-air analysis and next-day forecasting; no fabricated data."""
from pathlib import Path
import argparse, hashlib, json, sys, time, urllib.request, urllib.parse, platform, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn, scipy, joblib
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
CASE2=ROOT/'02_Air_Quality_Prediction'
COMMIT='0fe9a615baf614346e282796f685ce72b852187f'
BASE=f'https://raw.githubusercontent.com/qazybekb/AirDatakz-OpenData/{COMMIT}/rest_of_kz/'
PARAMS=['pm2_5','pm10','no2','so2','co','h2s','o3']
STATIONS={'Atyrau':'32','Aktau':'22','Aktobe':'31'}
CITIES={'Atyrau':'Атырау','Aktau':'Актау','Aktobe':'Актобе'}
MODEL_LABELS={'mean':'Среднее обучения','persistence':'Вчерашнее значение','seasonal_naive7':'Неделю назад','ridge':'Ridge','forest':'Случайный лес'}
POLLUTANT_LABELS={'pm2_5':'PM2.5','pm10':'PM10','no2':'NO₂','so2':'SO₂','co':'CO','h2s':'H₂S','o3':'O₃'}
SEASONS={12:'Зима',1:'Зима',2:'Зима',3:'Весна',4:'Весна',5:'Весна',6:'Лето',7:'Лето',8:'Лето',9:'Осень',10:'Осень',11:'Осень'}
WHO={'pm2_5':15.,'pm10':45.,'no2':25.,'so2':40.,'co':4000.}
KZ={'pm2_5':35.,'pm10':60.,'no2':40.,'so2':50.,'co':3000.}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,obj): Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding='utf8')
def savefig(case,name):
    plt.tight_layout();plt.savefig(case/'figures'/name,dpi=180,bbox_inches='tight');plt.close()
def setup():
    for c in [HERE,CASE2]:
        for d in ['raw','results','models','figures']: (c/d).mkdir(parents=True,exist_ok=True)

def download(url,path):
    # Standard public HTTPS; atomic download, no credentials or access workarounds.
    tmp=Path(str(path)+'.part')
    with urllib.request.urlopen(url,timeout=240) as r,tmp.open('wb') as out: shutil.copyfileobj(r,out)
    tmp.replace(path)

def prepare():
    setup(); logs=[]; manifest=[]; daily=[]; metadata=[]
    lock=json.loads((HERE/'data_lock.json').read_text(encoding='utf8'))
    for param in PARAMS:
        raw=HERE/'raw'/f'{param}.csv.gz'
        if not raw.exists() or raw.stat().st_size==0: download(BASE+f'{param}.csv.gz',raw)
        assert sha(raw)==lock['archive_sha256']['raw/'+raw.name], 'Raw archive checksum changed: '+raw.name
        manifest.append({'source_id':'AIR_'+param,'url':BASE+f'{param}.csv.gz','download_date':'2026-09-25','sha256':sha(raw),'bytes':raw.stat().st_size,'local_file':'raw/'+raw.name,'license':'CC BY-NC 4.0; publisher states upstream KGMT agreement','owner':'AirData.kz / Global Shapers Almaty Hub; upstream KazHydroMet'})
        parts=[];nr=0
        for chunk in pd.read_csv(raw,chunksize=400000,dtype={'station_id':str}):
            nr+=len(chunk)
            parts.append(chunk[chunk['city'].isin(STATIONS)])
        df=pd.concat(parts,ignore_index=True); sel_n=len(df)
        df['time_utc']=pd.to_datetime(df.datetime_utc,utc=True,errors='coerce')
        df['date']=df.time_utc.dt.tz_convert('Asia/Atyrau').dt.tz_localize(None).dt.normalize()
        invalid=df.time_utc.isna()|~np.isfinite(df.value_ugm3)|(df.value_ugm3<0)
        invalid_n=int(invalid.sum());df=df[~invalid].copy()
        unit_bad=int((~np.isclose(df.value_ugm3,df.raw_value*1000,rtol=1e-7,atol=1e-5)).sum())
        if len(df): assert unit_bad==0, 'mg/m3 to ug/m3 disagreement'
        dup=int(df.duplicated(['station_id','time_utc']).sum())
        df=df.drop_duplicates(['station_id','time_utc'])
        df=df[df.date.between('2022-01-01','2025-12-31')].copy()
        # Original per-parameter archive preserved above; selected extracts are traceable derivatives.
        df.to_csv(HERE/'raw'/f'selected_{param}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        stations=df[['city','station_id','station_name','lat','lon']].drop_duplicates();stations['parameter']=param;metadata.append(stations)
        g=df.groupby(['city','station_id','date']).agg(mean_ugm3=('value_ugm3','mean'),n_hours=('time_utc','nunique')).reset_index()
        g['parameter']=param;g['valid_day']=g.n_hours>=18
        assert (g.n_hours<=24).all(), 'Hourly archive contains >24 unique timestamps/day'
        g['value_ugm3']=g.mean_ugm3.where(g.valid_day)
        g['reference_station']=g.apply(lambda x:STATIONS[x.city]==x.station_id,axis=1)
        daily.append(g)
        logs.append({'parameter':param,'source_rows':nr,'selected_city_rows_all_dates':sel_n,'invalid_selected':invalid_n,'duplicate_selected':dup,'unit_conversion_disagreements':unit_bad,'retained_hours_2022_2025':len(df),'station_days':len(g),'valid_station_days':int(g.valid_day.sum())})
    all_daily=pd.concat(daily,ignore_index=True)
    all_daily.to_csv(HERE/'dataset.csv',index=False)
    pd.concat(metadata).to_csv(HERE/'results'/'station_metadata.csv',index=False)
    pd.DataFrame(logs).to_csv(HERE/'results'/'cleaning_audit.csv',index=False)
    pd.DataFrame(manifest).to_csv(HERE/'results'/'download_manifest.csv',index=False)
    # ERA5 is a modeled weather reanalysis, not local pollution measurements or an archived forecast.
    wp=HERE/'raw'/'atyrau_era5.json'
    qs={'latitude':47.096617,'longitude':51.942701,'start_date':'2022-01-01','end_date':'2025-12-31','daily':'temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_mean,precipitation_sum','timezone':'Asia/Atyrau','wind_speed_unit':'ms','models':'era5'}
    wu='https://archive-api.open-meteo.com/v1/archive?'+urllib.parse.urlencode(qs)
    if not wp.exists(): download(wu,wp)
    daily_hash=hashlib.sha256(json.dumps(json.loads(wp.read_text(encoding='utf8'))['daily'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert daily_hash==lock['weather_daily_sha256'], 'ERA5 daily series changed; investigate before reproducing results'
    weather=pd.DataFrame(json.loads(wp.read_text(encoding='utf8'))['daily']).rename(columns={'time':'date'});weather.date=pd.to_datetime(weather.date)
    js(HERE/'results'/'weather_manifest.json',{'url':wu,'sha256':sha(wp),'download_date':'2026-09-25','license':'CC BY 4.0, Open-Meteo / Copernicus ERA5','model':'ERA5','daily_units':json.loads(wp.read_text(encoding='utf8'))['daily_units']})
    weather.to_csv(HERE/'results'/'weather_daily.csv',index=False)
    return all_daily,weather

def calendar(df):
    d=df.date.dt
    df['sin_doy']=np.sin(2*np.pi*d.dayofyear/365.25);df['cos_doy']=np.cos(2*np.pi*d.dayofyear/365.25)
    df['sin_week']=np.sin(2*np.pi*d.dayofweek/7);df['cos_week']=np.cos(2*np.pi*d.dayofweek/7)
    return df

def metric(y,p): return {'MAE':float(mean_absolute_error(y,p)),'RMSE':float(np.sqrt(mean_squared_error(y,p))),'R2':float(r2_score(y,p)),'n':int(len(y))}

def fit_models(case,df,features,forecast=False):
    df=df.copy().sort_values('date')
    df['split']=np.where(df.date<'2024-01-01','train',np.where(df.date<'2025-01-01','validation','test'))
    df.to_csv(case/'results'/'model_dataset.csv',index=False)
    if forecast: df.to_csv(case/'dataset.csv',index=False)
    train=df[df.split=='train'];val=df[df.split=='validation'];test=df[df.split=='test']
    assert train.date.max()<val.date.min() and val.date.max()<test.date.min()
    assert all(len(x)>30 for x in [train,val,test])
    splits={s:{'start':str(g.date.min().date()),'end':str(g.date.max().date()),'n':len(g)} for s,g in df.groupby('split')}
    js(case/'results'/'splits.json',splits)
    candidates={}
    for alpha in [.1,1.,10.]: candidates[f'ridge_{alpha}']=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),Ridge(alpha=alpha))
    for leaf in [3,10]: candidates[f'forest_{leaf}']=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),RandomForestRegressor(n_estimators=200,max_depth=8,min_samples_leaf=leaf,random_state=42,n_jobs=2))
    validation=[]
    for name,m in candidates.items():
        m.fit(train[features],train.value_ugm3)
        z=metric(val.value_ugm3,np.maximum(0,m.predict(val[features])));z.update(model=name);validation.append(z)
    if forecast:
        for name,col in [('persistence','lag1'),('seasonal_naive7','lag7')]: validation.append(dict(model=name,**metric(val.value_ugm3,val[col])))
    else:validation.append(dict(model='mean',**metric(val.value_ugm3,np.repeat(train.value_ugm3.mean(),len(val)))))
    vt=pd.DataFrame(validation);vt.to_csv(case/'results'/'validation_metrics.csv',index=False)
    selected={fam:vt[vt.model.str.startswith(fam)].sort_values(['MAE','model']).iloc[0].model for fam in ['ridge','forest']}
    winner=vt.sort_values(['MAE','model']).iloc[0].model
    dev=pd.concat([train,val]);pred=test[['date','value_ugm3']].rename(columns={'date':'target_date','value_ugm3':'actual'}).reset_index(drop=True)
    if forecast:
        pred.insert(0,'forecast_origin',pred.target_date.dt.strftime('%Y-%m-%d')+'T00:00:00+05:00')
        pred['latest_feature_date']=pred.target_date-pd.Timedelta(days=1)
        pred['persistence']=test.lag1.to_numpy();pred['seasonal_naive7']=test.lag7.to_numpy()
    else:pred['mean']=dev.value_ugm3.mean()
    fitted={};trainlog=[]
    for fam,name in selected.items():
        m=candidates[name];t=time.perf_counter();m.fit(dev[features],dev.value_ugm3);elapsed=time.perf_counter()-t
        pred[fam]=np.maximum(0,m.predict(test[features]));joblib.dump(m,case/'models'/f'{fam}.joblib');fitted[fam]=m
        trainlog.append({'model':fam,'selected_configuration':name,'fit_rows':len(dev),'fit_seconds':elapsed,'parameters':{k:repr(v) for k,v in m.get_params().items()}})
    pcols=['persistence','seasonal_naive7','ridge','forest'] if forecast else ['mean','ridge','forest']
    mets=[]
    for col in pcols:mets.append(dict(model=col,**metric(pred.actual,pred[col])))
    mt=pd.DataFrame(mets);mt.to_csv(case/'results'/'metrics.csv',index=False);js(case/'results'/'metrics.json',mets)
    pred.to_csv(case/'results'/'predictions.csv',index=False)
    selected_output='ridge' if str(winner).startswith('ridge') else 'forest' if str(winner).startswith('forest') else winner
    js(case/'models'/'baseline.json',{'kind':'persistence' if forecast else 'mean','training_mean_ugm3':float(dev.value_ugm3.mean()),'definition':'prediction(t)=observed_daily(t-1)' if forecast else 'prediction(t)=mean(training+validation targets)','chosen_prediction_column':selected_output})
    js(case/'models'/'metadata.json',{'dataset_file':'results/model_dataset.csv','dataset_file_base':'case_directory','dataset_sha256':sha(case/'results'/'model_dataset.csv'),'target':'station32 daily PM2.5 ug/m3; >=18 unique hours','horizon':'next calendar day, origin at target day 00:00+05' if forecast else 'contemporaneous explanation; not a forecast','features':features,'selected_on_validation_MAE':winner,'selected_prediction_column':selected_output,'split':splits,'seed':42,'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__},'training':trainlog,'metrics':'../results/metrics.json','upstream_QC_caveat':'Retrospective publisher filtering and measurement availability latency not reconstructable; offline benchmark only.'})
    # Permutation sensitivity for tree model, diagnostic only, never model selection.
    pi=permutation_importance(fitted['forest'],test[features],test.value_ugm3,n_repeats=10,random_state=42,scoring='neg_mean_absolute_error',n_jobs=2)
    importance=pd.DataFrame({'feature':features,'MAE_increase_mean':pi.importances_mean,'MAE_increase_std':pi.importances_std}).sort_values('MAE_increase_mean',ascending=False)
    importance.to_csv(case/'results'/'permutation_importance.csv',index=False)
    errors=pred.assign(residual=pred[selected_output]-pred.actual,absolute_error=(pred[selected_output]-pred.actual).abs())
    errors.sort_values('absolute_error',ascending=False).head(15).to_csv(case/'results'/'largest_errors.csv',index=False)
    seasonal=[]
    for season,g in errors.groupby(errors.target_date.dt.month.map(SEASONS)):
        for col in pcols:seasonal.append(dict(season=season,model=col,**metric(g.actual,g[col])))
    pd.DataFrame(seasonal).to_csv(case/'results'/'seasonal_metrics.csv',index=False)
    peak_threshold=float(dev.value_ugm3.quantile(.9));pk=[]
    for label,mask in [('peak_actual_above_trainval_p90',pred.actual>peak_threshold),('other_actual',pred.actual<=peak_threshold)]:
        for col in pcols:
            if mask.sum()>1:pk.append(dict(group=label,model=col,threshold=peak_threshold,**metric(pred.loc[mask,'actual'],pred.loc[mask,col])))
    pd.DataFrame(pk).to_csv(case/'results'/'peak_metrics.csv',index=False)
    checks=[{'check':'chronological_nonoverlap','status':'PASS'},{'check':'no_imputed_target','status':'PASS'},{'check':'identical_test_targets_all_models','status':'PASS'},{'check':'metric_recalculation','status':'PASS'},{'check':'publisher_realtime_availability','status':'NOT_VERIFIABLE'},{'check':'station32_PM_fraction_physical_consistency','status':'FAIL'}]
    disk=pd.read_csv(case/'results'/'predictions.csv')
    for row in mets:assert np.isclose(metric(disk.actual,disk[row['model']])['MAE'],row['MAE'])
    if forecast:
        assert (pred.latest_feature_date<pred.target_date).all(); checks.append({'check':'all_model_features_strictly_before_target_start','status':'PASS'})
    pd.DataFrame(checks).to_csv(case/'results'/'quality_checks.csv',index=False)
    plt.figure(figsize=(9,4));x=np.arange(len(mt));plt.bar(x-.18,mt.MAE,.36,label='MAE');plt.bar(x+.18,mt.RMSE,.36,label='RMSE',hatch='//');plt.xticks(x,[MODEL_LABELS[m] for m in mt.model]);plt.ylabel('мкг/м³');plt.legend();plt.title('Отложенный 2025 год: одинаковые целевые сутки');savefig(case,'model_comparison.png')
    graph=pred.set_index('target_date').reindex(pd.date_range(pred.target_date.min(),pred.target_date.max()))
    plt.figure(figsize=(10,4));plt.plot(graph.index,graph.actual,'-',lw=1,label='Наблюдение');plt.plot(graph.index,graph[selected_output],'--',lw=1,label='По валидации: '+MODEL_LABELS[selected_output]);plt.ylabel('PM2.5, мкг/м³');plt.legend();plt.title('Наблюдения и результаты модели; пропуски разрывают линии');savefig(case,'actual_predicted.png')
    fig,axs=plt.subplots(1,2,figsize=(10,4));axs[0].scatter(pred.actual,errors.residual,s=13,alpha=.55);axs[0].axhline(0,color='k',ls='--');axs[0].set(xlabel='Фактическая PM2.5, мкг/м³',ylabel='Модель − факт, мкг/м³');axs[1].hist(errors.residual,bins=25,edgecolor='white');axs[1].set(xlabel='Остаток, мкг/м³',ylabel='Число суток');savefig(case,'residuals.png')
    fig,ax=plt.subplots(figsize=(8,4.5));z=importance.iloc[:10].sort_values('MAE_increase_mean');ax.barh(z.feature,z.MAE_increase_mean,xerr=z.MAE_increase_std);ax.set_xlabel('Изменение MAE при перестановке, мкг/м³');ax.set_title('Random Forest: диагностическая важность, не причинность');savefig(case,'feature_importance.png')
    return mt,pred,splits,selected_output

def run_case1(prepared=None):
    all_daily,weather=prepared if prepared else prepare()
    ref=all_daily[all_daily.reference_station].copy();ref['year']=ref.date.dt.year
    cov=ref.groupby(['city','station_id','parameter','year']).agg(observed_days=('date','size'),valid_days=('valid_day','sum'),hours=('n_hours','sum')).reset_index()
    cov['calendar_days']=np.where(cov.year==2024,366,365);cov['valid_day_coverage']=cov.valid_days/cov.calendar_days;cov.to_csv(HERE/'results'/'coverage.csv',index=False)
    pm=ref[ref.parameter=='pm2_5'].pivot(index='date',columns='city',values='value_ugm3');common=pm.dropna();common.to_csv(HERE/'results'/'common_dates_pm25.csv')
    pair=all_daily[all_daily.valid_day].pivot(index=['city','station_id','date'],columns='parameter',values='value_ugm3').dropna(subset=['pm2_5','pm10'])
    pair['pm25_above_pm10']=pair.pm2_5>pair.pm10
    pair['pm25_above_pm10_by20pct']=pair.pm2_5>pair.pm10*1.2
    physical=pair.groupby(['city','station_id']).agg(paired_days=('pm25_above_pm10','size'),pm25_gt_pm10_days=('pm25_above_pm10','sum'),pm25_gt_pm10_fraction=('pm25_above_pm10','mean'),pm25_gt_1_2pm10_days=('pm25_above_pm10_by20pct','sum'))
    physical.to_csv(HERE/'results'/'cross_pollutant_quality.csv')
    ref[ref.valid_day].groupby(['city','parameter']).value_ugm3.agg(['count','mean','std','min','max']).to_csv(HERE/'results'/'pollutant_summary.csv')
    summary=common.agg(['count','mean','median','std','max']).T;summary['who15_days']=(common>15).sum();summary['who15_pct']=100*summary.who15_days/summary['count'];summary.to_csv(HERE/'results'/'city_comparison.csv')
    thresholds=[]
    for (city,param),g in ref[ref.valid_day].groupby(['city','parameter']):
        if param in WHO:
            thresholds.append({'city':city,'parameter':param,'threshold_ugm3':WHO[param],'reference':'WHO2021_24h','valid_days':len(g),'exceedance_days':int((g.value_ugm3>WHO[param]).sum()),'exceedance_pct':float(100*(g.value_ugm3>WHO[param]).mean()),'mean_ugm3':float(g.value_ugm3.mean())})
    threshold_df=pd.DataFrame(thresholds);threshold_df.to_csv(HERE/'results'/'threshold_comparison.csv',index=False)
    national=[]
    for (city,param),g in ref[ref.valid_day].groupby(['city','parameter']):
        if param in KZ:
            national.append({'city':city,'parameter':param,'threshold_ugm3':KZ[param],'reference':'KZ_DSM70_current_as_read_2026_09_25_24h','valid_days':len(g),'exceedance_days':int((g.value_ugm3>KZ[param]).sum()),'exceedance_pct':float(100*(g.value_ugm3>KZ[param]).mean())})
    pd.DataFrame(national).to_csv(HERE/'results'/'national_threshold_comparison.csv',index=False)
    # A fixed PM2.5 station avoids an artificial trend caused by changing station mix.
    aty=ref[(ref.city=='Atyrau')&(ref.parameter=='pm2_5')].merge(weather,on='date',how='left')
    aty=calendar(aty);aty['season']=aty.date.dt.month.map(SEASONS)
    aty.groupby(['year','season']).agg(n=('value_ugm3','count'),mean=('value_ugm3','mean'),median=('value_ugm3','median')).to_csv(HERE/'results'/'seasonal_summary.csv')
    fields=['temperature_2m_mean','relative_humidity_2m_mean','wind_speed_10m_mean','precipitation_sum']
    corr=aty[['value_ugm3']+fields].corr(method='spearman');corr.to_csv(HERE/'results'/'weather_spearman.csv')
    features=fields+['sin_doy','cos_doy','sin_week','cos_week']
    metrics,pred,splits,selected=fit_models(HERE,aty.dropna(subset=['value_ugm3']),features)
    info={'period':['2022-01-01','2025-12-31'],'common_days':len(common),'fixed_stations':STATIONS,'atyrau_valid_days':int(aty.value_ugm3.notna().sum()),'atyrau_missing_or_incomplete_days':1461-int(aty.value_ugm3.notna().sum()),'highest_pm25_station_city_in_archive':summary['mean'].idxmax(),'selected_model':selected,'status':'PARTIAL','material_limitation':'Channel/calibration metadata unavailable; Atyrau32 PM25 frequently exceeds paired PM10. Archive comparison cannot validate city exposure ranking.','atyrau_pm25_gt_pm10_days':int(physical.loc[('Atyrau','32'),'pm25_gt_pm10_days']),'atyrau_paired_days':int(physical.loc[('Atyrau','32'),'paired_days']),'city_results':summary.reset_index().to_dict(orient='records')}
    js(HERE/'results'/'summary.json',info)
    fig,axs=plt.subplots(3,1,figsize=(10,7),sharex=True)
    for ax,city in zip(axs,['Atyrau','Aktau','Aktobe']):ax.plot(pm.index,pm[city],lw=.6);ax.axhline(15,ls='--',color='darkred');ax.set_ylabel(CITIES[city]+'\nмкг/м³')
    axs[0].set_title('PM2.5 на фиксированных станциях; пропуски не заполнены');savefig(HERE,'timeseries_cities.png')
    fig,ax=plt.subplots(figsize=(7,4));s=summary.sort_values('mean');ax.bar([CITIES[x] for x in s.index],s['mean'],color=['#6094b8','#487a92','#235268']);ax.set_ylabel('Средняя PM2.5 по общим датам, мкг/м³');ax.set_title(f'Общие пригодные даты: {len(common)}\nЭто сравнение станций, не рейтинг всего Казахстана');savefig(HERE,'city_comparison.png')
    monthly=pm.resample('MS').mean();plt.figure(figsize=(10,4));
    for city,mark in zip(['Atyrau','Aktau','Aktobe'],['o','s','^']):plt.plot(monthly.index,monthly[city],marker=mark,ms=3,label=CITIES[city])
    plt.ylabel('Месячная средняя PM2.5, мкг/м³');plt.legend();plt.title('Средние по имеющимся пригодным суткам; покрытие меняется');savefig(HERE,'monthly_means.png')
    tab=aty.pivot_table(index=aty.date.dt.year,columns=aty.date.dt.month,values='value_ugm3',aggfunc='mean');fig,ax=plt.subplots(figsize=(9,3.5));im=ax.imshow(tab,aspect='auto',cmap='YlOrRd');ax.set(xticks=range(12),xticklabels=range(1,13),yticks=range(len(tab)),yticklabels=tab.index,xlabel='Месяц',ylabel='Год',title='Атырау, пост № 32: средняя PM2.5 по месяцам');fig.colorbar(im,ax=ax,label='мкг/м³');savefig(HERE,'seasonal_heatmap.png')
    cc=cov[cov.parameter=='pm2_5'].pivot(index='city',columns='year',values='valid_day_coverage');fig,ax=plt.subplots(figsize=(7,3));im=ax.imshow(cc,vmin=0,vmax=1,cmap='Blues');ax.set(xticks=range(4),xticklabels=cc.columns,yticks=range(len(cc)),yticklabels=[CITIES[x] for x in cc.index],title='Доля суток с ≥18 часовыми наблюдениями')
    for i in range(len(cc)):
        for j in range(4):ax.text(j,i,f'{cc.iloc[i,j]:.0%}',ha='center',va='center',color='white' if cc.iloc[i,j]>.6 else 'black')
    fig.colorbar(im,ax=ax);savefig(HERE,'coverage.png')
    fig,axs=plt.subplots(1,2,figsize=(10,4));axs[0].scatter(aty.temperature_2m_mean,aty.value_ugm3,s=8,alpha=.3);axs[0].set(xlabel='ERA5: температура, °C',ylabel='PM2.5, мкг/м³');axs[1].scatter(aty.wind_speed_10m_mean,aty.value_ugm3,s=8,alpha=.3);axs[1].set(xlabel='ERA5: ветер, м/с',ylabel='PM2.5, мкг/м³');savefig(HERE,'weather_associations.png')
    th=threshold_df[threshold_df.city=='Atyrau'];fig,ax=plt.subplots(figsize=(8,4));ax.bar([POLLUTANT_LABELS[x] for x in th.parameter],th.exceedance_pct);ax.set(ylabel='Доля пригодных суток выше ориентира ВОЗ, %',title='Атырау, пост № 32: отдельный знаменатель для каждого вещества');savefig(HERE,'pollutant_exceedance.png')
    return info

def run_case2(prepared=None):
    setup()
    if prepared:all_daily=prepared[0]
    elif (HERE/'dataset.csv').exists():all_daily=pd.read_csv(HERE/'dataset.csv',parse_dates=['date'],dtype={'station_id':str},float_precision='round_trip')
    else:all_daily,_=prepare()
    a=all_daily[(all_daily.city=='Atyrau')&(all_daily.station_id=='32')&(all_daily.parameter=='pm2_5')].set_index('date').value_ugm3
    a=a.reindex(pd.date_range('2022-01-01','2025-12-31'))
    df=pd.DataFrame({'date':a.index,'value_ugm3':a.to_numpy()})
    for lag in [1,2,3,7,14]:df['lag'+str(lag)]=a.shift(lag).to_numpy()
    for w in [3,7,14]:df['rolling_mean'+str(w)]=a.shift(1).rolling(w,min_periods=max(2,w//2)).mean().to_numpy()
    df['rolling_std7']=a.shift(1).rolling(7,min_periods=3).std().to_numpy();df=calendar(df)
    features=[x for x in df if x not in ['date','value_ugm3']]
    before=len(df);valid_target=int(df.value_ugm3.notna().sum())
    # Persistence and seasonal-naive need genuine yesterday and last-week observations.
    df=df.dropna(subset=['value_ugm3','lag1','lag7']).copy()
    assert np.allclose(df.lag1,a.reindex(df.date-pd.Timedelta(days=1)).to_numpy())
    df['latest_feature_date']=df.date-pd.Timedelta(days=1)
    metrics,pred,splits,selected=fit_models(CASE2,df,features,True)
    m=metrics.set_index('model');improve=100*(m.loc['persistence','MAE']-m.loc[selected,'MAE'])/m.loc['persistence','MAE']
    info={'station_id':'32','station_name':'PCP #8','city':'Atyrau','calendar_days':before,'valid_target_days':valid_target,'modeled_days':len(df),'excluded_missing_lag1_or_lag7':valid_target-len(df),'selected_model':selected,'MAE_improvement_vs_persistence_pct':float(improve),'test_n':len(pred),'shared_source_case':'01_Air_Pollution','status':'PARTIAL','material_limitation':'Calibration/channel consistency unresolved; retrospective provider QC and real-time latency unknown; offline forecasting experiment only.'}
    js(CASE2/'results'/'summary.json',info)
    run_sensitivity235(all_daily)
    fig,ax=plt.subplots(figsize=(10,4));ax.plot(a.index,a,lw=.7,color='#35667c');ax.axvline(pd.Timestamp('2024-01-01'),color='k',ls='--');ax.axvline(pd.Timestamp('2025-01-01'),color='k',ls='--');ax.axvspan(pd.Timestamp('2025-01-01'),pd.Timestamp('2025-12-31'),alpha=.12,color='orange');ax.set(ylabel='PM2.5, мкг/м³',title='Обучение 2022–2023 / валидация 2024 / проверка 2025');savefig(CASE2,'temporal_split.png')
    fig,ax=plt.subplots(figsize=(8,3.5));pd.Series({k:int(v['n']) for k,v in splits.items()}).reindex(['train','validation','test']).plot.bar(ax=ax,rot=0);ax.set(ylabel='Число пригодных прогнозов',title='Целевые метки и необходимые baseline-лаги доступны');savefig(CASE2,'split_counts.png')
    return info

def run_sensitivity235(all_daily):
    """Exploratory secondary channel; hyperparameters frozen from station32, no tuning."""
    dest=CASE2/'results'/'sensitivity235';dest.mkdir(exist_ok=True)
    modeldir=CASE2/'models'/'sensitivity235';modeldir.mkdir(exist_ok=True)
    a=all_daily[(all_daily.city=='Atyrau')&(all_daily.station_id=='235')&(all_daily.parameter=='pm2_5')].set_index('date').value_ugm3
    a=a.reindex(pd.date_range('2022-01-01','2025-12-31'))
    df=pd.DataFrame({'date':a.index,'value_ugm3':a.to_numpy()})
    for lag in [1,2,3,7,14]:df['lag'+str(lag)]=a.shift(lag).to_numpy()
    for w in [3,7,14]:df['rolling_mean'+str(w)]=a.shift(1).rolling(w,min_periods=max(2,w//2)).mean().to_numpy()
    df['rolling_std7']=a.shift(1).rolling(7,min_periods=3).std().to_numpy();df=calendar(df)
    df=df.dropna(subset=['value_ugm3','lag1','lag7']).copy()
    features=json.loads((CASE2/'models'/'metadata.json').read_text(encoding='utf8'))['features']
    df['split']=np.where(df.date<'2024-01-01','train',np.where(df.date<'2025-01-01','validation','test'))
    df.to_csv(dest/'dataset.csv',index=False)
    dev=df[df.date<'2025-01-01'];test=df[df.date>='2025-01-01'];assert dev.date.max()<test.date.min()
    # Deserialize only locally created known models; this copies chosen parameters, never their fitted state.
    from sklearn.base import clone
    fixed={name:clone(joblib.load(CASE2/'models'/f'{name}.joblib')) for name in ['ridge','forest']}
    pred=test[['date','value_ugm3','lag1','lag7']].rename(columns={'date':'target_date','value_ugm3':'actual','lag1':'persistence','lag7':'seasonal_naive7'}).reset_index(drop=True)
    pred.insert(0,'forecast_origin',pred.target_date.dt.strftime('%Y-%m-%d')+'T00:00:00+05:00');pred['latest_feature_date']=pred.target_date-pd.Timedelta(days=1)
    for name,m in fixed.items():m.fit(dev[features],dev.value_ugm3);pred[name]=np.maximum(0,m.predict(test[features]));joblib.dump(m,modeldir/f'{name}.joblib')
    metrics=pd.DataFrame([dict(model=col,**metric(pred.actual,pred[col])) for col in ['persistence','seasonal_naive7','ridge','forest']]);metrics.to_csv(dest/'metrics.csv',index=False);pred.to_csv(dest/'predictions.csv',index=False)
    meta={'station_id':'235','station_name':'PCP #10','coordinates':[47.114687,51.860907],'status':'EXPLORATORY_SECONDARY','reason':'Added after observing station32 cross-channel inconsistency; main experiment kept unchanged.','hyperparameters':'Cloned unfitted pipelines selected on station32 validation; no station235 tuning','model_selection':'persistence frozen from main experiment; no selection by station235 test','fit_rows':len(dev),'test_rows':len(test),'fit_start':str(dev.date.min().date()),'fit_end':str(dev.date.max().date()),'test_start':str(test.date.min().date()),'test_end':str(test.date.max().date()),'features':features,'dataset_file':'results/sensitivity235/dataset.csv','dataset_file_base':'case_directory','dataset_sha256':sha(dest/'dataset.csv'),'test_mean':float(test.value_ugm3.mean()),'test_std':float(test.value_ugm3.std()),'limitations':'Physically more consistent PM channels do not independently validate calibration; retrospective QC and publication-latency limitations persist.'}
    js(dest/'metadata.json',meta);js(modeldir/'metadata.json',meta)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['1','10','both'],default='both');parser.add_argument('--verify',action='store_true');a=parser.parse_args()
    if a.verify:
        lock=json.loads((HERE/'data_lock.json').read_text(encoding='utf8'))
        for relative,expected in lock['archive_sha256'].items():assert sha(HERE/relative)==expected
        print('Pinned raw archive checksums PASS')
        for case in [HERE,CASE2]:
            p=pd.read_csv(case/'results'/'predictions.csv');m=pd.read_csv(case/'results'/'metrics.csv')
            for _,r in m.iterrows():
                z=metric(p.actual,p[r.model]);assert np.allclose([z['MAE'],z['RMSE'],z['R2']],[r.MAE,r.RMSE,r.R2])
            print(case.name,'metric verification PASS')
        sd=CASE2/'results'/'sensitivity235';p=pd.read_csv(sd/'predictions.csv');m=pd.read_csv(sd/'metrics.csv')
        for _,r in m.iterrows():
            z=metric(p.actual,p[r.model]);assert np.allclose([z['MAE'],z['RMSE'],z['R2']],[r.MAE,r.RMSE,r.R2])
        print('Secondary station235 metrics PASS')
        return
    prepared=prepare() if a.case in ['1','both'] else None
    if a.case in ['1','both']:print(json.dumps(run_case1(prepared),ensure_ascii=False,indent=2))
    if a.case in ['10','both']:print(json.dumps(run_case2(prepared),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
