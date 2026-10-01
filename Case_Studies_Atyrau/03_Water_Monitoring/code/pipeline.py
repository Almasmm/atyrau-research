"""Actual one-day water-level experiment. All paths relative to this case."""
from pathlib import Path
import argparse, json, hashlib, time, platform, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sklearn, joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CASE=Path(__file__).resolve().parents[1]
FEATURES=['level_lag1','delta_lag1','delta_lag2','delta_lag3','deviation_mean3','deviation_mean7','mean_change7','sin_doy','cos_doy']
SEED=42
NAMES={'persistence':'Последний уровень','drift':'Линейное продолжение','ridge':'Ridge','forest':'Random Forest'}

def dump(name,obj):
    p=CASE/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def metrics(y,p):
    return {'n':len(y),'MAE_cm':float(mean_absolute_error(y,p)),'RMSE_cm':float(np.sqrt(mean_squared_error(y,p))),'R2':float(r2_score(y,p))}

def make_features(df):
    """Target date index; every observed feature shifted strictly into the past."""
    blocks=df.date.diff().dt.days.ne(1).cumsum()
    parts=[]
    for _,d in df.groupby(blocks):
        d=d.copy(); s=d.level_cm
        f=pd.DataFrame(index=d.index)
        f['target_time']=d.date;f['forecast_origin_date']=d.date-pd.Timedelta(days=1)
        f['actual']=s;f['level_lag1']=s.shift(1)
        for lag in (1,2,3):f[f'delta_lag{lag}']=s.shift(lag)-s.shift(lag+1)
        f['deviation_mean3']=s.shift(1)-s.shift(1).rolling(3).mean()
        f['deviation_mean7']=s.shift(1)-s.shift(1).rolling(7).mean()
        f['mean_change7']=(s.shift(1)-s.shift(7))/6
        doy=d.date.dt.dayofyear
        f['sin_doy']=np.sin(2*np.pi*doy/365.25);f['cos_doy']=np.cos(2*np.pi*doy/365.25)
        f['delta_target']=s-s.shift(1)
        parts.append(f)
    out=pd.concat(parts).dropna().copy()
    out['split']=out.target_time.dt.year.map({2018:'train',2019:'validation',2022:'test'})
    return out.reset_index(drop=True)

def predict(model,frame):
    return frame.level_lag1.to_numpy()+model.predict(frame[FEATURES])

def savefig(name):
    plt.tight_layout();plt.savefig(CASE/'figures'/name,dpi=180,bbox_inches='tight');plt.close()

def run():
    start=time.time()
    for sub in ('results','models','figures'):(CASE/sub).mkdir(exist_ok=True)
    df=pd.read_csv(CASE/'dataset.csv',parse_dates=['date']);f=make_features(df)
    train=f[f.split=='train'];val=f[f.split=='validation'];test=f[f.split=='test']
    assert train.target_time.max()<val.target_time.min()<test.target_time.min()
    assert all((f.target_time-f.forecast_origin_date).dt.days==1)
    assert len(f)==1081 and len(train)==358 and len(val)==365 and len(test)==358
    assert df.date.is_monotonic_increasing and not df.date.duplicated().any()
    # Direct causal feature check at each target: exactly seven preceding calendar days.
    indexed=df.set_index('date').level_cm
    for row in f.itertuples():
        assert row.level_lag1==indexed.loc[row.forecast_origin_date]
        expected=indexed.reindex(pd.date_range(row.target_time-pd.Timedelta(days=7),row.forecast_origin_date))
        assert len(expected)==7 and expected.notna().all()
        assert abs(row.deviation_mean7-(expected.iloc[-1]-expected.mean()))<1e-10
    trials=[]; candidates={}
    for name,pred in [('persistence',val.level_lag1),('drift',val.level_lag1+val.delta_lag1)]:
        trials.append({'model':name,'parameter':'none',**metrics(val.actual,pred)})
    for alpha in (0.1,1,10,100):
        key=f'ridge_alpha_{alpha}'
        model=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
        model.fit(train[FEATURES],train.delta_target)
        trials.append({'model':'ridge','parameter':alpha,**metrics(val.actual,predict(model,val))})
        candidates[key]=model
    for leaf in (3,10):
        key=f'forest_leaf_{leaf}'
        model=RandomForestRegressor(n_estimators=200,max_depth=6,min_samples_leaf=leaf,random_state=SEED,n_jobs=2)
        model.fit(train[FEATURES],train.delta_target)
        trials.append({'model':'forest','parameter':leaf,**metrics(val.actual,predict(model,val))})
        candidates[key]=model
    trials=pd.DataFrame(trials);trials.to_csv(CASE/'results/validation_search.csv',index=False)
    best_ridge=trials[trials.model=='ridge'].sort_values('MAE_cm',kind='stable').iloc[0]
    best_forest=trials[trials.model=='forest'].sort_values('MAE_cm',kind='stable').iloc[0]
    selected=str(trials.sort_values('MAE_cm',kind='stable').iloc[0].model)
    frozen={'selected_model':selected,'ridge_alpha':float(best_ridge.parameter),'forest_min_samples_leaf':int(best_forest.parameter),'criterion':'lowest validation MAE; stable order breaks ties','test_year':2022,'seed':SEED}
    dump('results/frozen_selection.json',frozen)
    # Selection is frozen before any test prediction or metric computation.
    fit_data=f[f.split.isin(['train','validation'])]
    models={
        'ridge':make_pipeline(StandardScaler(),Ridge(alpha=float(best_ridge.parameter))),
        'forest':RandomForestRegressor(n_estimators=200,max_depth=6,min_samples_leaf=int(best_forest.parameter),random_state=SEED,n_jobs=2),
    }
    for name,model in models.items():
        model.fit(fit_data[FEATURES],fit_data.delta_target)
        joblib.dump(model,CASE/f'models/{name}.joblib')
    outputs=[];metric_rows=[]
    for split,frame in [('validation',val),('test',test)]:
        out=frame[['forecast_origin_date','target_time','actual','split']].copy()
        out['persistence']=frame.level_lag1;out['drift']=frame.level_lag1+frame.delta_lag1
        if split=='validation':
            out['ridge']=predict(candidates[f'ridge_alpha_{float(best_ridge.parameter):g}'],frame) if float(best_ridge.parameter)!=0.1 else predict(candidates['ridge_alpha_0.1'],frame)
            out['forest']=predict(candidates[f'forest_leaf_{int(best_forest.parameter)}'],frame)
        else:
            for name,model in models.items():out[name]=predict(model,frame)
        out['selected']=out[selected]
        for name in ('persistence','drift','ridge','forest'):
            metric_rows.append({'split':split,'model':name,**metrics(out.actual,out[name])})
        outputs.append(out)
    predictions=pd.concat(outputs).reset_index(drop=True)
    predictions.to_csv(CASE/'results/predictions.csv',index=False)
    m=pd.DataFrame(metric_rows);m.to_csv(CASE/'results/metrics.csv',index=False)
    dump('results/metrics.json',m.to_dict(orient='records'))
    f.to_csv(CASE/'results/features_and_split.csv',index=False)
    f[['target_time','forecast_origin_date','split']].to_csv(CASE/'results/split_manifest.csv',index=False)
    metadata={**frozen,'target':'next calendar day mean level, cm above gauge datum','station_id':19802,'datum_m_BS':-30.0,'features':FEATURES,'fit_rows':len(fit_data),'test_rows':len(test),'dataset_sha256':hashlib.sha256((CASE/'dataset.csv').read_bytes()).hexdigest(),'experiment_plan_sha256':hashlib.sha256((CASE/'EXPERIMENT_PLAN.md').read_bytes()).hexdigest(),'split':{name:{'n':len(g),'first_target':str(g.target_time.min().date()),'last_target':str(g.target_time.max().date())} for name,g in f.groupby('split')},'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikit-learn':sklearn.__version__,'matplotlib':matplotlib.__version__,'joblib':joblib.__version__},'hyperparameters':{'ridge':{'alpha':float(best_ridge.parameter),'scaler':'StandardScaler'},'forest':{'n_estimators':200,'max_depth':6,'min_samples_leaf':int(best_forest.parameter),'random_state':SEED,'n_jobs':2}},'metrics_path':'results/metrics.json'}
    dump('models/metadata.json',metadata)
    testp=predictions[predictions.split=='test'].copy()
    testp['month']=testp.target_time.dt.month
    seasons={12:'Зима',1:'Зима',2:'Зима',3:'Весна',4:'Весна',5:'Весна',6:'Лето',7:'Лето',8:'Лето',9:'Осень',10:'Осень',11:'Осень'}
    testp['season']=testp.month.map(seasons)
    seasonal=[]
    for season,g in testp.groupby('season'):
        for name in ('persistence',selected):seasonal.append({'season':season,'model':name,**metrics(g.actual,g[name])})
    pd.DataFrame(seasonal).to_csv(CASE/'results/seasonal_metrics.csv',index=False)
    testp['error_cm']=testp.actual-testp.selected
    testp['absolute_error_cm']=abs(testp.error_cm)
    testp.sort_values('absolute_error_cm',ascending=False).head(10).to_csv(CASE/'results/largest_errors.csv',index=False)
    annual=df.groupby('source_year').agg(n=('level_cm','size'),mean_cm=('level_cm','mean'),min_cm=('level_cm','min'),max_cm=('level_cm','max'),std_cm=('level_cm','std')).reset_index()
    annual.to_csv(CASE/'results/annual_summary.csv',index=False)
    monthly=df.assign(month=df.date.dt.month).groupby(['source_year','month']).level_cm.mean().unstack(0)
    monthly.to_csv(CASE/'results/monthly_means.csv')
    pd.DataFrame({'feature':FEATURES,'importance':models['forest'].feature_importances_}).sort_values('importance',ascending=False).to_csv(CASE/'results/forest_importance.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,1,figsize=(10,7),sharey=True)
    for ax,(year,g) in zip(axes,df.groupby('source_year')):
        ax.plot(g.date,g.level_cm,color='#146b83');ax.set_title(str(year),loc='left');ax.set_ylabel('Уровень, см');ax.grid(alpha=.2)
    fig.suptitle('Урал — Атырау: реальные среднесуточные уровни\nПропуск 2020–2021 гг. не интерполируется');savefig('01_daily_levels.png')
    plt.figure(figsize=(9,4.6))
    for i,year in enumerate(monthly.columns):plt.plot(monthly.index,monthly[year],marker=['o','s','^'][i],linestyle=['-','--',':'][i],label=str(year))
    plt.xticks(range(1,13));plt.xlabel('Месяц');plt.ylabel('Средний уровень, см');plt.title('Годовые различия и сезонный рисунок');plt.legend();plt.grid(alpha=.2);savefig('02_monthly_levels.png')
    plt.figure(figsize=(9,4.6))
    for i,(year,g) in enumerate(df.groupby('source_year')):plt.hist(g.level_cm.diff().dropna(),bins=np.arange(-70,91,5),histtype='step',linewidth=1.8,linestyle=['-','--',':'][i],label=str(year))
    plt.xlabel('Суточное изменение уровня, см');plt.ylabel('Количество дней');plt.title('Изменения уровня: редкие резкие скачки');plt.legend();savefig('03_daily_changes.png')
    fig,axes=plt.subplots(2,1,figsize=(10,7))
    for ax,part,title in [(axes[0],testp,'Отложенный 2022 год'),(axes[1],testp[testp.month.isin([3])],'Март 2022: крупные ошибки')]:
        ax.plot(part.target_time,part.actual,label='Наблюдение',color='#163a50');ax.plot(part.target_time,part.selected,label=NAMES[selected],linestyle='--',color='#c86528');ax.plot(part.target_time,part.persistence,label='Последний уровень',linestyle=':',color='#668871',alpha=.7);ax.set_title(title,loc='left');ax.set_ylabel('Уровень, см');ax.legend(fontsize=9,ncol=3);ax.grid(alpha=.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m' if len(part)<60 else '%m.%Y'))
    savefig('04_actual_vs_predicted.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5));tm=m[m.split=='test'].set_index('model')
    for ax,metric,title in zip(axes,['MAE_cm','RMSE_cm'],['MAE, см','RMSE, см']):
        bars=ax.bar(range(4),tm[metric],color=['#809aa7','#9cabb1','#146b83','#c86528']);ax.set_xticks(range(4),['Persistence','Drift','Ridge','RF']);ax.set_title(title);ax.bar_label(bars,fmt='%.2f')
    fig.suptitle('Одинаковые 358 целевых дней: сравнение test');savefig('05_model_comparison.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5));axes[0].scatter(testp.selected,testp.error_cm,s=13,alpha=.6,color='#146b83');axes[0].axhline(0,color='black',linestyle='--');axes[0].set_xlabel('Прогноз, см');axes[0].set_ylabel('Факт − прогноз, см');axes[0].set_title('Остатки на test')
    s=pd.DataFrame(seasonal); order=['Зима','Весна','Лето','Осень'];width=.36
    for idx,name in enumerate(dict.fromkeys(['persistence',selected])):
        values=s[s.model==name].drop_duplicates('season').set_index('season').reindex(order)
        axes[1].bar(np.arange(4)+(idx-.5)*width,values.MAE_cm,width,hatch=['','//'][idx],label=NAMES[name])
    axes[1].set_xticks(range(4),order);axes[1].set_ylabel('MAE, см');axes[1].set_title('Ошибка по сезонам 2022');axes[1].legend(fontsize=8);savefig('06_errors_seasons.png')
    base=float(tm.loc['persistence','MAE_cm']);chosen=float(tm.loc[selected,'MAE_cm'])
    summary={'selected_model':selected,'validation_selected_MAE_cm':float(m[(m.split=='validation')&(m.model==selected)].iloc[0].MAE_cm),'test_selected':metrics(testp.actual,testp.selected),'test_baseline':metrics(testp.actual,testp.persistence),'MAE_improvement_percent':100*(base-chosen)/base,'largest_error_date':str(testp.loc[testp.absolute_error_cm.idxmax(),'target_time'].date()),'largest_error_cm':float(testp.absolute_error_cm.max()),'annual':annual.to_dict(orient='records'),'elapsed_seconds':time.time()-start,'data_rows':len(df),'feature_rows':len(f),'training_search_fits':6,'final_refits':2}
    dump('results/summary.json',summary)
    from replay import replay
    replay()
    verify()
    return summary

def verify():
    checks=[]
    df=pd.read_csv(CASE/'dataset.csv',parse_dates=['date']);f=make_features(df)
    checks.append({'check':'schema_unique_dates_finite_numeric','status':'PASS' if not df.date.duplicated().any() and np.isfinite(df.level_cm).all() else 'FAIL'})
    checks.append({'check':'strictly_past_observations','status':'PASS' if (f.forecast_origin_date<f.target_time).all() else 'FAIL'})
    checks.append({'check':'target_split_disjoint','status':'PASS' if all(set(f.loc[f.split==a,'target_time']).isdisjoint(set(f.loc[f.split==b,'target_time'])) for a,b in [('train','validation'),('train','test'),('validation','test')]) else 'FAIL'})
    p=pd.read_csv(CASE/'results/predictions.csv');m=pd.read_csv(CASE/'results/metrics.csv')
    for row in m.itertuples():
        g=p[p.split==row.split];actual=metrics(g.actual,g[row.model])
        assert all(np.isclose(actual[k],getattr(row,k),atol=1e-10) for k in ['MAE_cm','RMSE_cm','R2'])
    checks.append({'check':'metrics_recomputed_from_saved_predictions','status':'PASS'})
    extraction=pd.read_csv(CASE/'results/extraction_validation.csv');assert extraction['pass'].all()
    checks.append({'check':'36_source_monthly_mean_checks','status':'PASS'})
    meta=json.loads((CASE/'models/metadata.json').read_text(encoding='utf-8'));assert meta['dataset_sha256']==hashlib.sha256((CASE/'dataset.csv').read_bytes()).hexdigest()
    checks.append({'check':'model_dataset_hash','status':'PASS'})
    selected=meta['selected_model'];replay_df=pd.read_csv(CASE/'results/replay_actions.csv')
    scored=replay_df[replay_df.prediction_next_day_cm.notna()].copy()
    check_p=p[p.split=='test'].copy();check_p['target_time']=pd.to_datetime(check_p.target_time)
    scored['target_time']=pd.to_datetime(scored.target_time)
    joined=check_p.merge(scored,on='target_time');assert len(joined)==358
    assert np.allclose(joined[selected],joined.prediction_next_day_cm)
    checks.append({'check':'sequential_replay_matches_batch_predictions','status':'PASS'})
    assert all(c['status']=='PASS' for c in checks)
    dump('results/checks.json',checks)
    return checks

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    print(json.dumps(verify() if args.verify_only else run(),ensure_ascii=True,indent=2))
