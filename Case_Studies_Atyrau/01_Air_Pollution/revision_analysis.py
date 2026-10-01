"""Post-hoc robustness checks, 2026-09-26. Never fits or selects a model.

Inputs are frozen selected hourly archives and saved out-of-sample predictions.
Missing calendar days stay missing inside resampled blocks. This is an exploratory
resampling sensitivity analysis, not a validated interval for future performance.
"""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CASE2 = HERE.parent / '02_Air_Quality_Prediction'
SEED, REPEATS = 42, 5000

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path, **kwargs):
    return pd.read_csv(path, float_precision='round_trip', **kwargs)

def run():
    inputs = [HERE/'raw'/f'selected_{p}.csv.gz' for p in ('pm2_5','pm10')]
    inputs += [HERE/'dataset.csv', HERE/'results/predictions.csv', CASE2/'results/predictions.csv', CASE2/'results/sensitivity235/predictions.csv']
    initial = {str(p.relative_to(HERE.parent)):sha(p) for p in inputs}
    parts = []
    for p in ('pm2_5','pm10'):
        d = read(HERE/'raw'/f'selected_{p}.csv.gz', dtype={'station_id':str})
        assert not d.duplicated(['city','station_id','time_utc']).any()
        parts.append(d[['city','station_id','time_utc','date','value_ugm3']].rename(columns={'value_ugm3':p}))
    pair = parts[0].merge(parts[1], on=['city','station_id','time_utc','date'], validate='one_to_one')
    pair['gt'] = pair.pm2_5 > pair.pm10
    pair['gt20'] = pair.pm2_5 > 1.2*pair.pm10
    hourly = pair.groupby(['city','station_id']).agg(paired_hours=('gt','size'),gt_hours=('gt','sum'),gt20_hours=('gt20','sum')).reset_index()
    hourly['gt_pct'] = hourly.gt_hours/hourly.paired_hours*100
    hourly.to_csv(HERE/'results/paired_hour_quality.csv',index=False)
    daily = pair.groupby(['city','station_id','date']).agg(shared_hours=('pm2_5','size'),pm25=('pm2_5','mean'),pm10=('pm10','mean')).reset_index()
    daily = daily[daily.shared_hours>=18].copy()
    daily['gt'] = daily.pm25>daily.pm10
    daily['gt20'] = daily.pm25>daily.pm10*1.2
    quality = daily.groupby(['city','station_id']).agg(paired_days=('gt','size'),pm25_gt_pm10_days=('gt','sum'),pm25_gt_1_2pm10_days=('gt20','sum')).reset_index()
    quality['gt_pct'] = 100*quality.pm25_gt_pm10_days/quality.paired_days
    quality.to_csv(HERE/'results/shared_hours_daily_quality.csv',index=False)
    daily.to_csv(HERE/'results/shared_hours_daily.csv',index=False)

    all_daily=read(HERE/'dataset.csv',dtype={'station_id':str},parse_dates=['date'])
    selected=all_daily[(all_daily.parameter=='pm2_5')&all_daily.reference_station]
    sensitivity=[]
    for minimum in [18,24]:
        common=selected[selected.n_hours>=minimum].pivot(index='date',columns='city',values='mean_ugm3').dropna()
        for city in common:
            sensitivity.append(dict(min_hours=minimum,city=city,common_days=len(common),mean_ugm3=common[city].mean(),median_ugm3=common[city].median()))
    pd.DataFrame(sensitivity).to_csv(HERE/'results/completeness_sensitivity.csv',index=False)

    configs=[('case1_station32',HERE/'results/predictions.csv','mean'),('case10_station32',CASE2/'results/predictions.csv','persistence'),('case10_station235',CASE2/'results/sensitivity235/predictions.csv','persistence')]
    outputs=[]
    for label,path,baseline in configs:
        pred=read(path,parse_dates=['target_date']).set_index('target_date')
        assert not pred.index.duplicated().any()
        calendar=pred.reindex(pd.date_range('2025-01-01','2025-12-31'))
        for model in ['ridge','forest']:
            delta=(calendar[model]-calendar.actual).abs()-(calendar[baseline]-calendar.actual).abs()
            values=delta.to_numpy(); n=len(values)
            for block in [7,14,28]:
                # Same deterministic draw indices across paired model comparisons.
                rng=np.random.default_rng(SEED)
                starts=rng.integers(0,n-block+1,size=(REPEATS,int(np.ceil(n/block))))
                index=(starts[:,:,None]+np.arange(block)).reshape(REPEATS,-1)[:,:n]
                sample=values[index]
                means=np.nansum(sample,axis=1)/np.isfinite(sample).sum(axis=1)
                lo,hi=np.quantile(means,[.025,.975])
                outputs.append(dict(experiment=label,model=model,baseline=baseline,block_days=block,n=int(delta.notna().sum()),delta_MAE=float(delta.mean()),lower95=float(lo),upper95=float(hi),bootstrap_repeats=REPEATS))
    out=pd.DataFrame(outputs)
    out.to_csv(CASE2/'results/paired_error_block_sensitivity.csv',index=False)
    out[out.experiment=='case1_station32'].to_csv(HERE/'results/paired_error_block_sensitivity.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),sharex=False)
    for ax,label,title in zip(axes,['case10_station32','case10_station235'],['Основной пост № 32','Дополнительный пост № 235']):
        subset=out[(out.experiment==label)&(out.block_days==14)]
        y=np.arange(len(subset))
        ax.errorbar(subset.delta_MAE,y,xerr=[subset.delta_MAE-subset.lower95,subset.upper95-subset.delta_MAE],fmt='o',capsize=5,color='#176a80')
        ax.axvline(0,color='#8b3a3a',ls='--',lw=1)
        ax.set(yticks=y,yticklabels=['Ridge','Случайный лес'],title=title,xlabel='Разница MAE относительно «вчера», мкг/м³')
        ax.grid(axis='x',alpha=.2)
    fig.suptitle('Дополнительная проверка: 95% диапазоны блочного пересэмплирования\nБлок 14 суток; отрицательная разница означает меньшую ошибку',fontsize=11)
    fig.tight_layout();fig.savefig(CASE2/'figures/error_difference_intervals.png',dpi=180,bbox_inches='tight');plt.close(fig)
    assert initial=={str(p.relative_to(HERE.parent)):sha(p) for p in inputs}
    meta={'status':'POST_HOC_EXPLORATORY','date':'2026-09-26','seed':SEED,'resamples':REPEATS,'block_days':[7,14,28],'method':'Non-circular moving calendar blocks sampled with replacement; concatenate and truncate to 365 calendar days; retain missing dates; paired absolute-loss difference; percentile 2.5% and 97.5%.','limitations':'No model refitting or tuning. Approximate resampling ranges conditional on frozen fits and observed 2025 data; weak-dependence/stationarity assumptions and informative missingness unverified. Not forecast prediction intervals or confirmatory p-values.','inputs_sha256':initial,'source':'Künsch (1989), Annals of Statistics 17(3), 1217–1241. DOI:10.1214/aos/1176347265'}
    (HERE/'results/revision_analysis_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
    print(quality.to_string(index=False));print(out[out.block_days==14].to_string(index=False))

if __name__=='__main__':run()
