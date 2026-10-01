"""Sequential replay of REAL observations, never a live sensor integration."""
from pathlib import Path
import json, numpy as np, pandas as pd, joblib
CASE=Path(__file__).resolve().parents[1]

def replay_observations(observations, threshold, selected, model=None):
    """Replay in supplied order; a forecast requires seven consecutive valid days."""
    from pipeline import make_features,predict
    observations=observations.copy()
    observations['level_cm']=pd.to_numeric(observations['level_cm'],errors='coerce')
    observations['date']=pd.to_datetime(observations['date'],errors='coerce')
    seen=[];log=[];previous=None
    for row in observations.itertuples():
        quality='valid' if np.isfinite(row.level_cm) else 'missing_or_invalid'
        valid_date=bool(pd.notna(row.date))
        if not valid_date:quality='invalid_date'
        consecutive=valid_date and previous is not None and pd.notna(previous.date) and (row.date-previous.date).days==1
        if quality=='valid' and previous is not None and not consecutive:quality='date_gap'
        valid_value=bool(np.isfinite(row.level_cm))
        if not consecutive or not valid_value or not valid_date:seen=[]
        delta=float(row.level_cm-previous.level_cm) if consecutive and valid_value and np.isfinite(previous.level_cm) else np.nan
        flag=bool(quality=='valid' and np.isfinite(delta) and abs(delta)>threshold)
        if valid_value and valid_date:seen.append({'date':row.date,'level_cm':row.level_cm})
        seen=seen[-7:]
        forecast=np.nan;target=row.date+pd.Timedelta(days=1)
        if quality=='valid' and len(seen)>=7:
            # Placeholder target used only to build shifted predictors; never supplied to model.
            scratch=pd.DataFrame(seen+[{'date':target,'level_cm':0.0}])
            last=make_features(scratch).iloc[[-1]]
            assert last.target_time.iloc[0]==target, 'Never reuse predictors from an earlier block'
            if selected=='persistence':forecast=float(last.level_lag1.iloc[0])
            elif selected=='drift':forecast=float((last.level_lag1+last.delta_lag1).iloc[0])
            else:forecast=float(predict(model,last)[0])
        action=('check_measurement_and_context_manually' if flag else 'routine_observation') if quality=='valid' else 'check_missing_or_invalid_record'
        log.append({'observation_date':row.date.date(),'level_cm':row.level_cm,'quality':quality,'delta_cm':delta,'threshold_abs_change_cm':threshold,'statistical_flag':flag,'target_time':target.date(),'prediction_next_day_cm':forecast,'proposed_action':action,'mode':'historical_replay_not_live_iot'})
        previous=row
    return pd.DataFrame(log)

def replay():
    df=pd.read_csv(CASE/'dataset.csv',parse_dates=['date'])
    history_train=df[df.date.dt.year.isin([2018,2019])]
    changes=history_train.level_cm.diff().where(history_train.date.diff().dt.days.eq(1))
    threshold=float(changes.abs().dropna().quantile(.99))
    meta=json.loads((CASE/'models/metadata.json').read_text(encoding='utf-8'))
    selected=meta['selected_model']
    model=joblib.load(CASE/f'models/{selected}.joblib') if selected in ['ridge','forest'] else None
    output=replay_observations(df[df.date.dt.year==2022],threshold,selected,model)
    output.to_csv(CASE/'results/replay_actions.csv',index=False)
    summary={'rows':len(output),'predictions':int(output.prediction_next_day_cm.notna().sum()),'statistical_flags':int(output.statistical_flag.sum()),'quality_failures':int((output.quality!='valid').sum()),'threshold_cm':threshold,'threshold_source':'99th percentile absolute daily change on 2018+2019 only','threshold_operator':'>','hazard_classification':False,'labeled_anomaly_metrics':'not applicable; no anomaly labels','actions_sent':False,'last_forecast_has_no_observed_target':True}
    (CASE/'results/replay_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    return summary

if __name__=='__main__':print(json.dumps(replay(),indent=2))
