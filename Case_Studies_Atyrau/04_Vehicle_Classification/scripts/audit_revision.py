"""Re-audit source annotations and derive descriptive diagnostics; no fitting/tuning."""
from pathlib import Path
import sys,json,hashlib,datetime
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix,classification_report

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pipeline import phash,CLASSES,SEED

def main():
    df=pd.read_csv(ROOT/'dataset.csv');pred=pd.read_csv(ROOT/'results/test_predictions.csv')
    source=json.loads((ROOT/'raw/instances_val2017.json').read_text(encoding='utf-8'))
    images={x['id']:x for x in source['images']};licenses={x['id']:x for x in source['licenses']}
    annotations={x['id']:x for x in source['annotations']}
    candidates=[a for a in source['annotations'] if a['category_id'] in [3,6,8]]
    funnel=[]
    def stage(name,items):funnel.append({'Этап':name,'Объекты':len(items),'Фотографии':len({a['image_id'] for a in items})})
    stage('Исходные car, bus, truck',candidates)
    licensed=[a for a in candidates if images[a['image_id']]['license'] in [1,2,4,5]];stage('Лицензия BY / BY-SA / BY-NC / BY-NC-SA',licensed)
    noncrowd=[a for a in licensed if a['iscrowd']==0];stage('Исключены crowd-объекты',noncrowd)
    selected=[a for a in noncrowd if a['bbox'][2]>=40 and a['bbox'][3]>=40];stage('Обе стороны рамки не меньше 40 пикселей',selected)
    pd.DataFrame(funnel).to_csv(ROOT/'results/selection_funnel.csv',index=False,encoding='utf-8-sig')
    checks={'all_eligible_annotations_retained':set(df.object_id)=={a['id'] for a in selected},'index_unique':bool(df.object_id.is_unique)}
    labelmap={3:'car',6:'bus',8:'truck'}
    index_agrees=[]
    for _,row in df.iterrows():
        annotation=annotations[int(row.object_id)]
        index_agrees.append(row['class']==labelmap[annotation['category_id']] and row.image_id==annotation['image_id'] and row.category_id==annotation['category_id'] and np.allclose([row.bbox_x,row.bbox_y,row.bbox_width,row.bbox_height],annotation['bbox'],rtol=0,atol=1e-9))
    checks['labels_and_boxes_match_original_annotations']=all(index_agrees)
    checks['license_fields_match_original_metadata']=all(r.license_id==images[r.image_id]['license'] and r.license_name==licenses[r.license_id]['name'] and r.license_url==licenses[r.license_id]['url'] for r in df.itertuples())
    checks['source_urls_match_original_metadata']=all(r.coco_url==images[r.image_id]['coco_url'] and r.flickr_url==images[r.image_id]['flickr_url'] for r in df.itertuples())
    crop_equal=[]
    for row in df.itertuples():
        im=Image.open(ROOT/row.image_path).convert('RGB');x,y,w,h=annotations[row.object_id]['bbox']
        bounds=(max(0,int(x)),max(0,int(y)),min(im.width,int(x+w+.999)),min(im.height,int(y+h+.999)))
        crop_equal.append(np.array_equal(np.asarray(im.crop(bounds)),np.asarray(Image.open(ROOT/row.crop_path))))
    checks['all_crops_equal_reconstruction_from_raw']=all(crop_equal)
    ids=sorted(df.image_id.unique());hashes={i:phash(ROOT/df.loc[df.image_id==i,'image_path'].iloc[0]) for i in ids}
    near=[(a,b) for k,a in enumerate(ids) for b in ids[k+1:] if (int(hashes[a],16)^int(hashes[b],16)).bit_count()<=6]
    split_by_image=df.groupby('image_id')['split'].first().to_dict()
    checks['no_near_source_image_cross_split']=all(split_by_image[a]==split_by_image[b] for a,b in near)
    crop_hashes=pd.DataFrame({'phash':[phash(ROOT/p) for p in df.crop_path],'split':df.split})
    checks['no_identical_crop_phash_cross_split']=bool(crop_hashes.groupby('phash').split.nunique().max()==1)
    folds=np.zeros(len(df),dtype=int)
    for n,(_,idx) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(df,df['class'],groups=df.scene_group)):folds[idx]=n
    expected=np.where(folds==0,'test',np.where(folds==1,'validation','train'))
    checks['split_replays_from_fixed_groups_and_seed']=np.array_equal(expected,df.split.to_numpy())
    attribution=pd.read_csv(ROOT/'results/image_attribution.csv')
    checks['attribution_covers_all_objects']=set(attribution.object_id)==set(df.object_id)
    checks['attribution_missing_creator_is_explicit']=bool(attribution.creator.str.contains('not supplied',regex=False).all())
    lic=df.groupby(['license_id','license_name'],sort=True).agg(objects=('object_id','size'),images=('image_id','nunique')).reset_index()
    lic.to_csv(ROOT/'results/license_counts.csv',index=False,encoding='utf-8-sig')
    accounting=[]
    for model in ['majority','logistic','random_forest']:
        cm=confusion_matrix(pred.actual,pred[model],labels=CLASSES)
        for i,c in enumerate(CLASSES):
            accounting.append({'model':model,'class':c,'correct':int(cm[i,i]),'actual_total':int(cm[i,:].sum()),'predicted_total':int(cm[:,i].sum()),'false_negative':int(cm[i,:].sum()-cm[i,i]),'false_positive':int(cm[:,i].sum()-cm[i,i]),'recall':cm[i,i]/cm[i,:].sum()})
    pd.DataFrame(accounting).to_csv(ROOT/'results/error_accounting.csv',index=False)
    metrics=json.loads((ROOT/'results/metrics.json').read_text(encoding='utf-8'));name=metrics['selected_model']
    pairs=pred.loc[pred.actual!=pred[name]].groupby(['actual',name]).size().reset_index(name='count').rename(columns={name:'predicted'}).sort_values(['count','actual','predicted'],ascending=[False,True,True])
    pairs.to_csv(ROOT/'results/error_pairs.csv',index=False)
    original_cm=pd.read_csv(ROOT/'results/confusion_matrix.csv',index_col=0)
    checks['confusion_matrix_recalculated']=np.array_equal(original_cm.loc[CLASSES,CLASSES],confusion_matrix(pred.actual,pred[name],labels=CLASSES))
    metadata=json.loads((ROOT/'models/metadata.json').read_text(encoding='utf-8'));history=pd.read_csv(ROOT/'results/training_history.csv')
    for family in ['logistic','random_forest']:
        rows=history.loc[history.model==family];best=rows.loc[rows.validation_macro_f1.idxmax()]
        checks['saved_parameters_follow_validation_'+family]=json.loads(best.parameters)==metadata['parameters'][family]
    checks['no_new_hyperparameters_in_revision']=history.parameters.tolist()==['{"C": 0.01}','{"C": 0.1}','{"C": 1.0}','{"max_depth": 8, "min_samples_leaf": 2}','{"max_depth": null, "min_samples_leaf": 2}']
    out={'audit_date':'2026-09-26','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'raw_near_duplicate_pairs':len(near),'objects':len(df),'images':df.image_id.nunique(),'test_already_seen_before_revision':True,'new_model_selection_performed':False,'annotation_labels_human_reaudited':False,'annotation_sha256':hashlib.sha256((ROOT/'raw/instances_val2017.json').read_bytes()).hexdigest()}
    (ROOT/'results/revision_checks.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    assert all(checks.values()),checks
    print('Independent revision audit PASS:',len(checks),'checks')

if __name__=='__main__':main()
