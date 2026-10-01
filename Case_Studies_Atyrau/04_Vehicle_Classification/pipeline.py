"""CPU vehicle-crop experiment. Run from any directory: python pipeline.py [--verify-only]."""
from pathlib import Path
import argparse, json, hashlib, time, sys, platform, importlib.metadata
import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
SEED=42; CLASSES=['bus','car','truck']
def dump(name,obj):
    (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def features(image):
    im=image.convert('RGB');ratio=np.log(max(im.width,1)/max(im.height,1))
    arr=np.asarray(im.resize((64,64),Image.Resampling.BILINEAR),dtype=np.float32)/255
    grad=hog(rgb2gray(arr),orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),block_norm='L2-Hys')
    hist=np.concatenate([np.histogram(arr[:,:,i],bins=16,range=(0,1),density=False)[0]/4096 for i in range(3)])
    return np.r_[grad,hist,ratio].astype(np.float32)
def phash(path):
    arr=np.asarray(Image.open(path).convert('L').resize((32,32)),dtype=float)
    vals=dctn(arr,type=2,norm='ortho')[:8,:8].ravel()[1:]
    return np.packbits(vals>np.median(vals)).tobytes().hex()
def metric(y,p):return {'macro_f1':float(f1_score(y,p,labels=CLASSES,average='macro',zero_division=0)),'accuracy':float(accuracy_score(y,p)),'n':int(len(y))}
def savefig(name):plt.tight_layout();plt.savefig(ROOT/'figures'/name,dpi=180,bbox_inches='tight');plt.close()

def split_data(df):
    ids=sorted(df.image_id.unique()); parent={i:i for i in ids}
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    hashes={i:phash(ROOT/df.loc[df.image_id==i,'image_path'].iloc[0]) for i in ids}
    pairs=[]
    for n,a in enumerate(ids):
        for b in ids[n+1:]:
            distance=(int(hashes[a],16)^int(hashes[b],16)).bit_count()
            if distance<=6:union(a,b);pairs.append({'image_a':a,'image_b':b,'phash_hamming':distance})
    # Full-image near hashes and identical crop hashes keep likely duplicates together.
    by_hash={}
    for row in df.itertuples():
        h=phash(ROOT/row.crop_path)
        if h in by_hash:union(row.image_id,by_hash[h])
        else:by_hash[h]=row.image_id
    df['scene_group']=df.image_id.map(lambda i:find(i));df['image_phash']=df.image_id.map(hashes)
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    folds=np.zeros(len(df),dtype=int)
    for fold,(_,idx) in enumerate(splitter.split(df,df['class'],groups=df.scene_group)):folds[idx]=fold
    df['split']=np.where(folds==0,'test',np.where(folds==1,'validation','train'))
    pd.DataFrame(pairs,columns=['image_a','image_b','phash_hamming']).to_csv(ROOT/'results/near_duplicate_groups.csv',index=False)
    dump('results/split_protocol.json',{'seed':SEED,'algorithm':'StratifiedGroupKFold(5,shuffle=True); fold0=test fold1=validation fold2-4=train','group':'source image plus connected full-image pHash Hamming<=6 or identical crop pHash','near_duplicate_pairs':len(pairs),'group_count':df.scene_group.nunique(),'claims_limit':'No trip or photographer IDs in COCO; grouping cannot prove all semantic scenes independent. Not adjacent video-frame splitting.','frozen_before_model_fitting':True})
    df.to_csv(ROOT/'dataset.csv',index=False,encoding='utf-8-sig')
    df[['object_id','image_id','scene_group','class','split','crop_path','crop_sha256','image_phash']].to_csv(ROOT/'results/split_manifest.csv',index=False)
    return df

def verify():
    df=pd.read_csv(ROOT/'dataset.csv');pred=pd.read_csv(ROOT/'results/test_predictions.csv');metrics=json.loads((ROOT/'results/metrics.json').read_text(encoding='utf-8'))
    checks={}
    checks['unique_object_ids']=bool(df.object_id.is_unique)
    checks['unique_test_prediction_ids']=bool(pred.object_id.is_unique)
    checks['all_three_classes']=set(df['class'])==set(CLASSES)
    checks['no_image_cross_split']=int(df.groupby('image_id')['split'].nunique().max())==1
    checks['no_scene_group_cross_split']=int(df.groupby('scene_group')['split'].nunique().max())==1
    checks['no_identical_crop_cross_split']=int(df.groupby('crop_sha256')['split'].nunique().max())==1
    checks['no_identical_source_cross_split']=int(df.groupby('image_sha256')['split'].nunique().max())==1
    checks['no_missing_required_fields']=not df[['object_id','image_id','class','crop_path','split']].isna().any().any()
    checks['all_crops_open_and_hash_match']=all((ROOT/r.crop_path).exists() and sha(ROOT/r.crop_path)==r.crop_sha256 for r in df.itertuples())
    pinned=json.loads((ROOT/'raw/source_integrity.json').read_text(encoding='utf-8'))['files']
    checks['pinned_raw_hashes_match']=all((ROOT/p).is_file() and sha(ROOT/p)==h for p,h in pinned.items() if p.endswith(('.json','.jpg')))
    checks['paths_relative_and_inside_case']=all(not Path(p).is_absolute() and (ROOT/str(p)).resolve().is_relative_to(ROOT) for col in ['crop_path','image_path'] for p in df[col])
    checks['same_test_for_all_models']=len(pred)==int((df.split=='test').sum()) and set(pred.object_id)==set(df.loc[df.split=='test','object_id'])
    for model,m in metrics['test'].items():
        recomputed=metric(pred.actual,pred[model])
        checks['recalculated_'+model]=all(abs(recomputed[k]-m[k])<1e-12 for k in ['macro_f1','accuracy','n'])
    checks['test_labels_match_index']=all(pred.set_index('object_id').actual==df.set_index('object_id').loc[pred.object_id,'class'])
    # Recompute from image inputs, not from features.npz or stored predictions.
    X=np.stack([features(Image.open(ROOT/p)) for p in df.crop_path]);tr=(df.split=='train').to_numpy()
    positions=df.reset_index().set_index('object_id')['index'].loc[pred.object_id].to_numpy()
    for name in metrics['test']:
        saved=joblib.load(ROOT/'models'/f'{name}.joblib')
        checks['saved_model_predicts_'+name]=np.array_equal(saved.predict(X[positions]),pred[name].to_numpy())
        if name=='logistic':
            checks['scaler_fitted_on_train_only']=bool(np.allclose(saved[0].mean_,X[tr].mean(axis=0,dtype=np.float64),rtol=1e-7,atol=1e-9)) and int(saved[0].n_samples_seen_)==int(tr.sum())
    metadata=json.loads((ROOT/'models/metadata.json').read_text(encoding='utf-8'))
    checks['model_dataset_hash_matches']=metadata['dataset_sha256']==sha(ROOT/'dataset.csv')
    checks['model_split_hash_matches']=metadata['split_sha256']==sha(ROOT/'results/split_manifest.csv')
    history=pd.read_csv(ROOT/'results/training_history.csv')
    checks['selection_uses_validation']=history.loc[history.validation_macro_f1.idxmax(),'model']==metrics['selected_model']
    dump('results/quality_checks.json',checks)
    assert all(checks.values()),checks
    print('Verification PASS:',len(checks),'checks')
    return checks

def main():
    started=time.perf_counter()
    for folder in ['results','models','figures','processed']:(ROOT/folder).mkdir(exist_ok=True)
    df=pd.read_csv(ROOT/'dataset.csv')
    if 'split' not in df:df=split_data(df)
    X=np.stack([features(Image.open(ROOT/p)) for p in df.crop_path]);y=df['class'].to_numpy()
    masks={s:(df.split==s).to_numpy() for s in ['train','validation','test']};tr=masks['train'];va=masks['validation'];te=masks['test']
    np.savez_compressed(ROOT/'processed/features.npz',X=X,object_id=df.object_id.to_numpy())
    candidates=[]
    for c in [.01,.1,1.0]:candidates.append(('logistic',{'C':c},make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=2000,class_weight='balanced',random_state=SEED))))
    for depth,leaf in [(8,2),(None,2)]:candidates.append(('random_forest',{'max_depth':depth,'min_samples_leaf':leaf},RandomForestClassifier(n_estimators=250,max_depth=depth,min_samples_leaf=leaf,class_weight='balanced_subsample',random_state=SEED,n_jobs=4)))
    history=[];best={}
    for family,params,model in candidates:
        t=time.perf_counter();model.fit(X[tr],y[tr]);a=metric(y[tr],model.predict(X[tr]));b=metric(y[va],model.predict(X[va]))
        row={'model':family,'parameters':json.dumps(params),'fit_seconds':time.perf_counter()-t,'train_n':int(tr.sum()),'validation_n':int(va.sum()),'train_macro_f1':a['macro_f1'],'validation_macro_f1':b['macro_f1'],'iterations':str(getattr(model[-1] if hasattr(model,'steps') else model,'n_iter_','not_applicable'))};history.append(row);print(row,flush=True)
        if family not in best or b['macro_f1']>best[family][0]:best[family]=(b['macro_f1'],model,params)
    selected=max(best,key=lambda k:best[k][0])
    models={'majority':DummyClassifier(strategy='most_frequent').fit(X[tr],y[tr]),**{k:v[1] for k,v in best.items()}}
    pd.DataFrame(history).to_csv(ROOT/'results/training_history.csv',index=False)
    out={s:df.loc[masks[s],['object_id','image_id','scene_group','class','crop_path','bbox_width','bbox_height']].rename(columns={'class':'actual'}).copy() for s in ['validation','test']}
    scores={s:{} for s in out}
    for s,pred in out.items():
        for name,model in models.items():pred[name]=model.predict(X[masks[s]]);scores[s][name]=metric(pred.actual,pred[name])
        pred.to_csv(ROOT/'results'/f'{s}_predictions.csv',index=False)
    m={'selected_model':selected,'selection_criterion':'validation macro-F1 only; models remain trained on training split only','classes':CLASSES,'seed':SEED,'validation':scores['validation'],'test':scores['test'],'counts':pd.crosstab(df.split,df['class']).to_dict(),'features':int(X.shape[1]),'feature_description':'64x64 RGB crop: HOG 9 orientations/8x8 cells/2x2 blocks L2-Hys + 16-bin RGB histogram each channel + log original width/height','pretrained_weights':False}
    dump('results/metrics.json',m)
    pd.DataFrame([{'split':s,'model':k,**v} for s,vals in scores.items() for k,v in vals.items()]).to_csv(ROOT/'results/metrics.csv',index=False)
    final=out['test'];p=final[selected]
    report=pd.DataFrame(classification_report(final.actual,p,labels=CLASSES,output_dict=True,zero_division=0)).T
    report.to_csv(ROOT/'results/classification_report.csv',index_label='class')
    cm=confusion_matrix(final.actual,p,labels=CLASSES);pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(ROOT/'results/confusion_matrix.csv',index_label='actual')
    for name,model in models.items():joblib.dump(model,ROOT/'models'/f'{name}.joblib')
    dump('models/metadata.json',{'dataset_sha256':sha(ROOT/'dataset.csv'),'split_sha256':sha(ROOT/'results/split_manifest.csv'),'selected_model':selected,'class_mapping':{3:'car',6:'bus',8:'truck'},'features':m['feature_description'],'feature_count':X.shape[1],'parameters':{k:v[2] for k,v in best.items()},'training_objects':int(tr.sum()),'validation_objects':int(va.sum()),'test_objects':int(te.sum()),'seed':SEED,'pretrained':False,'versions':{x:importlib.metadata.version(x) for x in ['numpy','pandas','scikit-learn','scikit-image','scipy','pillow','joblib']},'metrics':'../results/metrics.json','python':sys.version,'platform':platform.platform()})
    # Resample whole source groups, never individual object crops, for descriptive test uncertainty.
    rng=np.random.default_rng(SEED);groups=final.scene_group.unique();boot=[]
    for _ in range(500):
        take=pd.concat([final.loc[final.scene_group==g] for g in rng.choice(groups,len(groups),replace=True)],ignore_index=True)
        boot.append(metric(take.actual,take[selected])['macro_f1'])
    dump('results/bootstrap.json',{'method':'percentile bootstrap of held-out scene groups, 500 repetitions; not transportability bound','selected_model':selected,'macro_f1_95_percentile_interval':np.quantile(boot,[.025,.975]).tolist(),'groups':len(groups),'seed':SEED})
    pd.crosstab(df.split,df['class']).to_csv(ROOT/'results/class_counts.csv')
    eda=df.groupby('class')[['bbox_width','bbox_height']].agg(['count','median','min','max']);eda.to_csv(ROOT/'results/eda_box_sizes.csv')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    pd.crosstab(df['class'],df.split)[['train','validation','test']].plot.bar(figsize=(8,4),color=['#185B8B','#EEAF31','#498E60'],rot=0);plt.ylabel('Число объектов');plt.xlabel('Исходный класс COCO');plt.title('Состав выборок: сцены не пересекаются');savefig('class_distribution.png')
    plt.figure(figsize=(7,4));plt.boxplot([df.loc[df['class']==c,'bbox_width']/df.loc[df['class']==c,'bbox_height'] for c in CLASSES],tick_labels=CLASSES,showfliers=True);plt.ylabel('Ширина / высота рамки');plt.title('Форма размеченных объектов');savefig('aspect_ratios.png')
    tbl=pd.DataFrame([{'model':k,**v} for k,v in scores['test'].items()]).set_index('model');tbl[['macro_f1','accuracy']].plot.bar(figsize=(8,4),rot=0,color=['#185B8B','#EEAF31']);plt.ylim(0,1);plt.title('Одинаковый отложенный тест');savefig('model_comparison.png')
    plt.figure(figsize=(6,5));plt.imshow(cm,cmap='Blues');plt.xticks(range(3),CLASSES);plt.yticks(range(3),CLASSES);plt.xlabel('Предсказание');plt.ylabel('Истинная метка');plt.title(f'Матрица ошибок: {selected}');
    for i in range(3):
        for j in range(3):plt.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    savefig('confusion_matrix.png')
    errors=final.loc[final.actual!=final[selected]].copy();errors['predicted']=errors[selected];errors.to_csv(ROOT/'results/errors.csv',index=False)
    fig,axs=plt.subplots(2,3,figsize=(10,8))
    for ax,(_,r) in zip(axs.ravel(),errors.head(6).iterrows()):ax.imshow(Image.open(ROOT/r.crop_path).resize((64,64),Image.Resampling.BILINEAR));ax.set_title(f'{r.actual} → {r[selected]}\nCOCO {r.image_id}, object {r.object_id}',fontsize=10);ax.axis('off')
    for ax in axs.ravel():ax.axis('off')
    fig.suptitle('Первые ошибки: вход 64×64; порядок по ID',fontsize=14);savefig('error_examples.png')
    fig,axs=plt.subplots(3,3,figsize=(9,8))
    for i,c in enumerate(CLASSES):
        for ax,(_,r) in zip(axs[i],df.loc[df['class']==c].head(3).iterrows()):ax.imshow(Image.open(ROOT/r.crop_path).resize((64,64),Image.Resampling.BILINEAR));ax.set_title(f'{c}: COCO {r.image_id}\nobject {r.object_id}',fontsize=9);ax.axis('off')
    fig.suptitle('COCO: вход 64×64; атрибуция в image_attribution.csv',fontsize=13);savefig('crop_examples.png')
    checks=verify();dump('results/execution.json',{'completed_utc':pd.Timestamp.now(tz='UTC').isoformat(),'elapsed_seconds':time.perf_counter()-started,'fit_count':len(candidates),'cpu_models':True,'quality_passes':len(checks)})
    print(json.dumps(m,ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    verify() if args.verify_only else main()
