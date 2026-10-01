"""30 s real CC0 dashcam clip with separately timestamped manual-crop predictions."""
from pathlib import Path
import sys, json, hashlib, urllib.request, subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pipeline import features
import cv2, numpy as np, joblib
from PIL import Image
URL='https://upload.wikimedia.org/wikipedia/commons/transcoded/e/ea/Dashcam_Recording_%28urban%29.ogv/Dashcam_Recording_%28urban%29.ogv.480p.vp9.webm'

def main():
    source=ROOT/'raw/dashcam_urban_480p.webm'
    if not source.exists():source.write_bytes(urllib.request.urlopen(URL,timeout=120).read())
    expected=json.loads((ROOT/'raw/source_integrity.json').read_text(encoding='utf-8'))['files']['raw/dashcam_urban_480p.webm']
    if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:raise ValueError('Dashcam source checksum differs from the fixed research clip')
    selected=json.loads((ROOT/'results/metrics.json').read_text(encoding='utf-8'))['selected_model']
    model_path=ROOT/'models'/f'{selected}.joblib'
    model=joblib.load(model_path)
    cap=cv2.VideoCapture(str(source));fps=cap.get(cv2.CAP_PROP_FPS);total=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    assert cap.isOpened() and abs(fps-30)<.01 and total/fps>=115,'Fixed source must be a 30 fps clip covering 85–115 s'
    # Fixed rectangles selected by visual inspection, no detector or automatic tracking.
    rois=[{'source_second':60,'bbox':[180,279,94,44]},{'source_second':100,'bbox':[347,274,53,39]}]
    crops=[]
    for r in rois:
        cap.set(cv2.CAP_PROP_POS_MSEC,r['source_second']*1000);ok,frame=cap.read();assert ok
        x,y,w,h=r['bbox']; crop=frame[y:y+h,x:x+w]
        assert crop.shape[:2]==(h,w),'Manual rectangle must stay inside source frame'
        pred=str(model.predict(features(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)))[None,:])[0]);r['predicted']=pred;r['reference_label']='none_for_quantitative_evaluation';r['selection']='manually specified rectangle from visual inspection'
        crops.append(crop)
    output=ROOT/'results/demo_intermediate.mp4';writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),15,(1280,720));assert writer.isOpened()
    cap.set(cv2.CAP_PROP_POS_MSEC,85000)
    for k in range(450):
        # Read two source frames for each output frame at 15 fps (source is 30 fps).
        ok,frame=cap.read()
        if k>0:
            ok,frame=cap.read()
        assert ok,'Source unexpectedly short'
        canvas=np.full((720,1280,3),(25,31,42),dtype=np.uint8)
        def text(s,x,y,size=.7,color=(239,242,248)):cv2.putText(canvas,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)
        text('Vehicle crop classification | real dashcam demonstration',25,40,.9)
        text('Training: COCO stills. Filming location unknown; Atyrau unconfirmed.',25,75,.65)
        # Low-resolution and softened display limits incidental plate/face detail.
        shown=cv2.resize(cv2.GaussianBlur(frame,(5,5),0),(854,480));canvas[130:610,20:874]=shown
        text(f'Source video: {85+k/15:.1f} s | public location unspecified',25,645,.65)
        text('Fernost / Wikimedia Commons / CC0 1.0 (2013)',25,678,.65)
        text('MANUAL CROP EXAMPLES',900,135,.65,(110,210,249))
        for j,(r,crop) in enumerate(zip(rois,crops)):
            y0=165+j*235
            canvas[y0:y0+135,915:1245]=cv2.resize(crop,(330,135))
            text(f"Source t={r['source_second']} s",910,y0+162,.6)
            text(f"Model predicts: {r['predicted']}",910,y0+190,.7,(120,221,130))
        text('Rectangles selected manually.',900,651,.52)
        text('No detector, tracking or video metrics.',900,676,.46)
        writer.write(canvas)
        if k==225:cv2.imwrite(str(ROOT/'figures/demo_frame.png'),canvas)
    writer.release();cap.release()
    result=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(output),'-c:v','libx264','-pix_fmt','yuv420p','-crf','23','-movflags','+faststart',str(ROOT/'demo.mp4')],capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr)
    check=cv2.VideoCapture(str(ROOT/'demo.mp4'));n=int(check.get(cv2.CAP_PROP_FRAME_COUNT));fps2=check.get(cv2.CAP_PROP_FPS);read=0
    while True:
        ok,fr=check.read()
        if not ok:break
        read+=1
    check.release();assert n==450 and read==450 and abs(fps2-15)<.01
    meta={'source_url':URL,'source_page':'https://commons.wikimedia.org/wiki/File:Dashcam_Recording_(urban).ogv','creator':'Fernost','license':'CC0 1.0','license_url':'https://creativecommons.org/publicdomain/zero/1.0/','source_date':'2013 (page date June30; embedded timestamp June18; not resolved)','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_duration_seconds':total/fps,'output_duration_seconds':n/fps2,'frames_decoded':read,'output_resolution':[1280,720],'source_clip_start_seconds':85,'source_clip_duration_seconds':30,'output_sha256':hashlib.sha256((ROOT/'demo.mp4').read_bytes()).hexdigest(),'rois':rois,'ground_truth_evaluation':False,'camera_domain':'real dashcam; filming location not established; Atyrau unconfirmed','privacy':'Displayed footage downsampled and Gaussian-softened; no identities or plates transcribed.'}
    meta.update({'selected_model':selected,'model_file':model_path.relative_to(ROOT).as_posix(),'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),'source_fps':fps,'source_hash_matches_pinned_snapshot':True,'roi_labels_are_model_predictions_only':True})
    (ROOT/'results/demo_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    output.unlink()  # Only this script's intermediate encode; final and source are retained.
    print(json.dumps(meta,indent=2))

if __name__=='__main__':main()
