"""Fetch permissively transformable academic-use COCO vehicle photographs."""
from pathlib import Path
import csv, json, hashlib, urllib.request, concurrent.futures, io, re, datetime, time
from PIL import Image
from acquire_annotations import main as annotations_main
ROOT=Path(__file__).resolve().parents[1]
CLASSES={3:'car',6:'bus',8:'truck'}

def sha(blob):return hashlib.sha256(blob).hexdigest()
def main():
    annotations_main()
    integrity=json.loads((ROOT/'raw/source_integrity.json').read_text(encoding='utf-8'))
    d=json.loads((ROOT/'raw/instances_val2017.json').read_text(encoding='utf-8'))
    ims={x['id']:x for x in d['images']}; licenses={x['id']:x for x in d['licenses']}
    raw_a=[a for a in d['annotations'] if a['category_id'] in CLASSES]
    selected=[a for a in raw_a if ims[a['image_id']]['license'] in [1,2,4,5] and a['bbox'][2]>=40 and a['bbox'][3]>=40 and a['iscrowd']==0]
    selected.sort(key=lambda a:a['id'])
    imlist=[ims[i] for i in sorted({a['image_id'] for a in selected})]
    (ROOT/'raw/images').mkdir(exist_ok=True); (ROOT/'processed/crops').mkdir(parents=True,exist_ok=True)
    def fetch(im):
        p=ROOT/'raw/images'/im['file_name']
        status='cached'
        if not p.exists():
            for attempt in range(3):
                try:
                    blob=urllib.request.urlopen(im['coco_url'],timeout=45).read()
                    Image.open(io.BytesIO(blob)).verify();p.write_bytes(blob); status='downloaded';break
                except Exception as e:
                    if attempt==2:return {'image_id':im['id'],'status':'failed','error':str(e)}
                    time.sleep(1)
        blob=p.read_bytes();img=Image.open(io.BytesIO(blob));img.verify()
        relative=p.relative_to(ROOT).as_posix()
        if sha(blob)!=integrity['files'].get(relative):
            raise ValueError(f'Source image checksum mismatch: {relative}. The fixed source snapshot must not change silently.')
        if Image.open(io.BytesIO(blob)).size!=(im['width'],im['height']):
            raise ValueError(f'Source dimensions mismatch: {relative}')
        return {'image_id':im['id'],'status':status,'url':im['coco_url'],'relative_path':p.relative_to(ROOT).as_posix(),'bytes':len(blob),'sha256':sha(blob),'accessed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        downloads=list(ex.map(fetch,imlist))
    (ROOT/'results/download_manifest.json').write_text(json.dumps(downloads,indent=2))
    success={r['image_id']:r for r in downloads if r['status']!='failed'}
    if len(success)!=len(imlist):
        raise RuntimeError('One or more fixed-sample images could not be downloaded; refuse to train on a changed subset. See download_manifest.json.')
    rows=[]
    for a in selected:
        if a['image_id'] not in success:continue
        im=ims[a['image_id']]; path=ROOT/'raw/images'/im['file_name']
        img=Image.open(path).convert('RGB'); x,y,w,h=a['bbox']
        box=(max(0,int(x)),max(0,int(y)),min(img.width,int(x+w+0.999)),min(img.height,int(y+h+0.999)))
        crop=img.crop(box)
        # Preserve original resolution for the experiment; no augmentation is precomputed.
        out=ROOT/'processed/crops'/f"{a['id']}.png";crop.save(out)
        fid=re.search(r'/(\d+)_',im['flickr_url']).group(1)
        rows.append({'object_id':a['id'],'image_id':a['image_id'],'scene_id':str(a['image_id']),'video_id':'not_available_still_image','class':CLASSES[a['category_id']],'category_id':a['category_id'],'crop_path':out.relative_to(ROOT).as_posix(),'image_path':path.relative_to(ROOT).as_posix(),'bbox_x':x,'bbox_y':y,'bbox_width':w,'bbox_height':h,'image_width':im['width'],'image_height':im['height'],'iscrowd':a['iscrowd'],'license_id':im['license'],'license_name':licenses[im['license']]['name'],'license_url':licenses[im['license']]['url'],'flickr_url':im['flickr_url'],'flickr_photo_page':f'https://www.flickr.com/photo.gne?id={fid}','creator':'not supplied in COCO metadata; original Flickr source linked','title':f'COCO photograph {im["id"]}; original title not supplied','modification':'bounding-box crop; per-image original license retained','coco_url':im['coco_url'],'image_sha256':success[im['id']]['sha256'],'crop_sha256':sha(out.read_bytes()),'geography':'not specified by COCO image metadata; not claimed Atyrau','date_captured_metadata':im['date_captured']})
    with (ROOT/'dataset.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    with (ROOT/'results/image_attribution.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    meta={'source_images':len(d['images']),'source_annotations':len(d['annotations']),'vehicle_annotations_before_filters':len(raw_a),'eligible_objects':len(selected),'eligible_images':len(imlist),'downloaded_images':len(success),'prepared_objects':len(rows),'min_bbox_pixels':40,'allowed_license_ids':[1,2,4,5],'selection':'all qualifying objects in COCO val2017, no model-driven selection','failed_downloads':len(imlist)-len(success)}
    (ROOT/'results/preparation.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))

if __name__=='__main__':main()
