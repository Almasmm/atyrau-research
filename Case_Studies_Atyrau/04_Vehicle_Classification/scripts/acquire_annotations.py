"""Read the named public COCO ZIP member with standard HTTP Range requests."""
from pathlib import Path
import io, json, urllib.request, zipfile, hashlib, datetime

ROOT=Path(__file__).resolve().parents[1]
URL='http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
EXPECTED_SHA256='e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f'
class HTTPFile(io.RawIOBase):
    def __init__(self,url):
        self.url=url; self.pos=0
        self.size=int(urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=30).headers['Content-Length'])
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        if n==0:return b''
        req=urllib.request.Request(self.url,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}'})
        with urllib.request.urlopen(req,timeout=90) as r:
            assert r.status==206, 'Server must support Range'
            blob=r.read()
        self.pos+=len(blob)
        return blob

def main():
    output=ROOT/'raw/instances_val2017.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    (ROOT/'results').mkdir(exist_ok=True)
    was_cached=output.exists()
    if not output.exists():
        with zipfile.ZipFile(HTTPFile(URL)) as z:
            blob=z.read('annotations/instances_val2017.json')
        output.write_bytes(blob)
    digest=hashlib.sha256(output.read_bytes()).hexdigest()
    if digest!=EXPECTED_SHA256:
        raise ValueError('COCO annotation checksum differs from the research snapshot; do not silently replace the experiment source')
    doc=json.loads(output.read_text(encoding='utf-8'))
    print('Verified COCO JSON:',len(doc['images']),'images',len(doc['annotations']),'objects')
    print('Licenses:',doc['licenses'])
    meta={'url':URL,'member':'annotations/instances_val2017.json','sha256':digest,'bytes':output.stat().st_size,'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cache_used':was_cached,'method':'Pinned SHA-256 checked on every run; initial download used HTTP Range and ZIP CRC','checksum_scope':'Locally recorded research snapshot, not a separately published provider signature','transport_note':'Publisher links to HTTP; HTTPS host certificate name mismatch observed. No verification disabled.'}
    (ROOT/'results/annotation_download.json').write_text(json.dumps(meta,indent=2))

if __name__=='__main__':main()
