"""CPU-only canonical CSV preprocessing. No IO-VNBD interpretation or pairing decisions."""
import argparse,csv
from pathlib import Path
import numpy as np
from common import read,write,digest
CHANNELS=['ax','ay','az','gx','gy','gz']

def load_csv(path):
    # Canonical SI input supplied after human review, not an automatic raw-dataset adapter.
    with Path(path).open() as f: rows=list(csv.DictReader(f))
    a=np.array([[float(r[k]) for k in ['timestampNs',*CHANNELS,'speedMps']] for r in rows],dtype=np.float64)
    if a.ndim!=2 or a.shape[1]!=8 or len(a)<2 or not np.isfinite(a).all() or (np.diff(a[:,0])<=0).any():raise ValueError('Invalid canonical trip')
    return a

def resolve_trip_path(manifest, path_str):
    p = Path(path_str)
    if p.is_file(): return p
    mp = Path(manifest).resolve()
    for base in (mp.parent, mp.parent.parent, mp.parent.parent.parent):
        c = base / path_str
        if c.is_file(): return c
    return p

def prepare(manifest,out,toy=False,approval=None):
    m=read(manifest);h=digest(manifest)
    if m.get('schema')!='navdr.split.v1':raise ValueError('Invalid split schema')
    if not (toy and m.get('synthetic') is True) and approval!=h:raise ValueError('STOP: human-reviewed manifest SHA256 required; draft is not approved')
    if out.exists():raise ValueError('Choose a new output directory')
    # Verify grouping even for edited manifests; connected identifiers cannot cross splits.
    assignments={};ids=set()
    for t in m['trips']:
        if t['id'] in ids or t['split'] not in ('train','dev','test'):raise ValueError('Invalid trip/split')
        ids.add(t['id'])
        for k in ('group','recording_group','vehicle','phone','route'):
            if t.get(k) is not None:
                token=(k,str(t[k]));prior=assignments.setdefault(token,t['split'])
                if prior!=t['split']:raise ValueError('Group leakage: '+str(token))
        path = resolve_trip_path(manifest, t['path'])
        if digest(path)!=t['sha256']:raise ValueError('Trip hash changed')
    # Welford-style merged statistics use only training trips, before overlapping windows.
    n=0;mean=np.zeros(6);ss=np.zeros(6)
    for t in m['trips']:
        if t['split']!='train':continue
        x=load_csv(resolve_trip_path(manifest, t['path']))[:,1:7];count=len(x);mu=x.mean(0);delta=mu-mean
        ss+=((x-mu)**2).sum(0)+delta**2*n*count/(n+count);mean+=delta*count/(n+count);n+=count
    if n<2:raise ValueError('No usable training samples')
    std=np.maximum(np.sqrt(ss/n),1e-6);out.mkdir(parents=True)
    trips=[]
    for i,t in enumerate(m['trips']):
        a=load_csv(resolve_trip_path(manifest, t['path']));a[:,1:7]=(a[:,1:7]-mean)/std
        name=f'trip-{i:04d}.npy';np.save(out/name,a);trips.append({**t,'array':name})
    result=dict(schema='navdr.prepared.v1',synthetic=m.get('synthetic') is True,manifestSha256=h,
        approval='toy-only exemption' if toy else approval,normalization=dict(mean=mean.tolist(),std=std.tolist(),trainingRows=n,fitTripIds=[t['id'] for t in m['trips'] if t['split']=='train']),trips=trips)
    write(out/'manifest.json',result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path);p.add_argument('output',type=Path);p.add_argument('--toy',action='store_true');p.add_argument('--approved-manifest-sha256');a=p.parse_args()
    prepare(a.manifest,a.output,a.toy,a.approved_manifest_sha256);print(a.output/'manifest.json')
