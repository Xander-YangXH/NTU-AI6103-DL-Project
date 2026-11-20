"""Download a pinned public RoBERTa checkpoint without changing dependencies."""
import hashlib
import json
import time
import os
from pathlib import Path
import requests

ROOT=Path(os.environ.get('GWEVA_ROOT',Path(__file__).resolve().parents[2])).resolve()
OUT=ROOT/'outputs/gw_eva'
MODEL=ROOT/'models/roberta-base'


def main():
    OUT.mkdir(parents=True,exist_ok=True);MODEL.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    previous=OUT/'download.json'
    reference=previous if previous.exists() else ROOT/'configs/model.json'
    reference_info=json.loads(reference.read_text(encoding='utf-8')) if reference.exists() else {}
    pinned=reference_info.get('revision')
    known_hashes={item['name']:item['sha256'] for item in reference_info.get('files',[])}
    hub=os.environ.get('GW_HF_ENDPOINT','https://huggingface.co').rstrip('/')
    endpoint=hub+'/api/models/FacebookAI/roberta-base'
    if pinned:endpoint+='/revision/'+pinned
    api=requests.get(endpoint+'?blobs=true',timeout=60)
    api.raise_for_status();info=api.json()
    if pinned:assert info['sha']==pinned,'Unexpected model revision'
    wanted=['config.json','vocab.json','merges.txt','tokenizer.json','tokenizer_config.json']
    names={x['rfilename']:x for x in info['siblings']}
    wanted=[x for x in wanted if x in names]
    wanted.append('model.safetensors' if 'model.safetensors' in names else 'pytorch_model.bin')
    records=[]
    for name in wanted:
        target=MODEL/name;entry=names[name];expected=known_hashes.get(name) or entry.get('lfs',{}).get('sha256')
        valid=target.exists() and (not expected or hashlib.file_digest(target.open('rb'),'sha256').hexdigest()==expected)
        if not valid:
            partial=target.with_suffix(target.suffix+'.partial')
            with requests.get(f"{hub}/FacebookAI/roberta-base/resolve/{info['sha']}/{name}",stream=True,timeout=(30,180)) as response:
                response.raise_for_status()
                with partial.open('wb') as stream:
                    for chunk in response.iter_content(4*1024*1024):stream.write(chunk)
            actual=hashlib.file_digest(partial.open('rb'),'sha256').hexdigest()
            if expected and expected!=actual:raise ValueError('Checksum mismatch: '+name)
            partial.replace(target)
        records.append({'name':name,'sha256':hashlib.file_digest(target.open('rb'),'sha256').hexdigest(),'bytes':target.stat().st_size})
        print('VERIFIED',name,target.stat().st_size,flush=True)
    (OUT/'download.json').write_text(json.dumps({'repo':'FacebookAI/roberta-base','revision':info['sha'],'seconds':time.perf_counter()-started,'files':records},indent=2),encoding='utf-8')


if __name__=='__main__':main()
