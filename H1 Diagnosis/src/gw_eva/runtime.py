"""RoBERTa-only runtime, preserving all existing experiments and packages."""
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer,DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader
from .prepare import ROOT,OUT,MODEL

os.environ['TOKENIZERS_PARALLELISM']='false'


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


def configure(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def digest(model,encoder_only=False):
    value=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        if encoder_only and not name.startswith('roberta.'):continue
        value.update(name.encode());value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def model_load(seed=0):
    configure(seed)
    model=AutoModelForSequenceClassification.from_pretrained(MODEL,num_labels=2,local_files_only=True,attn_implementation='eager').float()
    return model.to('cuda' if torch.cuda.is_available() else 'cpu')


def targets(model):
    suffixes=['attention.self.query','attention.self.key','attention.self.value','attention.output.dense','intermediate.dense','output.dense']
    modules={f'roberta.encoder.layer.{i}.{s}':model.get_submodule(f'roberta.encoder.layer.{i}.{s}') for i in range(12) for s in suffixes}
    assert len(modules)==72 and all(isinstance(m,torch.nn.Linear) for m in modules.values())
    return modules


def rte_data(max_length=128):
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    path=ROOT/'data/glue_rte/train-00000-of-00001.parquet'
    if not path.exists():path=ROOT/'data/gw_glue/rte/train.parquet'
    raw=load_dataset('parquet',data_files={'train':str(path)},cache_dir=str(ROOT/'cache/gw_eva_data'))['train']
    data=raw.map(lambda b:tokenizer(b['sentence1'],b['sentence2'],max_length=max_length,truncation=True),batched=True,remove_columns=[k for k in raw.column_names if k!='label'])
    data=data.rename_column('label','labels')
    return data,DataCollatorWithPadding(tokenizer),hashlib.file_digest(path.open('rb'),'sha256').hexdigest()


def warmup(seed=0,max_length=128,batch=16,lr=.001):
    import time
    configure(seed);directory=OUT/f'rte_seed{seed}_len{max_length}';directory.mkdir(parents=True,exist_ok=True)
    target=directory/'head_warmup.pt';metadata=directory/'head_warmup.json'
    if target.exists() and metadata.exists() and json.loads(metadata.read_text())['status']=='completed':return directory
    started=time.perf_counter();model=model_load(seed);data,collator,data_sha=rte_data(max_length)
    original=digest(model,True);head_before={n:p.detach().cpu().clone() for n,p in model.classifier.named_parameters()}
    for p in model.parameters():p.requires_grad_(False)
    for p in model.classifier.parameters():p.requires_grad_(True)
    model.eval();model.classifier.train()
    optimizer=torch.optim.AdamW(model.classifier.parameters(),lr=lr,weight_decay=0)
    loader=DataLoader(data,batch_size=batch,shuffle=True,collate_fn=collator,generator=torch.Generator().manual_seed(seed))
    visits=0;steps=[];device=next(model.parameters()).device
    torch.cuda.reset_peak_memory_stats()
    for step,record in enumerate(loader,1):
        inputs={k:v.to(device) for k,v in record.items()};optimizer.zero_grad(set_to_none=True)
        loss=model(**inputs).loss;assert torch.isfinite(loss)
        loss.backward();optimizer.step();visits+=len(inputs['labels'])
        steps.append({'step':step,'loss':float(loss),'samples':len(inputs['labels'])})
        if step%20==0:print('WARMUP',step,'/',len(loader),'loss',float(loss),flush=True)
    assert visits==len(data) and digest(model,True)==original
    changed=sum(not torch.equal(p.detach().cpu(),head_before[n]) for n,p in model.classifier.named_parameters());assert changed>0
    torch.save(model.classifier.state_dict(),target)
    revision=json.loads((OUT/'download.json').read_text())['revision']
    write(metadata,{'status':'completed','model':'FacebookAI/roberta-base','revision':revision,'seed':seed,
          'data':'GLUE RTE train only; no validation loaded','data_sha256':data_sha,'train_samples':len(data),'epochs':1,
          'batch_size':batch,'max_length':max_length,'lr':lr,'optimizer':'AdamW','weight_decay':0,
          'classifier':'RoBERTa classification head dense + out_proj; frozen encoder (MLP head probing)',
          'steps':steps,'sample_visits':visits,'head_tensors_changed':changed,'encoder_sha256':original,
          'model_sha256_after_warmup':digest(model),'seconds':time.perf_counter()-started,
          'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'gpu':torch.cuda.get_device_name(),
          'shared_groups':['G3','G4','G5','G6']})
    print('WARMUP_COMPLETED',visits,'samples',time.perf_counter()-started,'seconds',flush=True)
    return directory
