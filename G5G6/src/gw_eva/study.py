"""Single-GPU server study: train-only LR selection then 26 formal jobs.

RTE/MRPC seeds0-4; SST-2 seeds0-2. No official dev is used for tuning.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
import numpy as np
import requests
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset,load_from_disk
from transformers import AutoTokenizer,DataCollatorWithPadding,get_linear_schedule_with_warmup
from peft import PeftModel
from .runtime import ROOT,OUT,MODEL,write,model_load,configure,digest
from .calibrate import calibrate
from .allocation import count_allocate,cost_allocate
from .smoke import attach,encoder_digest

STUDY=OUT/'server_study'
TASKS={'rte':{'epochs':80,'batch':32,'seeds':5,'selection_epochs':10},
       'mrpc':{'epochs':30,'batch':16,'seeds':5,'selection_epochs':10},
       'sst2':{'epochs':10,'batch':32,'seeds':3,'selection_epochs':3}}
LRS=[.0001,.0004,.001]
LENGTH=512


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare(task):
    directory=ROOT/'data/gw_glue'/task;directory.mkdir(exist_ok=True,parents=True)
    hub=os.environ.get('GW_HF_ENDPOINT','https://huggingface.co').rstrip('/')
    if not (directory/'download.json').exists():
        sources=read(ROOT/'configs/data_sources.json')
        pinned=sources[task]['revision']
        response=requests.get(hub+'/api/datasets/nyu-mll/glue/revision/'+pinned+'?blobs=true',timeout=60);response.raise_for_status();info=response.json();files=[]
        assert info['sha']==pinned,'Unexpected dataset revision'
        for split in ['train','validation']:
            name=f'{task}/{split}-00000-of-00001.parquet';entry=next(x for x in info['siblings'] if x['rfilename']==name)
            expected=entry['lfs']['sha256'];target=directory/(split+'.parquet')
            assert expected==next(x['sha256'] for x in sources[task]['files'] if x['split']==split)
            old=ROOT/'data/glue_rte'/f'{split}-00000-of-00001.parquet'
            if task=='rte' and old.exists() and hashlib.file_digest(old.open('rb'),'sha256').hexdigest()==expected:
                import shutil
                shutil.copyfile(old,target)
            if not target.exists() or hashlib.file_digest(target.open('rb'),'sha256').hexdigest()!=expected:
                content=requests.get(f"{hub}/datasets/nyu-mll/glue/resolve/{info['sha']}/{name}",timeout=180);content.raise_for_status();target.write_bytes(content.content)
            assert hashlib.file_digest(target.open('rb'),'sha256').hexdigest()==expected
            files.append({'split':split,'sha256':expected,'bytes':target.stat().st_size})
        write(directory/'download.json',{'revision':info['sha'],'files':files})
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);collator=DataCollatorWithPadding(tokenizer)
    cache=directory/f'len{LENGTH}'
    if (cache/'dataset_dict.json').exists():data=load_from_disk(str(cache))
    else:
        raw=load_dataset('parquet',data_files={s:str(directory/(s+'.parquet')) for s in ['train','validation']},cache_dir=str(ROOT/'cache/gw_study'))
        data=raw.map(lambda b:tokenizer(b['sentence'] if task=='sst2' else b['sentence1'],None if task=='sst2' else b['sentence2'],truncation=True,max_length=LENGTH),batched=True,remove_columns=[k for k in raw['train'].column_names if k!='label'])
        data=data.rename_column('label','labels');data.save_to_disk(str(cache))
    split_path=directory/'selection_split.json'
    reference_split=ROOT/'configs/splits'/f'{task}.json'
    if not split_path.exists() and reference_split.exists():write(split_path,read(reference_split))
    if not split_path.exists():
        raw=load_dataset('parquet',data_files={'train':str(directory/'train.parquet')},cache_dir=str(ROOT/'cache/gw_study'))['train']
        groups={}
        for i,row in enumerate(raw):
            key=(row.get('sentence',row.get('sentence1','')).strip(),row.get('sentence2','').strip());groups.setdefault(key,[]).append(i)
        rng=np.random.default_rng(0);selected=set();labels=np.array(data['train']['labels'])
        for label in [0,1]:
            indices=np.flatnonzero(labels==label);rng.shuffle(indices);selected.update(indices[:round(len(indices)*.1)].tolist())
        target=len(selected)
        for group in groups.values():
            g=set(group)
            if g&selected and not g<=selected:
                add=selected|g;remove=selected-g;selected=remove if abs(len(remove)-target)<=abs(len(add)-target) else add
        fit=sorted(set(range(len(labels)))-selected);selection=sorted(selected)
        fit_set=set(fit);selection_set=set(selection)
        require_no_overlap=all(not(set(g)&fit_set and set(g)&selection_set) for g in groups.values());assert require_no_overlap
        write(split_path,{'fit':fit,'selection':selection,'seed':0,'grouped_duplicate_texts':True,'exact_text_overlap':0})
    split=read(split_path)
    assert not set(split['fit'])&set(split['selection'])
    assert set(split['fit'])|set(split['selection'])==set(range(len(data['train'])))
    return data,collator,split,read(directory/'download.json')['files'][0]['sha256']


def context(task,seed,partition):
    data,collator,split,sha=prepare(task);train=data['train'].select(split['fit']) if partition=='fit' else data['train']
    directory=STUDY/task/partition/f'seed{seed}'/'initialization';directory.mkdir(exist_ok=True,parents=True)
    head_path=directory/'head_warmup.pt';metadata=directory/'head_warmup.json'
    if not head_path.exists() or not metadata.exists():
        started=time.perf_counter();model=model_load(seed);before=encoder_digest(model)
        for p in model.parameters():p.requires_grad_(False)
        for p in model.classifier.parameters():p.requires_grad_(True)
        old={n:p.detach().cpu().clone() for n,p in model.classifier.named_parameters()}
        model.eval();model.classifier.train();optimizer=torch.optim.AdamW(model.classifier.parameters(),lr=.001,weight_decay=0)
        loader=DataLoader(train,batch_size=32,shuffle=True,collate_fn=collator,generator=torch.Generator().manual_seed(seed));visits=0
        device=next(model.parameters()).device
        with (directory/'warmup_steps.jsonl').open('w') as log:
            for step,record in enumerate(loader,1):
                inputs={k:v.to(device) for k,v in record.items()};optimizer.zero_grad(set_to_none=True);loss=model(**inputs).loss
                assert torch.isfinite(loss);loss.backward();optimizer.step();visits+=len(inputs['labels'])
                log.write(json.dumps({'step':step,'loss':float(loss),'samples':len(inputs['labels'])})+'\n')
        assert visits==len(train) and encoder_digest(model)==before
        assert any(not torch.equal(p.detach().cpu(),old[n]) for n,p in model.classifier.named_parameters())
        torch.save({n:p.cpu() for n,p in model.classifier.state_dict().items()},head_path)
        write(metadata,{'status':'completed','task':task,'seed':seed,'partition':partition,'epochs':1,'samples':visits,
            'max_length':LENGTH,'head_lr':.001,'head_batch':32,'train_data_sha256':sha,
            'indices':split['fit'] if partition=='fit' else list(range(len(train))),
            'model_revision':read(OUT/'download.json')['revision'],'encoder_unchanged':True,'seconds':time.perf_counter()-started})
        del model,optimizer;gc.collect();torch.cuda.empty_cache()
    return {'task':task,'partition':partition,'directory':directory,'data':train,'collator':collator,'data_sha':sha,'modes':['gw_uncentered']}


def initialization(task,seed,partition):
    base=STUDY/task/partition/f'seed{seed}'/'initialization'
    if all((base/g/'initial_adapter/adapter_config.json').exists() and (base/g/'verification.json').exists() for g in ['G5','G6']):return base
    ctx=context(task,seed,partition)
    artifact=calibrate(seed,LENGTH,min(256,len(ctx['data'])),micro_batch=2,context=ctx)
    value=torch.load(artifact,map_location='cpu',weights_only=False)
    scores={n:v['gw_uncentered']['scores'].tolist() for n,v in value['layers'].items()}
    costs={n:d['d_in']+d['d_out'] for n,d in value['dimensions'].items()}
    g5=count_allocate(scores);g6,report=cost_allocate(scores,costs)
    assert sum(g5.values())==576 and .99<=report['fraction']<=1
    record=ctx['collator']([ctx['data'][0],ctx['data'][1]])
    layers={n:{**value['dimensions'][n],'A':v['gw_uncentered']['components']} for n,v in value['layers'].items()}
    for group,ranks in [('G5',g5),('G6',g6)]:
        directory=base/group;directory.mkdir(exist_ok=True,parents=True)
        model=model_load(seed);model.classifier.load_state_dict(torch.load(base/'head_warmup.pt',weights_only=True,map_location='cpu'));model.eval()
        before=encoder_digest(model);inputs={k:v.cuda() for k,v in record.items()}
        with torch.no_grad():original=model(**inputs).logits
        model,details=attach(model,ranks,layers);model.eval()
        with torch.no_grad():difference=float((model(**inputs).logits-original).abs().max())
        assert difference==0 and encoder_digest(model)==before
        model.save_pretrained(directory/'initial_adapter',safe_serialization=True)
        write(directory/'rank_pattern.json',ranks);write(directory/'parameters.json',details)
        write(directory/'verification.json',{'status':'passed','initial_logits_difference':difference,'encoder_unchanged':True,
            'rank_sum':sum(ranks.values()),'parameters':sum(costs[n]*r for n,r in ranks.items()),
            'G6_budget':report if group=='G6' else None,'calibration':str(artifact),
            'calibration_sha256':hashlib.file_digest(artifact.open('rb'),'sha256').hexdigest()})
        del model;gc.collect();torch.cuda.empty_cache()
    return base


def metrics(labels,logits,task):
    prediction=np.argmax(logits,axis=-1);accuracy=float(np.mean(prediction==labels));result={'accuracy':accuracy}
    if task=='mrpc':
        tp=int(((prediction==1)&(labels==1)).sum());fp=int(((prediction==1)&(labels==0)).sum());fn=int(((prediction==0)&(labels==1)).sum())
        result['f1']=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
        result['selection_score']=(result['accuracy']+result['f1'])/2
    else:result['selection_score']=accuracy
    return result


def train(task,group,seed,phase,lr,base):
    directory=STUDY/task/phase/group/f'seed{seed}'/(f'lr{lr:g}' if phase=='selection' else 'final')
    result_path=directory/'result.json'
    if result_path.exists() and read(result_path).get('status')=='completed':return read(result_path)
    directory.mkdir(exist_ok=True,parents=True);write(STUDY/'active.json',{'task':task,'group':group,'seed':seed,'phase':phase,'lr':lr,'output':str(directory),'pid':os.getpid(),'unix_time':time.time()})
    data,collator,split,sha=prepare(task);training=data['train'].select(split['fit']) if phase=='selection' else data['train']
    evaluation=data['train'].select(split['selection']) if phase=='selection' else data['validation']
    epochs=TASKS[task]['selection_epochs'] if phase=='selection' else TASKS[task]['epochs'];micro=TASKS[task]['batch'];accum=1
    model=PeftModel.from_pretrained(model_load(seed),base/group/'initial_adapter',is_trainable=True);before=encoder_digest(model)
    device=next(model.parameters()).device
    loader=DataLoader(training,batch_size=micro,shuffle=True,collate_fn=collator,generator=torch.Generator().manual_seed(seed),pin_memory=True)
    planned=math.ceil(len(loader)/accum)*epochs
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=0)
    scheduler=get_linear_schedule_with_warmup(optimizer,math.ceil(.06*planned),planned)
    step=0;visits=0;start_epoch=0;started=time.perf_counter();checkpoint=directory/'checkpoint.pt';journal=[]
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model,state['adapter'])
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);start_epoch=state['next_epoch'];step=state['step'];visits=state['visits']
        loader.generator.set_state(state['sampler_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);journal=state['journal']
    torch.cuda.reset_peak_memory_stats();model.train()
    with (directory/'steps.jsonl').open('w',encoding='utf-8') as stream:
        for row in journal:stream.write(json.dumps(row)+'\n')
        for epoch in range(start_epoch,epochs):
            iterator=iter(loader)
            while True:
                window=[]
                for _ in range(accum):
                    try:window.append(next(iterator))
                    except StopIteration:break
                if not window:break
                samples=sum(len(b['labels']) for b in window);optimizer.zero_grad(set_to_none=True);loss_value=0.;step_started=time.perf_counter()
                for batch in window:
                    inputs={k:v.to(device,non_blocking=True) for k,v in batch.items()}
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**inputs).loss
                    assert torch.isfinite(loss);weight=len(batch['labels'])/samples;(loss*weight).backward();loss_value+=float(loss.detach())*weight
                Bnorm=torch.stack([p.grad.float().square().sum() for n,p in model.named_parameters() if 'lora_B' in n and p.grad is not None]).sum().sqrt().item()
                assert math.isfinite(Bnorm)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);optimizer.step();scheduler.step();step+=1;visits+=samples
                torch.cuda.synchronize();row={'step':step,'epoch':epoch+(step-(math.ceil(len(loader)/accum)*epoch))/math.ceil(len(loader)/accum),
                    'loss':loss_value,'B_gradient_frobenius_unclipped':Bnorm,'samples':samples,'seconds':time.perf_counter()-step_started}
                journal.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                if step%100==0:print('STEP',task,group,phase,seed,step,'/',planned,'seconds',row['seconds'],flush=True)
            from peft import get_peft_model_state_dict
            state={'adapter':get_peft_model_state_dict(model),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'step':step,'visits':visits,
                'next_epoch':epoch+1,'sampler_rng':loader.generator.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'journal':journal}
            temporary=checkpoint.with_suffix('.tmp');torch.save(state,temporary);temporary.replace(checkpoint)
            print('EPOCH',task,group,phase,seed,epoch+1,'/',epochs,flush=True)
    assert step==planned and visits==len(training)*epochs and encoder_digest(model)==before
    training_seconds=time.perf_counter()-started;model.eval();logits=[];labels=[]
    with torch.no_grad():
        for batch in DataLoader(evaluation,batch_size=32,shuffle=False,collate_fn=collator):
            inputs={k:v.to(device) for k,v in batch.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(**inputs).logits
            logits.append(prediction.float().cpu().numpy());labels.append(inputs['labels'].cpu().numpy())
    logits=np.concatenate(logits);labels=np.concatenate(labels);assert np.isfinite(logits).all()
    np.savez(directory/'predictions.npz',logits=logits,labels=labels);score=metrics(labels,logits,task)
    model.save_pretrained(directory/'adapter',safe_serialization=True)
    probe=collator([evaluation[i] for i in range(min(8,len(evaluation)))]);probe={k:v.to(device) for k,v in probe.items()}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):expected_probe=model(**probe).logits.float().cpu()
    fresh=PeftModel.from_pretrained(model_load(seed),directory/'adapter');fresh.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):actual_probe=fresh(**probe).logits.float().cpu()
    reload_error=float((expected_probe-actual_probe).abs().max());assert reload_error<1e-5 and encoder_digest(fresh)==before
    del fresh;gc.collect();torch.cuda.empty_cache()
    result={'status':'completed','task':task,'group':group,'seed':seed,'phase':phase,'lr':lr,'metrics':score,'epochs':epochs,
        'training_samples':len(training),'evaluation_samples':len(evaluation),'evaluation_partition':'train_holdout' if phase=='selection' else 'official_validation',
        'steps':step,'planned_steps':planned,'sample_visits':visits,'encoder_unchanged':True,'max_length':LENGTH,'micro_batch':micro,'effective_batch':TASKS[task]['batch'],
        'bf16_training':True,'seconds':time.perf_counter()-started,'training_segment_seconds':training_seconds,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
        'mean_step_seconds':statistics.mean(r['seconds'] for r in journal[10:]) if len(journal)>10 else statistics.mean(r['seconds'] for r in journal),
        'initialization':str(base/group/'initial_adapter'),'train_data_sha256':sha,'final_epoch_only':True,'reload_max_logit_difference':reload_error}
    write(result_path,result);print('COMPLETED',task,group,phase,seed,score,flush=True)
    del model,optimizer,scheduler,state;gc.collect();torch.cuda.empty_cache()
    return result


def summarize():
    jobs=[read(p) for p in STUDY.glob('*/formal/*/seed*/final/result.json') if read(p).get('status')=='completed'];rows=[]
    for task in TASKS:
        for group in ['G5','G6']:
            selected=[j for j in jobs if j['task']==task and j['group']==group]
            rows.append({'task':task,'group':group,'seeds':sorted(j['seed'] for j in selected),'n':len(selected),
                'accuracy_mean':statistics.mean(j['metrics']['accuracy'] for j in selected) if selected else None,
                'accuracy_std':statistics.stdev(j['metrics']['accuracy'] for j in selected) if len(selected)>1 else None,
                'f1_mean':statistics.mean(j['metrics']['f1'] for j in selected) if selected and task=='mrpc' else None,
                'f1_std':statistics.stdev(j['metrics']['f1'] for j in selected) if len(selected)>1 and task=='mrpc' else None})
    paired=[]
    for task,setting in TASKS.items():
        by_group={g:{j['seed']:j for j in jobs if j['task']==task and j['group']==g} for g in ['G5','G6']}
        seeds=sorted(set(by_group['G5'])&set(by_group['G6']))
        if len(seeds)>1:
            from scipy.stats import t
            differences=[by_group['G6'][s]['metrics']['accuracy']-by_group['G5'][s]['metrics']['accuracy'] for s in seeds]
            mean=statistics.mean(differences);half=float(t.ppf(.975,len(seeds)-1))*statistics.stdev(differences)/len(seeds)**.5
            paired.append({'task':task,'seeds':seeds,'G6_minus_G5_accuracy':mean,'paired_t_95_CI':[mean-half,mean+half],
                'complete':len(seeds)==setting['seeds'],'interpretation':'exploratory; G5 and G6 do not necessarily have identical parameter counts'})
    write(STUDY/'summary.json',{'completed':len(jobs),'required':26,'rows':rows,'paired':paired})
    lines=['# G5/G6 server study','',f'Completed formal jobs: {len(jobs)}/26. Incomplete entries are not predictions.','',
        '| Task | Group | Seeds | Accuracy mean | Sample std | MRPC F1 mean | F1 std |','|---|---|---|---|---|---|---|']
    for row in rows:lines.append('| '+' | '.join(str(row[k]) for k in ['task','group','seeds','accuracy_mean','accuracy_std','f1_mean','f1_std'])+' |')
    lines+=['','RTE80 / MRPC30 / SST2 10 epochs; max length512; final official dev only. G5 uses 576 total ranks; G6 uses a parameter budget. Actual counts are in per-seed initialization verification.json.','',json.dumps(paired,indent=2)]
    (STUDY/'summary.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tasks',nargs='+',choices=list(TASKS),default=list(TASKS));args=parser.parse_args()
    STUDY.mkdir(exist_ok=True,parents=True);write(STUDY/'pipeline.json',{'status':'running','pid':os.getpid(),'started_unix':time.time(),'tasks':args.tasks,'required_jobs':26})
    write(STUDY/'protocol.json',{'tasks':TASKS,'max_length':LENGTH,'calibration_samples':256,'calibration_micro_batch':2,'LR_grid':LRS,
        'training_micro_batch':'full effective batch: RTE32 MRPC16 SST2 32; resource-verified on4090','gradient_accumulation':1,
        'head_epochs':1,'head_lr':.001,'head_and_calibration_train_fit_only_during_selection':True,'selection_seed':0,
        'selection_metric':'RTE/SST2 accuracy; MRPC average(accuracy,F1)','formal_dev':'final epoch once; no best-dev selection',
        'methods':['G5','G6'],'G5_budget':'576 ranks; parameter count reported, not forced within1%','G6_budget':1327104,
        'epochs_choice':'RTE80 MRPC30 SST2 10; declared fixed before results, proposal did not specify RoBERTa epochs',
        'source_model':read(OUT/'download.json')})
    for task in args.tasks:
        hp_path=STUDY/task/'frozen_lr.json'
        if not hp_path.exists():
            base=initialization(task,0,'fit');chosen={}
            for group in ['G5','G6']:
                candidates=[train(task,group,0,'selection',lr,base) for lr in LRS]
                winner=sorted(candidates,key=lambda r:(-r['metrics']['selection_score'],r['lr']))[0]
                chosen[group]={'lr':winner['lr'],'train_holdout_score':winner['metrics']['selection_score'],
                    'candidates':[{'lr':r['lr'],'score':r['metrics']['selection_score']} for r in candidates]}
            write(hp_path,{'official_dev_used':False,'groups':chosen})
        hp=read(hp_path)
        for seed in range(TASKS[task]['seeds']):
            base=initialization(task,seed,'full')
            for group in ['G5','G6']:
                train(task,group,seed,'formal',hp['groups'][group]['lr'],base);summarize()
    summarize();write(STUDY/'pipeline.json',{'status':'completed','tasks':args.tasks,'finished_unix':time.time()})


if __name__=='__main__':main()
