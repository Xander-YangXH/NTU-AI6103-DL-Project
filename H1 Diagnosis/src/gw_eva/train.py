"""Explicit opt-in training from a G5/G6 initialization; never auto-sweeps."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from peft import PeftModel
from .runtime import model_load,rte_data,write
from .smoke import encoder_digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--init-adapter',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--micro-batch',type=int,default=2);p.add_argument('--accumulation',type=int,default=16)
    p.add_argument('--max-length',type=int,default=128);p.add_argument('--lr',type=float,default=.004)
    p.add_argument('--max-steps',type=int);a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):raise ValueError('Use a fresh output directory to preserve experiments')
    if min(a.epochs,a.micro_batch,a.accumulation)<1:raise ValueError('Positive training parameters required')
    a.output.mkdir(exist_ok=True,parents=True)
    model=PeftModel.from_pretrained(model_load(a.seed),a.init_adapter,is_trainable=True)
    before=encoder_digest(model);data,collator,_=rte_data(a.max_length)
    loader=DataLoader(data,batch_size=a.micro_batch,shuffle=True,collate_fn=collator,generator=torch.Generator().manual_seed(a.seed))
    planned=math.ceil(len(loader)/a.accumulation)*a.epochs
    if a.max_steps is not None:planned=min(planned,a.max_steps)
    if planned<1:raise ValueError('Positive optimization steps required')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr,weight_decay=0)
    scheduler=get_linear_schedule_with_warmup(optimizer,math.ceil(.06*planned),planned)
    model.train();device=next(model.parameters()).device;step=0;visits=0;started=time.perf_counter();journal=[]
    with (a.output/'steps.jsonl').open('w',encoding='utf-8') as log:
        for epoch in range(a.epochs):
            iterator=iter(loader)
            while step<planned:
                window=[]
                for _ in range(a.accumulation):
                    try:window.append(next(iterator))
                    except StopIteration:break
                if not window:break
                examples=sum(len(b['labels']) for b in window);optimizer.zero_grad(set_to_none=True);loss_value=0.
                for batch in window:
                    inputs={k:v.to(device) for k,v in batch.items()};loss=model(**inputs).loss
                    assert torch.isfinite(loss);weight=len(batch['labels'])/examples
                    (loss*weight).backward();loss_value+=float(loss.detach())*weight
                Bnorm=sum(float(p.grad.float().square().sum()) for n,p in model.named_parameters() if 'lora_B' in n and p.grad is not None)**.5
                assert math.isfinite(Bnorm)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                optimizer.step();scheduler.step();step+=1;visits+=examples
                row={'step':step,'epoch_index':epoch,'loss':loss_value,'unclipped_B_gradient_frobenius':Bnorm,'samples':examples}
                journal.append(row);log.write(json.dumps(row)+'\n');log.flush()
            model.save_pretrained(a.output/'adapter',safe_serialization=True)
            torch.save({'step':step,'epoch_index':epoch,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},a.output/'optimizer_checkpoint.pt')
            if step>=planned:break
    assert step==planned and encoder_digest(model)==before
    assert all(torch.isfinite(p).all() for p in model.parameters() if p.requires_grad)
    write(a.output/'result.json',{'status':'completed','task':'rte','config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        'steps':step,'sample_visits':visits,'encoder_unchanged':True,'seconds':time.perf_counter()-started,
        'validation_loaded':False,'note':'Training-only module; no validation-driven checkpoint/initialization selection',
        'early_steps':journal[:100]})
    print('TRAINING_COMPLETED',step,'steps',flush=True)


if __name__=='__main__':main()
