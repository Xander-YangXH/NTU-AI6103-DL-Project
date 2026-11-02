"""Aligned per-token output-gradient hooks and bounded CPU SVD calibration."""
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .runtime import OUT,configure,model_load,targets,rte_data,digest,write,warmup
from .spectral import StreamingSVD,compare


def calibrate(seed=0,max_length=128,samples=64,micro_batch=1,context=None):
    directory=Path(context['directory']) if context else warmup(seed,max_length)
    destination=directory/f'calibration_n{samples}_b{micro_batch}.pt'
    if destination.exists():return destination
    configure(seed);model=model_load(seed);model.classifier.load_state_dict(torch.load(directory/'head_warmup.pt',weights_only=True))
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    data,collator,data_sha=(context['data'],context['collator'],context['data_sha']) if context else rte_data(max_length)
    if not 0<samples<=len(data):raise ValueError('Invalid sample count')
    indices=np.random.default_rng(seed).permutation(len(data))[:samples].tolist()
    loader=DataLoader(data.select(indices),batch_size=micro_batch,shuffle=False,collate_fn=collator)
    modules=targets(model);pending={};gradient_handles=[];handles=[];active={};sketches={}
    stats={n:{'tokens':0,'gradient_calls':0,'gradient_norm_sum':0.,'gradient_norm_square_sum':0.,
              'gradient_norm_min':float('inf'),'gradient_norm_max':0.,'sum_x2':0.,'sum_x2_square':0.,
              'sum_g2':0.,'sum_g2_square':0.,'sum_g2_x2':0.,'nonzero_gradient_tokens':0} for n in modules}
    modes=context.get('modes',['gw_uncentered']) if context else ['eva_centered','activation_uncentered','gw_uncentered']
    for name,module in modules.items():
        sketches[name]={mode:StreamingSVD(module.in_features,16,centered=(mode=='eva_centered')) for mode in modes}
    baseline=digest(model);embedding_flag={'calls':0}
    hooks_before=sum(len(m._forward_hooks) for m in model.modules())
    def embedding_hook(module,args,output):
        embedding_flag['calls']+=1
        return output.requires_grad_(True)
    handles.append(model.roberta.embeddings.register_forward_hook(embedding_hook))
    def make_hook(name):
        def forward(module,args,output):
            x=args[0];mask=active['mask'];batch=active['batch']
            if x.shape[:2]!=mask.shape or output.shape[:2]!=mask.shape or x.ndim!=3 or output.ndim!=3:
                raise ValueError('Input/output/mask mismatch '+name)
            if not output.requires_grad:raise RuntimeError('No graph for '+name)
            if name in pending:raise RuntimeError('Unexpected repeated target call')
            pending[name]={'x':x.detach().float().cpu(),'mask':mask.detach().cpu()}
            def backward(gradient):
                if gradient.shape!=output.shape:raise ValueError('Gradient alignment mismatch')
                # Cross entropy mean over examples: undo 1/B for per-example
                # gradients, including a possibly shorter final microbatch.
                g=gradient.detach().float()*batch
                if not torch.isfinite(g).all():raise ValueError('Nonfinite output gradient')
                pending[name]['norm']=g.norm(dim=-1).cpu()
                stats[name]['gradient_calls']+=1
            gradient_handles.append(output.register_hook(backward))
        return forward
    for name,module in modules.items():handles.append(module.register_forward_hook(make_hook(name)))
    started=time.perf_counter();device=next(model.parameters()).device;losses=[];valid_counts=[]
    torch.cuda.reset_peak_memory_stats()
    try:
        for step,record in enumerate(loader,1):
            inputs={k:v.to(device) for k,v in record.items()};active.update(mask=inputs['attention_mask'].bool(),batch=len(inputs['labels']))
            output=model(**inputs);loss=output.loss
            if not torch.isfinite(loss):raise ValueError('Nonfinite classification loss')
            loss.backward();losses.append(float(loss));valid_counts.append(int(active['mask'].sum()))
            if set(pending)!=set(modules):raise RuntimeError('Missing forward targets')
            for name in modules:
                row=pending[name]
                if 'norm' not in row:raise RuntimeError('Missing gradient '+name)
                mask=row['mask'];x=row['x'][mask];norm=row['norm'][mask]
                if len(x)!=valid_counts[-1]:raise RuntimeError('Token alignment')
                # Empty masks safely update no rows; a classification batch
                # itself must contain at least the nonpadding CLS token.
                if 'eva_centered' in modes:sketches[name]['eva_centered'].update(x)
                if 'activation_uncentered' in modes:sketches[name]['activation_uncentered'].update(x)
                sketches[name]['gw_uncentered'].update(x*norm[:,None])
                s=stats[name];s['tokens']+=len(x)
                if len(x):
                    n=norm.double();g2=n.square();x2=x.double().square().sum(-1)
                    s['gradient_norm_sum']+=float(n.sum());s['gradient_norm_square_sum']+=float(g2.sum())
                    s['gradient_norm_min']=min(s['gradient_norm_min'],float(n.min()));s['gradient_norm_max']=max(s['gradient_norm_max'],float(n.max()))
                    for key,value in [('sum_x2',x2.sum()),('sum_x2_square',x2.square().sum()),('sum_g2',g2.sum()),('sum_g2_square',g2.square().sum()),('sum_g2_x2',(g2*x2).sum())]:s[key]+=float(value)
                    s['nonzero_gradient_tokens']+=int((norm>0).sum())
            for handle in gradient_handles:handle.remove()
            gradient_handles.clear();pending.clear();active.clear();del output,loss,inputs;gc.collect()
            print('CALIBRATION',step,'/',len(loader),'valid_tokens',sum(valid_counts),'seconds',time.perf_counter()-started,flush=True)
    finally:
        for handle in handles+gradient_handles:handle.remove()
        pending.clear();active.clear()
    assert digest(model)==baseline and all(p.grad is None for p in model.parameters())
    assert sum(len(m._forward_hooks) for m in model.modules())==hooks_before
    for name,s in stats.items():
        n=s['tokens'];numerator=n*s['sum_g2_x2']-s['sum_g2']*s['sum_x2']
        denominator=max(0.,n*s['sum_g2_square']-s['sum_g2']**2)*max(0.,n*s['sum_x2_square']-s['sum_x2']**2)
        s['pearson_g_squared_vs_x_norm_squared']=numerator/denominator**.5 if denominator>0 else None
        s['gradient_norm_mean']=s['gradient_norm_sum']/n if n else 0.
        if not n:s['gradient_norm_min']=0.
        assert s['gradient_calls']==len(loader) and s['tokens']==sum(valid_counts)
    state={n:{mode:sketch.state() for mode,sketch in values.items()} for n,values in sketches.items()}
    h1={n:{'centered_eva_vs_gw':compare(v['eva_centered']['components'][:8],v['gw_uncentered']['components'][:8]),
           'uncentered_activation_vs_gw':compare(v['activation_uncentered']['components'][:8],v['gw_uncentered']['components'][:8])} for n,v in state.items()} if 'eva_centered' in modes else {}
    metadata={'task':context['task'] if context else 'rte','partition':context['partition'] if context else 'full','seed':seed,'max_length':max_length,'micro_batch':micro_batch,'samples':samples,
        'indices':indices,'train_data_sha256':data_sha,'official_validation_used':False,'head_checkpoint':str(directory/'head_warmup.pt'),
        'head_checkpoint_sha256':__import__('hashlib').file_digest((directory/'head_warmup.pt').open('rb'),'sha256').hexdigest(),
        'model_before_after_sha256':baseline,'encoder_and_classifier_unchanged':True,'embedding_requires_grad_calls':embedding_flag['calls'],
        'target_modules':72,'valid_tokens_per_batch':valid_counts,'losses':losses,'seconds':time.perf_counter()-started,
        'forward_hooks_before_after_equal':True,'temporary_tensors_cleared':not pending and not active,
        'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'gradient_scale':'mean loss gradient multiplied by actual microbatch B; per-example loss definition',
        'activation_precision':'FP32; no TF32/autocast','statistics_precision':'FP32 merge SVD; float64 scalar reductions',
        'finite_data_calibration_not_convergence_claim':True,'algorithm':'streaming merge/truncate max16; exact each merge, globally approximate'}
    temporary=destination.with_suffix('.tmp');torch.save({'metadata':metadata,'layers':state,'gradient_statistics':stats,'h1':h1,
          'dimensions':{n:{'d_in':m.in_features,'d_out':m.out_features} for n,m in modules.items()}},temporary);temporary.replace(destination)
    write(destination.with_suffix('.json'),{'metadata':metadata,'gradient_statistics':stats,'h1':h1})
    print('CALIBRATION_COMPLETED',destination,flush=True)
    return destination


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--max-length',type=int,default=128)
    p.add_argument('--samples',type=int,default=64);p.add_argument('--micro-batch',type=int,default=1)
    a=p.parse_args();calibrate(a.seed,a.max_length,a.samples,a.micro_batch)
