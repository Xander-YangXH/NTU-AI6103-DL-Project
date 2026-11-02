"""Create reusable G5/G6 adapters and actually test forward/backward/save/load."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import torch
from peft import LoraConfig,TaskType,get_peft_model,PeftModel
from .runtime import OUT,model_load,rte_data,targets,write
from .allocation import count_allocate,cost_allocate
from .calibrate import calibrate


def encoder_digest(model):
    if isinstance(model,PeftModel):model=model.get_base_model()
    values={n.replace('.base_layer.','.'):p for n,p in model.roberta.state_dict().items() if 'lora_' not in n}
    value=hashlib.sha256()
    for name in sorted(values):value.update(name.encode());value.update(values[name].detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def attach(model,ranks,layers):
    active=[name for name,rank in ranks.items() if rank]
    # PEFT rho=2 adjust_scaling_factors semantics: keep alpha/r = 1
    # across redistributed ranks (base r=8, alpha=8).
    config=LoraConfig(r=8,lora_alpha=8,lora_dropout=0,task_type=TaskType.SEQ_CLS,
        target_modules=active,modules_to_save=['classifier'],rank_pattern={n:ranks[n] for n in active},
        alpha_pattern={n:ranks[n] for n in active},bias='none')
    model=get_peft_model(model,config)
    modules=dict(model.get_base_model().named_modules());details={}
    with torch.no_grad():
        for name,rank in ranks.items():
            module=modules[name];d_in=layers[name]['d_in'];d_out=layers[name]['d_out']
            if rank:
                A=module.lora_A['default'].weight;B=module.lora_B['default'].weight
                expected=layers[name]['A'][:rank].to(A)
                A.copy_(expected);B.zero_()
                assert A.shape==(rank,d_in) and B.shape==(d_out,rank)
                assert torch.isfinite(A).all() and torch.equal(B,torch.zeros_like(B))
                assert float((A.norm(dim=1)-1).abs().max())<1e-5
                assert float((B@A).abs().max())==0
                actual=A.numel()+B.numel()
            else:actual=0
            assert actual==rank*(d_in+d_out)
            details[name]={'rank':rank,'d_in':d_in,'d_out':d_out,'per_rank_cost':d_in+d_out,'lora_parameters':actual}
    return model,details


def run(calibration):
    value=torch.load(calibration,map_location='cpu',weights_only=False);meta=value['metadata'];root=Path(calibration).parent
    scores={n:v['gw_uncentered']['scores'].tolist() for n,v in value['layers'].items()}
    costs={n:d['d_in']+d['d_out'] for n,d in value['dimensions'].items()}
    ranks5=count_allocate(scores);ranks6,budget=cost_allocate(scores,costs)
    assert sum(ranks5.values())==576
    assert .99<=budget['fraction']<=1
    write(root/'G5_rank_pattern.json',ranks5);write(root/'G6_rank_pattern.json',ranks6)
    write(root/'spectral_scores.json',scores);write(root/'G6_budget.json',budget)
    write(root/'rank_comparison.json',{n:{'G5':ranks5[n],'G6':ranks6[n],'cost':costs[n]} for n in ranks5})
    data,collator,_=rte_data(meta['max_length']);record=collator([data[int(meta['indices'][0])],data[int(meta['indices'][1])]])
    results={}
    for group,ranks in [('G5',ranks5),('G6',ranks6)]:
        started=time.perf_counter();model=model_load(meta['seed'])
        model.classifier.load_state_dict(torch.load(meta['head_checkpoint'],weights_only=True));model.eval()
        head_hash=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.classifier.parameters())).hexdigest()
        before=encoder_digest(model);inputs={k:v.to(next(model.parameters()).device) for k,v in record.items()}
        with torch.no_grad():original=model(**inputs).logits.detach().clone()
        layers={n:{**value['dimensions'][n],'A':v['gw_uncentered']['components']} for n,v in value['layers'].items()}
        model,details=attach(model,ranks,layers);model.eval()
        with torch.no_grad():initial=model(**inputs).logits.detach().clone()
        init_error=float((initial-original).abs().max());assert init_error==0
        assert encoder_digest(model)==before
        folder=root/group;folder.mkdir(exist_ok=True)
        model.save_pretrained(folder/'initial_adapter',safe_serialization=True)
        write(folder/'rank_pattern.json',ranks);write(folder/'parameters.json',details)
        lora=sum(p.numel() for n,p in model.named_parameters() if 'lora_' in n)
        head=sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and 'lora_' not in n)
        assert lora==sum(row['lora_parameters'] for row in details.values())
        # Test a deterministic actual update; no dropout noise in smoke probe.
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=0)
        optimizer.zero_grad(set_to_none=True);loss=model(**inputs).loss;assert torch.isfinite(loss);loss.backward()
        gradients={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
        assert gradients and all(__import__('math').isfinite(x) for x in gradients.values())
        Bgrad=sum(x*x for n,x in gradients.items() if 'lora_B' in n)**.5;assert Bgrad>0
        optimizer.step()
        nonzero_B=sum(int(torch.count_nonzero(p)) for n,p in model.named_parameters() if 'lora_B' in n);assert nonzero_B>0
        assert encoder_digest(model)==before
        with torch.no_grad():trained=model(**inputs).logits.detach().cpu()
        model.save_pretrained(folder/'smoke_adapter',safe_serialization=True)
        assert all(torch.isfinite(p).all() for p in model.parameters() if p.requires_grad)
        del optimizer,model;gc.collect();torch.cuda.empty_cache()
        reload_errors={}
        for checkpoint,expected in [('initial_adapter',original.cpu()),('smoke_adapter',trained)]:
            fresh=model_load(meta['seed']);fresh.classifier.load_state_dict(torch.load(meta['head_checkpoint'],weights_only=True))
            fresh=PeftModel.from_pretrained(fresh,folder/checkpoint);fresh.eval()
            with torch.no_grad():actual=fresh(**inputs).logits.detach().cpu()
            reload_errors[checkpoint]=float((actual-expected).abs().max());assert reload_errors[checkpoint]<1e-6
            assert encoder_digest(fresh)==before
            del fresh;gc.collect();torch.cuda.empty_cache()
        result={'status':'passed','group':group,'target_modules_considered':72,'active_modules':sum(r>0 for r in ranks.values()),
           'rank_sum':sum(ranks.values()),'rank_max':max(ranks.values()),'lora_parameters':lora,'classifier_parameters_separate':head,
           'budget_fraction':lora/1327104,'initial_logits_max_difference':init_error,'encoder_unchanged':True,
           'head_before_adapter_sha256':head_hash,'loss_before_step':float(loss),'B_gradient_frobenius':Bgrad,'B_nonzero_after_step':nonzero_B,
           'reload_errors':reload_errors,'seconds':time.perf_counter()-started,'calibration':str(calibration),
           'alpha_policy':'base r8 alpha8; alpha_i=r_i, scaling=1; no whitening',
           'not_formal_training':True,'gradients':gradients}
        write(folder/'smoke_result.json',result);results[group]=result
        print('SMOKE_PASSED',group,'rank',sum(ranks.values()),'parameters',lora,flush=True)
    assert results['G5']['head_before_adapter_sha256']==results['G6']['head_before_adapter_sha256']
    write(root/'smoke_results.json',results)
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--calibration',type=Path);a=p.parse_args()
    run(a.calibration or calibrate())
