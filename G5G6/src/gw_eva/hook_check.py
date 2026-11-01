"""Real-model graph, batch-scale, padding-alignment and cleanup checks."""
import json
import torch
from .runtime import OUT,model_load,rte_data,targets,digest,write


def main():
    folder=OUT/'rte_seed0_len128';model=model_load(0)
    model.classifier.load_state_dict(torch.load(folder/'head_warmup.pt',weights_only=True));model.eval()
    for p in model.parameters():p.requires_grad_(False)
    data,collator,_=rte_data(128);before=digest(model);device=next(model.parameters()).device
    sample=[data[0],data[1]];plain={k:v.to(device) for k,v in collator(sample).items()}
    output=model(**plain);assert not output.loss.requires_grad
    baseline_counts=sum(len(m._forward_hooks) for m in model.modules())
    embedding=model.roberta.embeddings.register_forward_hook(lambda m,a,o:o.requires_grad_(True))
    def capture(examples):
        batch={k:v.to(device) for k,v in collator(examples).items()};mask=batch['attention_mask'].bool()
        handles=[];grad_handles=[];result={};size=len(examples)
        def make(name):
            def forward(m,args,out):
                assert args[0].shape[:2]==mask.shape and out.shape[:2]==mask.shape
                assert out.requires_grad
                def backward(g):
                    norm=(g.detach().float()*size).norm(dim=-1)
                    result[name]=[norm[i][mask[i]].cpu() for i in range(size)]
                grad_handles.append(out.register_hook(backward))
            return forward
        for name,module in targets(model).items():handles.append(module.register_forward_hook(make(name)))
        try:model(**batch).loss.backward()
        finally:
            for h in handles+grad_handles:h.remove()
        assert len(result)==72
        return result
    try:
        together=capture(sample);separate=[capture([example]) for example in sample]
    finally:embedding.remove()
    errors={}
    for name in together:
        differences=[]
        for i in range(2):
            a=together[name][i];b=separate[i][name][0]
            assert a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
            differences.append(float((a-b).abs().max()))
            assert torch.allclose(a,b,atol=1e-6,rtol=1e-3),name
        errors[name]=max(differences)
    assert digest(model)==before and all(p.grad is None for p in model.parameters())
    assert sum(len(m._forward_hooks) for m in model.modules())==baseline_counts
    report={'status':'passed','frozen_without_embedding_loss_requires_grad':False,'with_embedding_grad_modules':72,
       'batch2_vs_single_loss_scale_undo':'multiply output gradients by actual microbatch B',
       'per_module_max_absolute_norm_error':errors,'max_absolute_norm_error':max(errors.values()),
       'padding_alignment_verified':True,'frozen_weights_unchanged':True,'frozen_parameter_grads_none':True,
       'all_hooks_cleaned':True,'source':'two real RTE training examples; FP32 eval eager attention; no optimizer'}
    write(folder/'gradient_hook_check.json',report);print('REAL_HOOK_CHECK_PASSED',report['max_absolute_norm_error'])


if __name__=='__main__':main()
