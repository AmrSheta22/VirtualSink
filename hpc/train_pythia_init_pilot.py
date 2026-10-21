"""Controlled Pythia410M pilot: standard, legacy sink init, nearly identity sink init."""
import argparse,json,math,os,time,types
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM


def attend(self,q,k,v,attention_mask=None,head_mask=None):
    n,m=q.shape[-2],k.shape[-2]
    scores=(q.float()@k.float().transpose(-1,-2))*self.norm_factor
    pos=torch.arange(m,device=q.device);mask=pos[None,:]<=pos[m-n:,None]
    scores=scores.masked_fill(~mask,-torch.inf)
    if attention_mask is not None:scores=scores+attention_mask
    lse=scores.logsumexp(-1)
    if self.calibrating:
        # Minimum over representative training queries bounds their initial sink mass.
        value=lse.detach().amin((0,2))
        self.calibration_lse=torch.minimum(self.calibration_lse,value)
    if self.sink_disabled:
        weights=scores.softmax(-1);gate=torch.ones_like(lse)
    else:
        logits=(q.float()*self.sink_w[None,:,None]).sum(-1)+self.sink_b[None,:,None]
        gate=torch.sigmoid(lse-logits)
        weights=torch.cat((scores,logits[...,None]),-1).softmax(-1)[...,:m]
    return weights.to(v.dtype)@v,gate


def adapt(model,arm):
    for l in model.gpt_neox.layers:
        a=l.attention
        a.sink_w=torch.nn.Parameter(torch.zeros(a.num_attention_heads,a.head_size))
        a.sink_b=torch.nn.Parameter(torch.full((a.num_attention_heads,),-3.5))
        if arm=='legacy':torch.nn.init.normal_(a.sink_w,std=1e-4)
        a.sink_disabled=arm=='standard';a.calibrating=False
        a.register_buffer('calibration_lse',torch.full((a.num_attention_heads,),torch.inf),persistent=False)
        a._attn=types.MethodType(attend,a)
    for name,param in model.named_parameters():
        sink=name.endswith(('.sink_w','.sink_b'))
        attention='.attention.' in name and any(p in name for p in ['query_key_value','dense'])
        param.requires_grad_(attention or (sink and arm!='standard'))


def hidden_loss(model,tokens):
    h=model.gpt_neox(tokens,use_cache=False).last_hidden_state[:,:-1]
    total=h.new_zeros(())
    for i in range(0,h.shape[1],32):
        logits=model.embed_out(h[:,i:i+32])
        total+=torch.nn.functional.cross_entropy(logits.reshape(-1,model.config.vocab_size),tokens[:,i+1:i+33].reshape(-1),reduction='sum')
    return total/(tokens.shape[0]*(tokens.shape[1]-1))


@torch.no_grad()
def validation(model,data):
    model.eval();return sum(hidden_loss(model,torch.tensor(data[i*512:(i+1)*512].copy(),device='cuda')[None]).item() for i in range(32))/32


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',choices=['standard','legacy','calibrated'],required=True)
    p.add_argument('--save-final',action='store_true',help='Save matched final-step updates even when validation does not improve.')
    a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    root=a.root;out=root/'pilot'/a.arm;out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((root/'manifest.json').read_text());torch.manual_seed(1729)
    model=AutoModelForCausalLM.from_pretrained(manifest['models']['pythia410']['path'],torch_dtype=torch.float32)
    adapt(model,a.arm);model.cuda()
    train=np.memmap(root/'data/pythia410-train.bin',dtype='<i8',mode='r')
    val=np.memmap(root/'data/pythia410-validation.bin',dtype='<i8',mode='r')
    initial={}
    model.eval()
    tokens=torch.tensor(train[:512].copy(),device='cuda')[None]
    if a.arm=='calibrated':
        for l in model.gpt_neox.layers:l.attention.sink_disabled=True;l.attention.calibrating=True
        with torch.no_grad():
            reference=model(tokens,use_cache=False).logits
            for i in range(4):model.gpt_neox(torch.tensor(train[i*512:(i+1)*512].copy(),device='cuda')[None],use_cache=False)
            for l in model.gpt_neox.layers:
                att=l.attention;att.calibrating=False
                att.sink_b.copy_(att.calibration_lse+math.log(1e-5/(1-1e-5)));att.sink_disabled=False
            actual=model(tokens,use_cache=False).logits
            relative=(actual-reference).norm()/reference.norm()
            assert relative<.001, 'initialization must preserve pretrained logits closely'
            initial={'target_sink_mass':1e-5,'relative_logit_change':relative.item(),
                     'biases':[l.attention.sink_b.tolist() for l in model.gpt_neox.layers]}
            del actual,reference
    initial['validation_ce']=validation(model,val)
    (out/'initial.json').write_text(json.dumps(initial,indent=2))
    groups=[{'params':[p for n,p in model.named_parameters() if p.requires_grad and '.sink_' not in n],'lr':2e-5,'weight_decay':.01}]
    if a.arm!='standard':groups.append({'params':[p for n,p in model.named_parameters() if p.requires_grad and '.sink_' in n],'lr':1e-4,'weight_decay':0.})
    optimizer=torch.optim.AdamW(groups,foreach=False)
    model.gradient_checkpointing_enable();model.enable_input_require_grads()
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min(s/25,1.) if s<25 else .5*(1+math.cos(math.pi*(s-25)/231)))
    best=initial['validation_ce'];best_step=0;started=time.monotonic()
    with (out/'metrics.jsonl').open('w') as log:
        for step in range(256):
            model.train();optimizer.zero_grad(set_to_none=True);total=0.
            for micro in range(8):
                i=(step*8+micro)*512;ids=torch.tensor(train[i:i+512].copy(),device='cuda')[None]
                loss=hidden_loss(model,ids);assert torch.isfinite(loss);(loss/8).backward();total+=loss.item()/8
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();scheduler.step()
            row={'step':step+1,'lm_ce':total,'elapsed_seconds':time.monotonic()-started}
            if (step+1)%32==0:
                ce=validation(model,val);row['validation_ce']=ce
                if ce<best:
                    best,best_step=ce,step+1
                    # Store only the changed projection/sink parameters; the immutable pretrained snapshot supplies all others.
                    torch.save({n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},out/'best-updates.pt')
            log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
    if a.save_final:
        torch.save({n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},out/'final-updates.pt')
    (out/'summary.json').write_text(json.dumps({'arm':a.arm,'steps':256,'input_tokens':1048576,'best_validation_ce':best,'best_step':best_step,'initial':initial,'elapsed_seconds':time.monotonic()-started},indent=2))


if __name__=='__main__':main()
