"""Evaluation-only GPTNeoX decoder, including StreamingLLM raw-key positional shifting."""
import math
import torch


class NeoXDecoder:
    def __init__(self,model,capacity=1024,prefix=0,rebase=False):
        if not 0<=prefix<capacity:raise ValueError('invalid cache budget')
        self.model=model.eval();self.capacity=capacity;self.prefix=prefix;self.rebase=rebase
        self.cache=[None]*len(model.gpt_neox.layers);self.position=0
        device=model.gpt_neox.embed_in.weight.device
        # Preserve the backend's precomputed RoPE values (including rounding),
        # rather than regenerating them with different GPU trigonometry.
        self.rope_tables=[(l.attention.rotary_emb.cos_cached.to(device),
                           l.attention.rotary_emb.sin_cached.to(device)) for l in model.gpt_neox.layers]

    @torch.no_grad()
    def step(self,token):
        x=self.model.gpt_neox.embed_in(token.reshape(1,1))
        for i,l in enumerate(self.model.gpt_neox.layers):
            a=l.attention;z=l.input_layernorm(x)
            qkv=a.query_key_value(z).reshape(1,1,a.num_attention_heads,3*a.head_size).permute(0,2,1,3)
            q,k,v=qkv.chunk(3,-1)
            def rotate(t,pos):
                d=a.rotary_ndims
                cos,sin=self.rope_tables[i]
                if int(pos.max())<cos.shape[2]:
                    c,s=cos.index_select(2,pos.long()),sin.index_select(2,pos.long())
                else:
                    inv=a.rotary_emb.inv_freq.to(x.device)
                    phase=pos.to(inv.dtype).reshape(-1,1)*inv[None]
                    phase=torch.cat((phase,phase),-1)[None,None];c,s=phase.cos(),phase.sin()
                rot=t[...,:d];left,right=rot.chunk(2,-1)
                rot=rot*c+torch.cat((-right,left),-1)*s
                return torch.cat((rot,t[...,d:]),-1)
            if not self.rebase:
                q=rotate(q,torch.tensor([self.position],device=x.device))
                k=rotate(k,torch.tensor([self.position],device=x.device))
            if self.cache[i] is not None:
                k=torch.cat((self.cache[i][0],k),2);v=torch.cat((self.cache[i][1],v),2)
            # Official evaluator trims AFTER prediction: transient attention has capacity+1 entries.
            if self.rebase:
                q=rotate(q,torch.tensor([k.shape[2]-1],device=x.device))
                keys=rotate(k,torch.arange(k.shape[2],device=x.device))
            else:keys=k
            # Stock _attn uses fused baddbmm scaling and its original softmax
            # dtype. Matching those operations is necessary for FP16 parity.
            attended,_=a._attn(q,keys,v,attention_mask=None,head_mask=None)
            out=a.dense(attended.transpose(1,2).reshape(1,1,-1))
            if self.model.config.use_parallel_residual:
                x=l.mlp(l.post_attention_layernorm(x))+out+x
            else:
                x=out+x;x=l.mlp(l.post_attention_layernorm(x))+x
            if k.shape[2]>self.capacity:
                recent=self.capacity-self.prefix
                k=torch.cat((k[:,:,:self.prefix],k[:,:,-recent:]),2)
                v=torch.cat((v[:,:,:self.prefix],v[:,:,-recent:]),2)
            self.cache[i]=(k,v)
        self.position+=1
        return self.model.embed_out(self.model.gpt_neox.final_layer_norm(x))[0,0]
