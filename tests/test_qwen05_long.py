import copy
import pytest
import torch
from test_recurrent_sink import tiny, assert_state_close
from test_recurrent_training import corpus
from virtualsink.uptraining.qwen05_long import train_pair, save, make_optimizer, set_rates
from virtualsink.uptraining.train_recurrent import load_checkpoint, unpack_state

SETTINGS={'segment':3,'unroll_segments':2,'loss_chunk':2}


@pytest.mark.parametrize('adapting',[False,True])
def test_dual_trajectories_resume_exactly(tmp_path,monkeypatch,adapting):
    data,_=corpus(tmp_path)
    model=tiny();opt=make_optimizer(model,'full_block_teacher',adapting)
    losses,n,states,cursor=train_pair(model,data,(0,0),(None,None),'full_block_teacher','cpu',SETTINGS)
    opt.step();opt.zero_grad(set_to_none=True)
    assert states[0].position==states[1].position==6
    assert not torch.allclose(states[0].layers[1].key,states[1].layers[1].key)
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    path=tmp_path/'resume.pt'
    save(path,model,opt,1,cursor,states,n,adapting,{'test':True},4.,'digest')
    saved=load_checkpoint(path)
    restored=copy.deepcopy(model);restored.load_state_dict(saved['model'])
    other_opt=make_optimizer(restored,'full_block_teacher',adapting);other_opt.load_state_dict(saved['optimizer'])
    recovered=tuple(unpack_state(s,'cpu') for s in saved['states'])
    for a,b in zip(states,recovered):assert_state_close(a,b)
    results=[]
    for current,optimizer,state in ((model,opt,states),(restored,other_opt,recovered)):
        set_rates(optimizer,129 if adapting else 1,adapting)
        result=train_pair(current,data,cursor,state,'full_block_teacher','cpu',SETTINGS)
        optimizer.step();results.append(result)
    assert results[0][:2]==results[1][:2] and results[0][3]==results[1][3]
    for p,q in zip(model.parameters(),restored.parameters()):
        torch.testing.assert_close(p,q,atol=1e-7,rtol=1e-6)


def test_fifo_freezes_during_memory_warmup_then_updates_only_backbone(tmp_path):
    model=tiny();assert make_optimizer(model,'fifo',False) is None
    assert not any(p.requires_grad for p in model.parameters())
    opt=make_optimizer(model,'fifo',True)
    assert opt is not None and all(not p.requires_grad for p in model.memory.parameters())
    data,_=corpus(tmp_path)
    losses,n,states,cursor=train_pair(model,data,(0,0),(None,None),'fifo','cpu',SETTINGS)
    assert losses[0]==losses[1]
    assert model.base.model.layers[0].self_attn.q_proj.weight.grad.norm()>0
    assert all(p.grad is None for p in model.memory.parameters())
