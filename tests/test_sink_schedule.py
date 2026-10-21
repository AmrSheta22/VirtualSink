import pytest
import torch
from virtualsink.uptraining.train import lr_multipliers


def rates(step):
    a,b = lr_multipliers(step,1000,100,2e-5,5e-4,300)
    return 2e-5*a,5e-4*b


def test_sink_schedule_peaks_matches_and_decays():
    assert rates(0) == (0.,0.)
    assert rates(100) == pytest.approx((2e-5,5e-4))
    ratios = [rates(s)[1]/rates(s)[0] for s in range(100,1000)]
    assert all(a >= b-1e-12 for a,b in zip(ratios,ratios[1:]))
    for s in range(300,1001):
        assert rates(s)[0] == pytest.approx(rates(s)[1])
    assert rates(1000) == (0.,0.)
    for group in (0,1):
        values = [rates(s)[group] for s in range(100,1001)]
        assert all(a >= b for a,b in zip(values,values[1:]))


def test_schedule_state_resume_preserves_next_rates():
    # Bind group indices explicitly: no late-binding lambdas in the real trainer.
    functions = [lambda s:lr_multipliers(s,1000,100,2e-5,5e-4,300)[0],
                 lambda s:lr_multipliers(s,1000,100,2e-5,5e-4,300)[1]]
    def build():
        o = torch.optim.AdamW([{'params':[torch.nn.Parameter(torch.ones(1))],'lr':2e-5},
                              {'params':[torch.nn.Parameter(torch.ones(1))],'lr':5e-4}])
        return o,torch.optim.lr_scheduler.LambdaLR(o,functions)
    o,s = build()
    for _ in range(247):
        o.step();s.step()
    state_o,state_s = o.state_dict(),s.state_dict()
    fresh,restored = build(); fresh.load_state_dict(state_o);restored.load_state_dict(state_s)
    for _ in range(100):
        o.step();s.step();fresh.step();restored.step()
        assert s.get_last_lr() == restored.get_last_lr()
