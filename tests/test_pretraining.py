"""CPU checks of initialization, batch equivalence and partial-step recovery."""
import copy
import json
import math
from unittest.mock import patch

import numpy as np
import pytest
import torch

from virtualsink.pretraining.recipe import initialize, shared_hash, schedule, mixture_order, optimizer_groups
from virtualsink.pretraining.data import document_split
from virtualsink.pretraining.train import language_loss, save, restore, measure_attention
from virtualsink.pretraining.compare import compare_runs, paired_difference_bootstrap
from virtualsink.uptraining.qwen import disable_sinks


def tiny():
    return dict(model_type="llama", hidden_size=24, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=3, num_key_value_heads=1,
        max_position_embeddings=16, vocab_size=31, hidden_act="silu",
        rms_norm_eps=1e-5, rope_theta=10000., tie_word_embeddings=True)


def test_shared_initialization_and_ties():
    a, b = initialize(8, 3, tiny()), initialize(8, 3, tiny())
    disable_sinks(b)
    assert shared_hash(a) == shared_hash(b)
    assert shared_hash(a) != shared_hash(initialize(9, 3, tiny()))
    assert a.lm_head.weight is a.model.embed_tokens.weight
    for name, value in a.named_parameters():
        if "norm" in name:
            assert torch.equal(value, torch.ones_like(value))
    assert a.model.layers[0].self_attn.o_proj.weight.std().item() == pytest.approx(.0416 / 2, rel=.15)
    assert a.model.embed_tokens.weight.std().item() == pytest.approx(.0416, rel=.15)


def test_original_and_shortened_schedule():
    assert schedule(0, 600000) == 0
    assert schedule(2500, 600000) == .5
    assert schedule(5000, 600000) == 1
    assert schedule(499999, 600000) == 1
    assert schedule(525000, 600000) == .5
    assert schedule(600000, 600000) == 0
    assert schedule(0, 100) == 0 and schedule(1, 100) == 1
    assert schedule(83, 100) == 1 and schedule(100, 100) == 0


def test_explicit_ten_update_warmup():
    rates = [.003 * schedule(step, 70, warmup_steps=10) for step in range(11)]
    assert rates == pytest.approx([step * .0003 for step in range(11)])
    assert schedule(57, 70, 10) == 1
    assert 0 < schedule(59, 70, 10) < 1
    assert schedule(70, 70, 10) == 0
    with pytest.raises(ValueError):
        schedule(0, 70, 58)


def test_mixture_and_document_partition():
    order, quota = mixture_order(51200)
    assert quota.tolist() == [35840, 7680, 4096, 3072, 512]
    np.testing.assert_array_equal(np.bincount(order), quota)
    np.testing.assert_array_equal(order, mixture_order(51200)[0])
    assert document_split(" duplicate text ") == document_split("duplicate text")
    assert set(document_split(str(i)) for i in range(10000)) == {"train", "test", "validation"}


def test_all_shifted_targets_and_accumulation_equivalence():
    torch.manual_seed(100)
    a = initialize(8, 3, tiny())
    b = copy.deepcopy(a)
    tokens = torch.randint(0, 31, (2, 9))
    loss, _ = language_loss(a, tokens, 3)
    hidden = a.model(tokens[:, :-1]).last_hidden_state
    expected = torch.nn.functional.cross_entropy(a.lm_head(hidden).reshape(-1,31), tokens[:,1:].reshape(-1))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    for row in tokens:
        (language_loss(b, row[None], 2)[0] / 2).backward()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4)


def test_partial_accumulation_resume(tmp_path):
    model = initialize(8, 3, tiny())
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(optimizer_groups(model, "query_sink"), lr=.003, betas=(.9,.95), foreach=False)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: schedule(s, 100))
    tokens = torch.randint(0,31,(2,9))
    # Populate Adam moments before testing a partial later optimizer step.
    language_loss(model,tokens,3)[0].backward()
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    (language_loss(model,tokens[:1],3)[0]/2).backward()
    path = tmp_path / "partial.pt"
    config = {"test": True}
    with patch.object(torch.cuda,"get_rng_state_all",return_value=[]):
        save(path,model,opt,sched,dict(step=1,micro=1),config)
    ref = copy.deepcopy(model)
    # deepcopy does not copy .grad; retain the accumulated gradients explicitly.
    for p,q in zip(model.parameters(),ref.parameters()):
        q.grad = None if p.grad is None else p.grad.clone()
    refopt=torch.optim.AdamW(optimizer_groups(ref,"query_sink"),lr=.003,betas=(.9,.95),foreach=False)
    refopt.load_state_dict(copy.deepcopy(opt.state_dict()))
    new = initialize(9,3,tiny())
    newopt = torch.optim.AdamW(optimizer_groups(new,"query_sink"), lr=.003,betas=(.9,.95),foreach=False)
    newsched = torch.optim.lr_scheduler.LambdaLR(newopt, lambda s:schedule(s,100))
    with patch.object(torch.cuda,"set_rng_state_all"):
        progress=restore(path,new,newopt,newsched,config)
    assert progress == dict(step=1,micro=1)
    for m in (ref,new):
        (language_loss(m,tokens[1:],3)[0]/2).backward()
    for p,q in zip(ref.parameters(),new.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-6)
    refopt.step(); newopt.step()
    for p,q in zip(ref.parameters(),new.parameters()):
        torch.testing.assert_close(p,q,rtol=1e-5,atol=1e-6)


def test_diagnostics_are_joint_probabilities():
    model=initialize(8,3,tiny()).eval()
    class Data:
        length=8
        def __len__(self): return 1
        def __getitem__(self,i): return torch.arange(9)
    stats=measure_attention(model,Data(),1,"cpu")
    values=np.array(stats["by_layer_head"])
    assert (values[:,:,2]>0).all()
    disable_sinks(model)
    base=np.array(measure_attention(model,Data(),1,"cpu")["by_layer_head"])
    assert (base[:,:,2]==0).all()
    assert (values[:,:,0]<base[:,:,0]).all()


def test_window_diagnostics_and_stronger_initialization():
    from virtualsink.uptraining.native import set_window
    model = initialize(8, 3, tiny(), sink_weight_std=.02, sink_bias=4.).eval()
    assert shared_hash(model) == shared_hash(initialize(8, 3, tiny()))
    assert model.model.layers[0].self_attn.sink_b.tolist() == [4.] * 3
    assert model.model.layers[0].self_attn.sink_w.std() > .01
    set_window(model, 2)
    class Data:
        length = 8
        def __len__(self): return 1
        def __getitem__(self, i): return torch.arange(9)
    values = np.array(measure_attention(model, Data(), 1, "cpu")["by_layer_head"])
    assert (values[:, :, 0] == 0).all()  # Position zero is outside the late-query window.
    assert (values[:, :, 5] > 0).all()
    np.testing.assert_allclose(values[:, :, 6] + values[:, :, 2], 1., atol=1e-6)


def test_comparison_script(tmp_path):
    std_dir = tmp_path / "runs" / "standard"
    sink_dir = tmp_path / "runs" / "query_sink"
    out_dir = tmp_path / "comparison"
    std_dir.mkdir(parents=True)
    sink_dir.mkdir(parents=True)

    config_std = {"variant": "standard", "shared_initial_weights_sha256": "deadbeef1234"}
    config_sink = {"variant": "query_sink", "shared_initial_weights_sha256": "deadbeef1234"}
    common = dict(manifest_sha256="data-hash", steps=30, model_seed=8, data_seed=42,
                  initialization="random", global_batch=512, sequence_length=2048,
                  auxiliary="none", gamma=0., precision="fp32", lr=.003, betas=[.9,.95],
                  epsilon=1e-8, weight_decay=.01, clip_grad=1., torch="1.12.1",
                  source_sha256={"train.py": "source-hash"}, recipe="recipe-url",
                  microbatch=1, chunk_size=32, eval_blocks=16, eval_every=10)
    config_std.update(common)
    config_sink.update(common)
    (std_dir / "config.json").write_text(json.dumps(config_std))
    (sink_dir / "config.json").write_text(json.dumps(config_sink))

    status = {"status": "complete", "step": 30}
    (std_dir / "status.json").write_text(json.dumps(status))
    (sink_dir / "status.json").write_text(json.dumps(status))

    summary_std = {
        "step": 30,
        "final_test": {
            "ce": 3.5, "perplexity": 33.1,
            "per_block_ce": [3.4, 3.6, 3.5, 3.5],
            "physical_gate_by_layer_head": [[1.0, 1.0], [0.95, 0.85]]
        },
        "attention": {"by_layer_head": [[[0.2, 0.4, 0.0, -10.0, 0.5, 1.0], [0.2, 0.4, 0.0, -10.0, 0.5, 1.0]]]}
    }
    summary_sink = {
        "step": 30,
        "final_test": {
            "ce": 3.2, "perplexity": 24.5,
            "per_block_ce": [3.1, 3.3, 3.2, 3.2],
            "physical_gate_by_layer_head": [[0.98, 0.97], [0.92, 0.91]]
        },
        "final_test_sinks_disabled": {"ce": 3.45, "perplexity": 31.5},
        "disabled_minus_enabled_ce": 0.25,
        "attention": {"by_layer_head": [[[0.1, 0.2, 0.05, -3.0, 0.5, 0.98], [0.1, 0.2, 0.06, -2.8, 0.5, 0.97]]]}
    }
    (std_dir / "summary.json").write_text(json.dumps(summary_std))
    (sink_dir / "summary.json").write_text(json.dumps(summary_sink))

    report = compare_runs(std_dir, sink_dir, out_dir)
    assert report["verified_initial_hash"] == "deadbeef1234"
    assert report["test_comparison"]["ce_reduction"] == pytest.approx(0.3)
    assert report["disabled_intervention"]["disabled_minus_enabled_ce"] == pytest.approx(0.25)
    assert report["gate_statistics"]["standard_heads_under_90pct"] == 1
    assert report["gate_statistics"]["query_sink_heads_under_90pct"] == 0
    assert (out_dir / "comparison.json").exists()
    assert (out_dir / "comparison.md").exists()

    for key in ("manifest_sha256", "lr", "steps", "source_sha256"):
        changed = dict(config_sink, **{key: "mismatch"})
        (sink_dir / "config.json").write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="configuration mismatch"):
            compare_runs(std_dir, sink_dir)
    (sink_dir / "config.json").write_text(json.dumps(config_sink))
    for directory, filename, original in ((sink_dir, "status.json", status),
                                           (sink_dir, "summary.json", summary_sink)):
        (directory / filename).write_text(json.dumps(dict(original, step=29)))
        with pytest.raises(ValueError, match="Completed steps"):
            compare_runs(std_dir, sink_dir)
        (directory / filename).write_text(json.dumps(original))
    for directory, config in ((std_dir, config_std), (sink_dir, config_sink)):
        (directory / "config.json").write_text(json.dumps({k:v for k,v in config.items() if k != "shared_initial_weights_sha256"}))
    with pytest.raises(ValueError, match="Initial shared weight"):
        compare_runs(std_dir, sink_dir)
    (std_dir / "config.json").write_text(json.dumps(config_std))

    # Verify hash mismatch assertion
    (sink_dir / "config.json").write_text(json.dumps({"variant": "query_sink", "shared_initial_weights_sha256": "mismatch"}))
    with pytest.raises(ValueError, match="Initial shared weight SHA256 mismatch"):
        compare_runs(std_dir, sink_dir, out_dir)


@pytest.mark.parametrize("a,b", [([], []), ([1,2], [1]), ([float("nan")], [1]), ([[1]], [[1]])])
def test_bootstrap_rejects_invalid_pairs(a, b):
    with pytest.raises(ValueError, match="Paired test losses"):
        paired_difference_bootstrap(a, b)
