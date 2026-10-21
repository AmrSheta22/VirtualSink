"""Exercise the isolated evaluator against the original frozen SmolLM backend."""
import os
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'hpc'))
from smollm_streaming_decoder import StreamingDecoder
from smollm_rebased_reference import RebasedReference


def frozen_model():
    # Separate package avoids loading the workspace's evolving attention code.
    package = '_eviction_frozen'
    directory = Path(os.environ.get('FROZEN_UPTRAINING_ROOT', str(ROOT / 'artifacts/smollm-warm70-20261014/release/src/virtualsink/uptraining')))
    if package not in sys.modules:
        import types
        module = types.ModuleType(package)
        module.__path__ = [str(directory)]
        sys.modules[package] = module
    from importlib import import_module
    NativeLM = import_module(package + '.native').NativeLM
    torch.manual_seed(9)
    return NativeLM(dict(model_type='llama', hidden_size=24, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=3, num_key_value_heads=1,
        vocab_size=31, hidden_act='silu', rms_norm_eps=1e-5,
        rope_theta=10000., tie_word_embeddings=True)).eval()


@pytest.mark.parametrize('disabled', [False, True])
def test_frozen_window_parity_and_prefix_preservation(disabled):
    model = frozen_model()
    for layer in model.model.layers:
        layer.self_attn.sink_disabled = disabled
        layer.self_attn.window_size = 8
    tokens = torch.arange(25)
    with torch.no_grad():
        expected = model.lm_head(model.model(tokens[None]).last_hidden_state)[0]
    decoder = StreamingDecoder(model, 8)
    for i, token in enumerate(tokens):
        actual, *_ = decoder.step(token)
        torch.testing.assert_close(actual, expected[i], atol=3e-6, rtol=2e-5)
        assert all(k.shape[2] == min(i+1, 8) for k, _ in decoder.cache)
    decoder = StreamingDecoder(model, 8, 4)
    for token in tokens[:4]:
        decoder.step(token)
    first = [(k.clone(), v.clone()) for k, v in decoder.cache]
    for token in tokens[4:]:
        decoder.step(token)
    for (k, v), (pk, pv) in zip(decoder.cache, first):
        torch.testing.assert_close(k[:, :, :4], pk)
        torch.testing.assert_close(v[:, :, :4], pv)


@pytest.mark.parametrize('prefix', [0, 4])
def test_frozen_rebasing_matches_independent_delta_oracle(prefix):
    model = frozen_model()
    for layer in model.model.layers:
        layer.self_attn.sink_disabled = True
    decoder, oracle = StreamingDecoder(model, 8, prefix, True), RebasedReference(model, 8, prefix)
    for token in torch.arange(25):
        actual, *_ = decoder.step(token)
        torch.testing.assert_close(actual, oracle.step(token), atol=4e-6, rtol=2e-5)


def test_policy_design_is_complete_and_balanced():
    from evaluate_smollm_eviction import policies, label
    assert len(policies()) == len(set(map(label, policies()))) == 12
    for rebase in [False, True]:
        for prefix in [0, 4]:
            assert ('query_sink', False, prefix, rebase) in policies()
            assert ('query_sink', True, prefix, rebase) in policies()
            assert ('standard', True, prefix, rebase) in policies()


def test_missing_units_cannot_generate_completed_summary(tmp_path, capsys):
    from evaluate_smollm_eviction import summarize
    (tmp_path / 'identity.json').write_text('{}')
    summarize(tmp_path)
    import json
    result = json.loads((tmp_path / 'INCOMPLETE.json').read_text())
    assert not result['complete'] and len(result['missing']) == 30
    assert not (tmp_path / 'aggregate.json').exists()


def test_summary_rejects_corrupted_policy_identity(tmp_path):
    from evaluate_smollm_eviction import summarize, policies, label
    import json
    (tmp_path / 'identity.json').write_text('{}')
    unit = tmp_path / 'results/seed8-fineweb'
    unit.mkdir(parents=True)
    (unit / 'complete.json').write_text('{}')
    (unit / (label(policies()[0]) + '.json')).write_text(json.dumps(
        dict(identity_sha256='wrong', per_token_ce=[1.] * 4096)))
    with pytest.raises(AssertionError):
        summarize(tmp_path)
    assert not (tmp_path / 'aggregate.json').exists()
