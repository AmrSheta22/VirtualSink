from pathlib import Path
import sys
import json
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'hpc'))
from evaluate_pythia_positive_control import policies, label, contrast, summarize


def test_same_checkpoint_comparison_separates_prefix_and_position():
    # A null sink cannot be credited for a prefix benefit caused by changing RoPE.
    means = {}
    for arm,off,prefix,relative in policies():
        means[label((arm,off,prefix,relative))] = 5. - 2.*relative - 1.*(prefix == 4)
    for relative in [False,True]:
        result = contrast(means,relative)
        assert result['standard_prefix_rescue'] == 1.
        for arm in ['legacy','calibrated']:
            assert result[arm]['fifo_sink_enabling_benefit'] == 0.
            assert result[arm]['prefix_rescue_disabled'] == result[arm]['prefix_rescue_enabled'] == 1.


def test_within_checkpoint_gate_rescue_has_correct_direction():
    means = {}
    for arm,off,prefix,relative in policies():
        means[label((arm,off,prefix,relative))] = 3. if prefix or not off else 5.
    value = contrast(means,False)['calibrated']
    assert value['fifo_sink_enabling_benefit'] == 2.
    assert value['prefix_rescue_disabled'] == 2.
    assert value['prefix_rescue_enabled'] == 0.


def test_missing_policy_cannot_yield_a_complete_result(tmp_path):
    summarize(tmp_path)
    value = json.loads((tmp_path/'INCOMPLETE.json').read_text())
    assert not value['complete'] and len(value['missing']) == 24
    assert not (tmp_path/'aggregate.json').exists()


def test_policy_identity_mismatch_is_rejected(tmp_path):
    (tmp_path/'identity.json').write_text('{}')
    folder = tmp_path/'results'/label(policies()[0])
    folder.mkdir(parents=True)
    (folder/'complete.json').write_text(json.dumps(dict(policy=['wrong'],identity_sha256='wrong')))
    with pytest.raises(AssertionError):
        summarize(tmp_path)
