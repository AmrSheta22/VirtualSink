import pytest
from virtualsink.uptraining.evaluate_streaming import CONFIRMATION_TITLES
from virtualsink.uptraining.summarize_streaming_confirmation import paired_effect


def test_confirmation_is_disjoint_and_unique():
    assert len(set(CONFIRMATION_TITLES))==12
    assert '= American Beauty ( 1999 film ) =' not in CONFIRMATION_TITLES
    assert '= 2003 Pacific typhoon season =' not in CONFIRMATION_TITLES


def test_document_pairing_and_direction():
    r=paired_effect([2,4,8],[1.8,3.8,7.8])
    assert r['wins']==3
    assert r['mean_ce_advantage']==pytest.approx(.2)
    assert r['article_bootstrap_95pct_ce_interval']==pytest.approx([.2,.2])
    assert r['sink_to_standard_ppl_ratio']<1
    assert 'supported advantage' in r['conclusion']
    assert 'inconclusive' in paired_effect([1,1,1],[1,1,1])['conclusion']
    with pytest.raises(ValueError):paired_effect([1,2],[1])
