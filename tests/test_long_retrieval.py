import copy
import re
import torch
from test_recurrent_sink import tiny,assert_state_close
from virtualsink.uptraining.long_retrieval import build_dataset,payload_sha256,score_logits,score_rows,clear_memory


class FakeTokenizer:
    def __init__(self):self.vocab={}
    def encode(self,text):
        pieces=re.findall(r' ?[A-Za-z]+|[0-9]{1,3}|[^A-Za-z0-9]',text)
        return [self.vocab.setdefault(piece,len(self.vocab)) for piece in pieces]


def test_counterfactual_balance_distances_and_no_answer_leak():
    dataset=build_dataset(FakeTokenizer().encode)
    assert payload_sha256(dataset)==payload_sha256(build_dataset(FakeTokenizer().encode))
    assert len(dataset['blocks'])==8 and dataset['cases_per_gap']==64
    for block in dataset['blocks']:
        assert len(block['background'])==16384
        label=block['label_position']
        for truth,prefix in enumerate(block['prefixes']):
            assert prefix[label]==dataset['candidate_token_ids'][truth]
            assert [j for j,(a,b) in enumerate(zip(prefix,block['missing_prefix'])) if a!=b]==[label]
            assert all(a==b for j,(a,b) in enumerate(zip(prefix,block['prefixes'][0])) if j!=label)
        for gap in dataset['gaps']:
            assert block['actual_prediction_distances'][str(gap)]==len(block['prefixes'][0])+gap+len(block['query'])-1-label
        assert block['actual_prediction_distances']['0']<128
        assert block['actual_prediction_distances']['256']>128


def test_balanced_missing_answer_control_scores_at_chance():
    dataset=build_dataset(FakeTokenizer().encode)
    ids=dataset['candidate_token_ids'];logits=torch.zeros(1,max(ids)+10)
    logits[0,ids[2]]=4
    scored=score_logits(logits,ids)
    rows=score_rows(scored,dataset['blocks'][:1],dataset,4096,'missing_answer',0.)
    assert len(rows)==8 and sum(r['choice_correct'] for r in rows)==1
    assert sum(r['greedy_correct'] for r in rows)==1
    assert all(r['actual_prediction_distance'] is None for r in rows)
    logits[0,-1]=100
    rows=score_rows(score_logits(logits,ids),dataset['blocks'][:1],dataset,0,'missing_answer',0.)
    assert sum(r['choice_correct'] for r in rows)==1
    assert not any(r['greedy_correct'] for r in rows)


def test_query_and_clear_branches_leave_carried_prefix_unchanged():
    model=tiny().eval();torch.manual_seed(9)
    prefix=torch.randint(0,23,(2,9));query=torch.randint(0,23,(2,5));tail=torch.randint(0,23,(2,7))
    with torch.no_grad():
        saved=model(prefix).state;snapshot=copy.deepcopy(saved)
        cleared=clear_memory(saved)
        assert all(s.cell.count_nonzero()==0 and s.exposed.count_nonzero()==0 for s in cleared.layers)
        for a,b in zip(saved.layers,cleared.layers):
            assert a.key.data_ptr()==b.key.data_ptr() and a.value.data_ptr()==b.value.data_ptr()
        model(query,saved);model(query,cleared)
        actual=model(tail,saved);expected=model(torch.cat((prefix,tail),1))
        assert_state_close(saved,snapshot)
        assert_state_close(actual.state,expected.state)
        torch.testing.assert_close(actual.hidden,expected.hidden[:,-7:],atol=3e-6,rtol=3e-5)
