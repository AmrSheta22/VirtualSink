"""Document boundaries, token loss accounting, TBPTT and exact resume tests."""

import copy
import json

import numpy as np
import pytest
import torch

from virtualsink.uptraining.data import sha256
from virtualsink.uptraining.recurrent_data import TokenDocuments, prepare_split
from virtualsink.uptraining.train_recurrent import (
    evaluate, load_checkpoint, pack_state, save_checkpoint, token_losses, unpack_state, unroll_loss,
)
from test_recurrent_sink import tiny, assert_state_close


def corpus(tmp_path):
    # Internal EOS-like tokens are intentional: offsets, not token contents,
    # define boundaries. The first document ends part-way through an unroll.
    documents = [[1, 2, 0, 4, 5, 6, 7, 0], [7, 6, 5, 4, 3, 2, 1, 8, 9, 10, 0]]
    tokens = np.asarray([token for doc in documents for token in doc], dtype="<u4")
    token_path, offsets_path = tmp_path / "train.bin", tmp_path / "train.offsets.npy"
    tokens.tofile(token_path)
    np.save(offsets_path, np.asarray([0, 8, 19], dtype=np.int64), allow_pickle=False)
    manifest = {"format": "document-uint32-v1", "eos_token_id": 0, "splits": {"train": {
        "file": token_path.name, "offsets_file": offsets_path.name,
        "tokens": 19, "documents": 2, "sha256": sha256(token_path),
        "offsets_sha256": sha256(offsets_path)}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return TokenDocuments(tmp_path, "train", 23), documents


def test_preparation_keeps_authoritative_offsets(tmp_path):
    class Tokenizer:
        eos_token_id = 0
        def encode(self, text, add_special_tokens=False):
            return [int(x) for x in text.split()]
    path = tmp_path / "source.jsonl"
    path.write_text('{"text":"1 0 2"}\n{"text":"3 4"}\n{"text":"  "}\n')
    entry = prepare_split(path, tmp_path, "train", Tokenizer())
    assert entry["documents"] == 2 and entry["tokens"] == 7
    assert np.load(tmp_path / entry["offsets_file"]).tolist() == [0, 4, 7]


def test_offsets_and_checksums_are_required(tmp_path):
    data, _ = corpus(tmp_path)
    inputs, targets = data.segment(0, 1, 100)
    assert inputs.tolist() == [[2, 0, 4, 5, 6, 7]]
    assert targets.tolist() == [[0, 4, 5, 6, 7, 0]]
    with (tmp_path / "train.bin").open("ab") as out:
        out.write(b"x")
    with pytest.raises(ValueError, match="checksum"):
        TokenDocuments(tmp_path, "train", 23)


def test_segment_loss_counts_boundary_predictions_and_resets_document_memory(tmp_path):
    data, docs = corpus(tmp_path)
    model = tiny()
    actual, count, state, cursor = unroll_loss(model, data, (0, 0), None, 3, 10, {}, "cpu", 2)
    expected = torch.zeros(())
    for doc in docs:
        tokens = torch.tensor(doc)[None]
        result = model(tokens[:, :-1])
        expected += token_losses(model, result.hidden, tokens[:, 1:], 2).sum()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert count == 17 and cursor == (2, 0) and state is None
    report = evaluate(model, data, [0, 1], 3, 100, {}, "cpu")
    assert report["all"]["tokens"] == 17
    assert report["post_eviction"]["tokens"] == 11
    assert report["beyond_10_windows"]["tokens"] == 0
    assert report["beyond_10_windows"]["ce"] is None
    assert report["all"]["ce"] == pytest.approx(actual.item() / 17, rel=1e-6)


def test_detached_checkpoint_resume_matches_next_loss_state_and_gradients(tmp_path):
    data, _ = corpus(tmp_path)
    model = tiny()
    model.parameter_groups("memory")
    for bank in model.memory:
        with torch.no_grad():
            bank.memory_bias.fill_(-1.)
    loss, count, state, cursor = unroll_loss(model, data, (0, 0), None, 3, 2, {}, "cpu")
    assert count == 6 and cursor == (0, 6)
    state = state.detach()
    restored = unpack_state(pack_state(state), "cpu")
    assert_state_close(state, restored)
    resumed = copy.deepcopy(model)
    a, count_a, state_a, cursor_a = unroll_loss(model, data, cursor, state, 3, 2, {}, "cpu")
    b, count_b, state_b, cursor_b = unroll_loss(resumed, data, cursor, restored, 3, 2, {}, "cpu")
    assert count_a == count_b == 4 and cursor_a == cursor_b == (1, 3)
    torch.testing.assert_close(a, b)
    assert_state_close(state_a, state_b)
    a.backward()
    b.backward()
    for (name, p), (_, q) in zip(model.named_parameters(), resumed.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, msg=name)


def test_saved_optimizer_and_stream_resume_match_uninterrupted_update(tmp_path, monkeypatch):
    data, _ = corpus(tmp_path)
    model = tiny()
    groups = model.parameter_groups("memory")
    optimizer = torch.optim.AdamW(groups, foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
    first, count, state, cursor = unroll_loss(model, data, (0, 0), None, 3, 2, {}, "cpu")
    (first / count).backward()
    optimizer.step()
    scheduler.step()
    state = state.detach()
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    path = tmp_path / "resume.pt"
    save_checkpoint(path, model, optimizer, scheduler, state, cursor, 1, 2., count, {"test": True})
    saved = load_checkpoint(path)
    restored = tiny()
    restored_optimizer = torch.optim.AdamW(restored.parameter_groups("memory"), foreach=False)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda step: 1 / (step + 1))
    restored.load_state_dict(saved["model"])
    restored_optimizer.load_state_dict(saved["optimizer"])
    restored_scheduler.load_state_dict(saved["scheduler"])
    restored_state = unpack_state(saved["state"], "cpu")
    for current, opt, sched, stream in ((model, optimizer, scheduler, state),
                                        (restored, restored_optimizer, restored_scheduler, restored_state)):
        opt.zero_grad(set_to_none=True)
        loss, n, _, _ = unroll_loss(current, data, tuple(saved["cursor"]), stream, 3, 2, {}, "cpu")
        (loss / n).backward()
        opt.step()
        sched.step()
    for (name, p), (_, q) in zip(model.named_parameters(), restored.named_parameters()):
        torch.testing.assert_close(p, q, atol=1e-7, rtol=1e-6, msg=name)
    assert optimizer.param_groups[0]["lr"] == restored_optimizer.param_groups[0]["lr"]
