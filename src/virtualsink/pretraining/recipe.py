"""Public SmolLM1-135M recipe, with explicitly shortened schedules.

Source: huggingface/smollm@a041759883ec7152d18fb985ea49be641a0bceef,
text/pretraining/smollm1/config_smollm1_135M.yaml. Initialization follows
release-era nanotron@59bfb6b38c4487b04e425d8540b6e44b2a7fbcf9.
"""
import hashlib
import math

import numpy as np
import torch

from ..uptraining.native import NativeLM

MODEL_CONFIG = dict(model_type="llama", hidden_size=576, intermediate_size=1536,
    num_hidden_layers=30, num_attention_heads=9, num_key_value_heads=3,
    max_position_embeddings=2048, vocab_size=49152, hidden_act="silu",
    rms_norm_eps=1e-5, rope_theta=10000., tie_word_embeddings=True,
    attention_bias=False, attention_dropout=0., bos_token_id=0, eos_token_id=0,
    initializer_range=.02, pretraining_tp=1)
WEIGHTS = dict(fineweb=.70, cosmopedia=.15, python=.08, math=.06, stackoverflow=.01)
RECIPE_URL = "https://github.com/huggingface/smollm/blob/a041759883ec7152d18fb985ea49be641a0bceef/text/pretraining/smollm1/config_smollm1_135M.yaml"


def schedule(step, total):
    """Nanotron LambdaLR convention: step=0 precedes the first update.

Scale YAML's 5k warmup / 500k decay start / 100k decay across 600k steps.
The YAML cooldown fraction is 1/6, despite the blog's rounded 20% description.
"""
    warmup = max(1, round(total * 5000 / 600000))
    decay_start = round(total * 500000 / 600000)
    if step <= warmup:
        return step / warmup
    if step < decay_start:
        return 1.
    return max(0., 1 - math.sqrt(min(1., (step - decay_start) / max(1, total - decay_start))))


def initialize(seed=8, chunk_size=64, config=None, sink_weight_std=1e-4, sink_bias=-3.5):
    """Initialize all shared tensors anew; never read pretrained weights.

Separate RNG streams prevent sink insertion from shifting shared draws.
The paired arms use the SAME native constructor, then disable sinks in control.
"""
    torch.manual_seed(seed)
    model = NativeLM(dict(MODEL_CONFIG if config is None else config), chunk_size)
    generator = torch.Generator().manual_seed(seed)
    sink_generator = torch.Generator().manual_seed(seed + 1000003)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("sink_w"):
                param.normal_(0., sink_weight_std, generator=sink_generator)
            elif name.endswith("sink_b"):
                param.fill_(sink_bias)
            elif name.endswith("bias"):
                param.zero_()
            elif "norm" in name:
                param.fill_(1.)
            else:
                std = .0416
                if name.endswith(("o_proj.weight", "down_proj.weight")):
                    std /= math.sqrt(2 * model.config.num_hidden_layers)
                param.normal_(0., std, generator=generator)
    return model


def shared_hash(model):
    digest = hashlib.sha256()
    for name, param in model.named_parameters():
        if not name.endswith(("sink_w", "sink_b")):
            digest.update(name.encode())
            digest.update(param.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def mixture_order(count, seed=42):
    """Exact rounded block quotas, shuffled once identically for both arms."""
    weights = np.array(list(WEIGHTS.values()))
    quota = np.floor(weights * count).astype(np.int64)
    for i in np.argsort(-(weights * count - quota))[:count - quota.sum()]:
        quota[i] += 1
    order = np.repeat(np.arange(len(weights)), quota)
    np.random.RandomState(seed).shuffle(order)
    return order, quota


def optimizer_groups(model, variant):
    parameters = []
    for name, param in model.named_parameters():
        sink = name.endswith(("sink_w", "sink_b"))
        param.requires_grad_(not (variant == "standard" and sink))
        if not param.requires_grad:
            continue
        parameters.append(param)
    # Release-era Llama inherits an EMPTY weight-decay exclusion list. Do not
    # silently substitute the common modern convention of excluding norms.
    return [dict(params=parameters, weight_decay=.01)]
