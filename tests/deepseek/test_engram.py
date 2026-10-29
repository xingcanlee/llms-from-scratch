import torch

from llms_from_scratch.deepseek.engram import (
    DEAD,
    Engram,
    EngramLayout,
    hash_multipliers,
    is_prime,
    ngram_hash_ids,
)


def _setup():
    layout = EngramLayout.build(n_layers=2, max_ngram_size=4, n_heads=2, vocab_size=50)
    mult = hash_multipliers((1, 14), 4, compressed_vocab=40)
    return layout, mult


def test_primes_are_distinct_and_ranges_disjoint():
    layout, _ = _setup()
    flat = [p for layer in layout.primes for ngram in layer for p in ngram]
    assert len(set(flat)) == len(flat) and all(is_prime(p) and p >= 50 for p in flat)
    assert layout.num_embeddings(0) == sum(flat[:6])


def test_hash_matches_bruteforce():
    layout, mult = _setup()
    tokens = torch.tensor([[3, 7, 7, 11, 0, 5]])
    ids = ngram_hash_ids(tokens, layout, mult, pad_id=1)
    t, layer = 5, 1
    ctx = [5, 0, 11, 7]  # 位置 5 往回看：自身、前 1、前 2、前 3
    m = mult[layer].tolist()
    rolling, cols, offset = ctx[0] * m[0], [], 0
    for n in range(1, 4):  # 2-gram、3-gram、4-gram
        rolling ^= ctx[n] * m[n]
        for p in layout.primes[layer][n - 1]:
            cols.append(rolling % p + offset)
            offset += p
    assert ids[0, t, layer].tolist() == cols


def test_ngrams_never_cross_dead_tokens_or_sequence_start():
    layout, mult = _setup()
    a = torch.tensor([[4, 9, DEAD, 6, 8]])
    b = torch.tensor([[2, 2, DEAD, 6, 8]])  # 断点之前的内容不同
    ia, ib = (ngram_hash_ids(x, layout, mult, pad_id=1) for x in (a, b))
    torch.testing.assert_close(ia[:, 3:], ib[:, 3:])
    c = torch.tensor([[6, 8]])  # 从序列开头起算，和断点之后看到的 n-gram 完全相同
    torch.testing.assert_close(ngram_hash_ids(c, layout, mult, pad_id=1), ia[:, 3:])


def test_engram_gate_and_mask():
    torch.manual_seed(0)
    layout, mult = _setup()
    ids = ngram_hash_ids(torch.randint(0, 40, (2, 5)), layout, mult, pad_id=1)[:, :, 0]
    eng = Engram(layout.num_embeddings(0), ids.size(-1), head_dim=8, dim=16, hc_mult=4)
    x = torch.randn(2, 5, 4, 16)
    y = eng(x, ids)
    delta = y - x  # 4 条流写入同一个值，只是门的大小（且门为正）不同：方向完全一致
    cos = torch.nn.functional.cosine_similarity(delta, delta[..., :1, :], dim=-1)
    torch.testing.assert_close(cos, torch.ones_like(cos), atol=1e-4, rtol=0)
    mask = torch.tensor([[True, False, True, True, True]] * 2)
    torch.testing.assert_close(eng(x, ids, mask)[:, 1], x[:, 1])
