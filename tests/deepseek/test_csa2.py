import copy

import torch

from llms_from_scratch.deepseek.csa2 import (
    CSA2Config,
    CSA2Model,
    exact_replay_length,
    layer_modes,
    select_candidate_blocks,
)


def test_official_v41_flash_mode_assignment():
    ratios = [0, 0] + [2] * 18 + [1] * 20
    modes = layer_modes(ratios, [2, 8, 14, 20], [2, 8, 14, 20, 24, 28, 32, 36])
    assert modes[:2] == ["swa", "swa"]
    assert [i for i, m in enumerate(modes) if m == "full"] == [2, 8, 14, 20]
    assert [i for i, m in enumerate(modes) if m == "reindex"] == [24, 28, 32, 36]
    assert modes.count("reuse") == 40 - 2 - 4 - 4  # 编码器 3×5 + 解码器 5×3 = 30


def _model(**kw):
    torch.manual_seed(0)
    return CSA2Model(CSA2Config(**kw)).eval()


@torch.no_grad()
def test_prefill_equals_token_by_token_decode():
    for kw in ({}, {"candidate_source_layer": 4}):
        model = _model(**kw)
        tokens = torch.randint(0, 64, (2, 23))
        full = model(tokens)
        for prefill in (1, 6, 11):
            outs = [model(tokens[:, :prefill], 0)]
            outs += [model(tokens[:, t:t + 1], t) for t in range(prefill, 23)]
            torch.testing.assert_close(torch.cat(outs, 1), full, atol=1e-5, rtol=1e-4)


@torch.no_grad()
def test_reuse_layer_attends_with_source_indices():
    model = _model()
    tokens = torch.randint(0, 64, (1, 16))
    model(tokens)
    # 层 7 是 Reuse：它没有索引器与压缩器，读取的正是层 6（Reindex）最后写入的索引
    assert model.layers[7].attn.indexer is None and model.layers[7].attn.compressor is None
    assert model.layers[5].attn.mode == "reuse" and model.layers[6].attn.mode == "reindex"


def test_candidate_pool_restricts_but_keeps_newest_block():
    logits = torch.tensor([[5.0, 4.0, 0.0, 0.0, 1.0, float("-inf")]])
    keep = select_candidate_blocks(logits, torch.tensor([[5]]), topk_blocks=1, block_size=2)
    # 只保留 1 个块：最新位置所在的块（位置 4..5）被强制保留，而不是分数最高的块 0
    assert keep.tolist() == [[False, False, False, False, True, True]]


@torch.no_grad()
def test_candidate_pool_covering_everything_changes_nothing():
    plain = _model()
    hier = _model(candidate_source_layer=4, candidate_topk_blocks=64)
    tokens = torch.randint(0, 64, (2, 20))
    torch.testing.assert_close(hier(tokens), plain(tokens))


@torch.no_grad()
def test_ced_bounded_replay_exact_and_approximate():
    model = _model()
    tokens = torch.randint(0, 64, (2, 30))
    reference = copy.deepcopy(model)(tokens)[:, -1]
    cfg = model.cfg
    n_dec = cfg.n_layers - cfg.n_encoder_layers
    exact = exact_replay_length(n_dec, cfg.window)
    logits, processed = model.prefill_ced(tokens, replay=exact)
    assert processed == exact < tokens.size(1)
    torch.testing.assert_close(logits, reference, atol=1e-5, rtol=1e-4)
    approx, processed = model.prefill_ced(tokens, replay=cfg.window)
    assert processed == cfg.window
    assert not torch.allclose(approx, reference, atol=1e-5)
    # 截断只影响滑窗：近似误差远小于 logits 本身的尺度
    assert (approx - reference).abs().max() < reference.abs().max()
