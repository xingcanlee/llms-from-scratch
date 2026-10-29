import torch

from llms_from_scratch.deepseek.v4_attention import (
    Compressor,
    V4Attention,
    compress_topk_idxs,
    sparse_attn,
    sqrtsoftplus_scores,
)


def _stream(comp, x, prefill):
    outs = [comp(x[:, :prefill], 0)]
    outs += [comp(x[:, t:t + 1], t) for t in range(prefill, x.size(1))]
    return torch.cat([o for o in outs if o is not None], dim=1)


def test_compressor_streaming_equals_prefill():
    torch.manual_seed(0)
    for ratio in (4, 8):  # 4 = CSA（重叠），8 代表 HCA 式的不重叠压缩
        comp = Compressor(16, 8, ratio)
        x = torch.randn(2, 6 * ratio + 1, 16)
        full = comp(x, 0)
        for prefill in (1, ratio, 2 * ratio + 3):
            torch.testing.assert_close(_stream(comp, x, prefill), full, atol=1e-5, rtol=1e-4)


def test_overlap_entry_sees_previous_group():
    torch.manual_seed(0)
    comp = Compressor(16, 8, 4)
    x = torch.randn(1, 12, 16)
    base = comp(x, 0)
    x2 = x.clone()
    x2[:, 3] += 1.0  # 改动组 0 的最后一个 token
    out = comp(x2, 0)
    assert not torch.allclose(out[:, 0], base[:, 0])
    assert not torch.allclose(out[:, 1], base[:, 1])  # 组 1 的条目也汇聚了组 0
    torch.testing.assert_close(out[:, 2], base[:, 2])


def test_sparse_attn_sink_and_padding():
    torch.manual_seed(0)
    q, kv = torch.randn(1, 3, 2, 4), torch.randn(1, 5, 4)
    idxs = torch.tensor([[[0, -1], [0, 1], [-1, -1]]])
    no_sink = sparse_attn(q, kv, torch.full((2,), -1e9), idxs, 0.5)
    torch.testing.assert_close(no_sink[0, 0], kv[0, 0].expand(2, 4))  # 只有一个有效键
    assert no_sink[0, 2].abs().max() == 0  # 没有有效键：输出全零
    big_sink = sparse_attn(q, kv, torch.full((2,), 30.0), idxs, 0.5)
    assert big_sink.abs().max() < 1e-6  # 汇聚吸走了全部权重


def test_output_depends_only_on_relative_positions():
    torch.manual_seed(0)
    for ratio in (0, 4):
        attn = V4Attention(ratio=ratio)
        x = torch.randn(2, 13, 64)
        torch.testing.assert_close(attn(x, 0), attn(x, 37), atol=1e-4, rtol=1e-4)


def test_hca_visibility_and_router_scores():
    idx = compress_topk_idxs(4, 9, offset=100)
    assert idx[2].tolist() == [-1, -1] and idx[3].tolist() == [100, -1]
    assert idx[8].tolist() == [100, 101]
    s = sqrtsoftplus_scores(torch.tensor([-20.0, 0.0, 20.0]))
    assert s[0] > 0 and abs(s[2] - 20**0.5) < 1e-3
