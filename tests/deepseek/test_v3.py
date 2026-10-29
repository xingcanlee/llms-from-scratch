import torch

from llms_from_scratch.deepseek.v3 import (
    MLA,
    MoE,
    V3Args,
    block_dequant,
    block_quant,
    softmax_scale_with_yarn,
)


def test_naive_and_absorb_agree_for_prefill_and_decode():
    torch.manual_seed(0)
    args = V3Args()
    naive, absorb = MLA(args, "naive"), MLA(args, "absorb")
    absorb.load_state_dict(naive.state_dict())
    x = torch.randn(2, 10, args.dim)
    full = naive(x)
    torch.testing.assert_close(absorb(x), full, atol=1e-5, rtol=1e-4)
    # absorb 路径：prefill 6 个 token，再逐个 decode，只读潜向量缓存
    outs = [absorb(x[:, :6], 0)] + [absorb(x[:, t:t + 1], t) for t in range(6, 10)]
    torch.testing.assert_close(torch.cat(outs, 1), full, atol=1e-5, rtol=1e-4)
    assert absorb.kv_cache.shape[-1] + absorb.pe_cache.shape[-1] == 24


def test_gate_respects_group_limit_and_bias_only_selects():
    torch.manual_seed(0)
    args = V3Args()
    moe = MoE(args)
    x = torch.randn(32, args.dim)
    weights, idx = moe.gate(x)
    group_size = args.n_routed_experts // args.n_expert_groups
    groups = idx // group_size
    assert all(len(set(row.tolist())) <= args.n_limited_groups for row in groups)
    torch.testing.assert_close(weights.sum(-1), torch.full((32,), args.route_scale))
    # 给专家 0 一个很大的偏置：它总被选中，但权重仍来自原始 sigmoid 分数
    moe.gate.bias[0] = 10.0
    w2, idx2 = moe.gate(x)
    assert (idx2 == 0).any(-1).all()
    raw = torch.sigmoid(x @ moe.gate.weight.T).gather(1, idx2)
    torch.testing.assert_close(w2, raw / raw.sum(-1, keepdim=True) * args.route_scale)


def test_moe_forward_and_backward():
    torch.manual_seed(0)
    moe = MoE(V3Args())
    x = torch.randn(2, 5, 64, requires_grad=True)
    moe(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_block_fp8_quant_error_is_relative_to_block_max():
    torch.manual_seed(0)
    w = torch.randn(256, 256)
    w[:128, :128] *= 100  # 一个子块的量级大得多，只影响它自己的缩放
    q, s = block_quant(w)
    assert q.dtype == torch.float8_e4m3fn and s.shape == (2, 2)
    err = (block_dequant(q, s) - w).abs()
    # E4M3 有 3 位尾数：相对误差不超过 2^-4
    assert (err[128:, 128:] <= w[128:, 128:].abs() * 2**-4 + 1e-6).all()
    assert err[128:, 128:].max() < 0.2


def test_yarn_mscale_increases_softmax_scale():
    assert softmax_scale_with_yarn(192, 40.0) > 192**-0.5
