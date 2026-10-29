"""V4/V4.1 的 mHC 实现细节与 Single-Pass mHC（对照 V4.1 model.py 的 ``Block`` 与 kernel.py 的
``hc_split_sinkhorn``）。

一个子层（注意力或 FFN）由三组逐 token 的系数包住：A（pre，1×n，读）、C（post，n×1，写）、
B（comb，n×n，混合）。原始 mHC 中子层 l 用自己由 X_l 算出的 A_l；Single-Pass mHC 把 A
推迟一个子层，用上一个子层算出的 A_{l-1}，于是“残差更新 + 读入混合 + 系数预测”可以在
一次遍历 X_l 中完成。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# region split_sinkhorn
def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
                      hc: int, iters: int = 20, eps: float = 1e-6):
    """mixes: [..., (2+hc)·hc] → pre [..., hc], post [..., hc], comb [..., hc, hc]。

    与官方 kernel 逐步一致：pre = σ(·)+eps，post = 2σ(·)；comb 先按行 softmax（+eps），
    再列归一化，之后交替行、列归一化共 iters 轮，最后一步是列归一化。
    """
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]
    comb = comb.unflatten(-1, (hc, hc)).softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
# endregion split_sinkhorn


def hc_mixes(x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
             hc: int, eps: float = 1e-20):
    """系数预测 H(X)：把 [.., hc, d] 展平成 hc·d，先投影、后除以 RMS（归一化权重已折进 hc_fn）。"""
    flat = x.flatten(-2)
    mixes = F.linear(flat, hc_fn) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
    return hc_split_sinkhorn(mixes, hc_scale, hc_base, hc)


def hc_pre(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """A·X：n 条流加权求和成子层输入 [.., d]。"""
    return (pre.unsqueeze(-1) * x).sum(-2)


def hc_post(y: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
    """B^T X + C y：out[k] = Σ_j comb[j,k]·X[j] + post[k]·y（与官方 ``hc_post`` 的求和维一致）。"""
    return post.unsqueeze(-1) * y.unsqueeze(-2) + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(-3)


# region stack
def run_sublayers(x: torch.Tensor, sublayers, params, hc: int, single_pass: bool) -> torch.Tensor:
    """依次执行若干子层。params[i] = (hc_fn, hc_scale, hc_base)。

    single_pass=False：子层 i 用 H(X_i) 给出的 A_i（原始 mHC，需要先完整归约 X_i 才能读入）；
    single_pass=True：子层 i 用前一个子层给出的 A_{i-1}，第一个子层用“只读第 0 条流”。
    """
    pre_prev = torch.zeros(*x.shape[:-2], hc)
    pre_prev[..., 0] = 1.0
    for f, (fn, scale, base) in zip(sublayers, params, strict=True):
        pre, post, comb = hc_mixes(x, fn, scale, base, hc)
        use = pre_prev if single_pass else pre
        x = hc_post(f(hc_pre(x, use)), x, post, comb)
        pre_prev = pre
    return x
# endregion stack


def activation_traffic(n: int, d: int, variant: str) -> int:
    """每个子层边界的激活读写量（元素数），见 V4.1 报告 §2.4.1。"""
    return {
        "lower_bound": (2 * n + 2) * d,  # 读 X_{l-1}、Y_{l-1}，写 X_l 与子层输入
        "three_kernels": (4 * n + 4) * d,  # V4 的多遍实现（含 pre-norm）
        "two_pass": (3 * n + 2) * d,  # 融合残差更新与系数预测，读入混合仍需再读一遍 X_l
        "single_pass": (2 * n + 2) * d,  # Mega-mHC：读一次、写一次
    }[variant]
