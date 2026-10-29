"""DeepSeek-V3.2 的 DSA（DeepSeek-V3.2-Exp/inference/model.py 中的 ``Indexer`` 与 ``MLA``）。

官方实现的要点：
* 索引器的查询来自 MLA 的查询潜向量 ``qr``（与主注意力共享 wq_a），键来自隐藏状态，
  所有索引头共享一个 128 维键；RoPE 只作用在前 rope_dim 维（官方为非交错布局，这里沿用
  全书的交错实现，两者只差一个固定的通道置换）。
* 打分前对 q、k 做 Hadamard 旋转，再量化为 FP8：正交旋转不改变点积，却把离群通道摊平。
* 分数 I[t,s] = Σ_h w[t,h] · ReLU(q[t,h]·k[s])，w 由 ``weights_proj(x)`` 给出。
* prefill 用“MHA 模式”（还原每头 K/V，加 top-k 掩码），decode 用“MQA 模式”
  （吸收后直接读潜向量缓存），两者选中同一组 token，结果相同。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from llms_from_scratch.deepseek.v3 import MLA, V3Args
from llms_from_scratch.transformer.model import apply_rope


# region hadamard
def hadamard_matrix(n: int) -> torch.Tensor:
    """Sylvester 构造的 n×n Hadamard 矩阵（n 为 2 的幂），已除以 √n，因此是正交矩阵。"""
    assert n & (n - 1) == 0, "n 必须是 2 的幂"
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / n**0.5


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """官方 ``rotate_activation``：hadamard_transform(x, scale=d^-1/2)。"""
    return x @ hadamard_matrix(x.shape[-1]).T
# endregion hadamard


def fp8_roundtrip(x: torch.Tensor, block: int = 128) -> torch.Tensor:
    """每 block 个元素一个缩放的 FP8 E4M3 量化—反量化，模拟官方的 ``act_quant``。"""
    tiles = x.unflatten(-1, (-1, block))
    scale = tiles.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0
    return ((tiles / scale).to(torch.float8_e4m3fn).float() * scale).flatten(-2)


# region indexer
class Indexer(nn.Module):
    def __init__(self, args: V3Args, n_heads: int = 4, head_dim: int = 16, topk: int = 8,
                 quantize: bool = False) -> None:
        super().__init__()
        self.n_heads, self.head_dim, self.topk = n_heads, head_dim, topk
        self.rope_dim = args.qk_rope_head_dim
        self.quantize = quantize
        self.wq_b = nn.Linear(args.q_lora_rank, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(args.dim, head_dim, bias=False)
        self.k_norm = nn.LayerNorm(head_dim)  # 官方用 LayerNorm 而不是 RMSNorm
        self.weights_proj = nn.Linear(args.dim, n_heads, bias=False)

    def scores(self, x: torch.Tensor, qr: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """x: [B, T, d]，qr: [B, T, q_lora_rank] → 索引分数 [B, T, T]（未加因果掩码）。"""
        B, T, _ = x.shape
        rd = self.rope_dim
        q = self.wq_b(qr).view(B, T, self.n_heads, self.head_dim)
        q = torch.cat([apply_rope(q[..., :rd].transpose(1, 2), freqs).transpose(1, 2),
                       q[..., rd:]], dim=-1)
        k = self.k_norm(self.wk(x))
        k = torch.cat([apply_rope(k[..., :rd], freqs), k[..., rd:]], dim=-1)
        q, k = rotate_activation(q), rotate_activation(k)
        if self.quantize:
            q, k = fp8_roundtrip(q, self.head_dim), fp8_roundtrip(k, self.head_dim)
        w = self.weights_proj(x) * self.n_heads**-0.5 * self.head_dim**-0.5  # [B, T, H]
        logits = torch.einsum("bthd,bsd->bths", q, k)
        return (F.relu(logits) * w.unsqueeze(-1)).sum(dim=2)

    def forward(self, x: torch.Tensor, qr: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        causal = torch.ones(T, T, dtype=torch.bool).tril()
        s = self.scores(x, qr, freqs).masked_fill(~causal, float("-inf"))
        return s.topk(min(self.topk, T), dim=-1).indices  # [B, T, k]
# endregion indexer


def topk_mask(indices: torch.Tensor, T: int) -> torch.Tensor:
    """把 top-k 索引变成加性掩码：选中 0，其余 -inf；再与因果掩码叠加。"""
    m = torch.full((*indices.shape[:-1], T), float("-inf")).scatter(-1, indices, 0.0)
    return m + torch.full((T, T), float("-inf")).triu(1)


# region sparse_mla
class SparseMLA(MLA):
    """V3.2 的 MLA：同一组 top-k 索引，prefill 走 MHA 模式，decode 走 MQA（吸收）模式。"""

    def __init__(self, args: V3Args, index_topk: int = 8) -> None:
        super().__init__(args, attn_impl="absorb")
        self.indexer = Indexer(args, topk=index_topk)

    def forward(self, x: torch.Tensor, mode: str = "mha") -> torch.Tensor:  # type: ignore[override]
        a = self.args
        B, T, _ = x.shape
        freqs = self.freqs[:T]
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).view(B, T, a.n_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([a.qk_nope_head_dim, a.qk_rope_head_dim], dim=-1)
        q_pe = apply_rope(q_pe.transpose(1, 2), freqs).transpose(1, 2)
        kv, k_pe = self.wkv_a(x).split([a.kv_lora_rank, a.qk_rope_head_dim], dim=-1)
        kv, k_pe = self.kv_norm(kv), apply_rope(k_pe, freqs)
        mask = topk_mask(self.indexer(x, qr, freqs), T).unsqueeze(2)  # [B, T, 1, T]
        wkv_b = self.wkv_b.weight.view(a.n_heads, -1, a.kv_lora_rank)
        if mode == "mha":  # 每头 192 维的 q·k，V 显式还原：计算密集的 prefill 更划算
            k_nope, v = self.wkv_b(kv).view(B, T, a.n_heads, -1).split(
                [a.qk_nope_head_dim, a.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.unsqueeze(2).expand(-1, -1, a.n_heads, -1)], dim=-1)
            scores = torch.einsum("bshd,bthd->bsht", torch.cat([q_nope, q_pe], -1), k)
            probs = (scores * self.softmax_scale + mask).softmax(-1)
            out = torch.einsum("bsht,bthd->bshd", probs, v)
        else:  # 每头 576 维的 q·c，所有头共享同一份潜向量：访存受限的 decode 更划算
            q_lat = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :a.qk_nope_head_dim])
            scores = (torch.einsum("bshc,btc->bsht", q_lat, kv)
                      + torch.einsum("bshr,btr->bsht", q_pe, k_pe))
            probs = (scores * self.softmax_scale + mask).softmax(-1)
            out = torch.einsum("bsht,btc->bshc", probs, kv)
            out = torch.einsum("bshc,hdc->bshd", out, wkv_b[:, -a.v_head_dim:])
        return self.wo(out.flatten(2))
# endregion sparse_mla


def indexer_distill_target(attn_probs: torch.Tensor) -> torch.Tensor:
    """稠密预热阶段的目标分布：主注意力概率 [B, H, T, S] 对头求和后按行 L1 归一化。"""
    p = attn_probs.sum(dim=1)
    return p / p.sum(-1, keepdim=True)
