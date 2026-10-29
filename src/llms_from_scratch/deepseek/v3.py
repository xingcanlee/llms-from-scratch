"""DeepSeek-V3 官方推理代码（deepseek-ai/DeepSeek-V3/inference/model.py）的教学移植。

保留官方的权重布局与两条注意力路径，去掉张量并行：
    wkv_a: d → kv_lora_rank + rope_dim     一次投影同时得到潜向量 c_kv 与位置键 k_R
    wkv_b: kv_lora_rank → H·(nope + v)    每个头的 W_UK 与 W_UV 拼在同一个矩阵里
``attn_impl="naive"`` 缓存还原后的 K/V，``"absorb"`` 只缓存 c_kv 与 k_R。
路由器 ``Gate`` 实现分组受限路由（node-limited routing）与无辅助损失偏置；
``block_quant``/``block_dequant`` 模拟 V3 的 128×128 块级 FP8 权重量化。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from llms_from_scratch.transformer.model import RMSNorm, apply_rope, rope_frequencies


@dataclass
class V3Args:
    dim: int = 64
    n_heads: int = 4
    q_lora_rank: int = 32
    kv_lora_rank: int = 16
    qk_nope_head_dim: int = 16
    qk_rope_head_dim: int = 8
    v_head_dim: int = 16
    max_seq_len: int = 64
    max_batch_size: int = 2
    # MoE：n_routed_experts 个专家分成 n_expert_groups 组，每个 token 只能从 n_limited_groups 组中选
    moe_inter_dim: int = 32
    n_routed_experts: int = 16
    n_shared_experts: int = 1
    n_activated_experts: int = 4
    n_expert_groups: int = 4
    n_limited_groups: int = 2
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_scale: float = 2.5


# region mla
class MLA(nn.Module):
    def __init__(self, args: V3Args, attn_impl: Literal["naive", "absorb"] = "absorb") -> None:
        super().__init__()
        a = args
        self.args, self.attn_impl = a, attn_impl
        self.qk_head_dim = a.qk_nope_head_dim + a.qk_rope_head_dim
        self.wq_a = nn.Linear(a.dim, a.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(a.q_lora_rank)
        self.wq_b = nn.Linear(a.q_lora_rank, a.n_heads * self.qk_head_dim, bias=False)
        self.wkv_a = nn.Linear(a.dim, a.kv_lora_rank + a.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(a.kv_lora_rank)
        self.wkv_b = nn.Linear(a.kv_lora_rank, a.n_heads * (a.qk_nope_head_dim + a.v_head_dim),
                               bias=False)
        self.wo = nn.Linear(a.n_heads * a.v_head_dim, a.dim, bias=False)
        self.softmax_scale = self.qk_head_dim ** -0.5
        B, S, H = a.max_batch_size, a.max_seq_len, a.n_heads
        if attn_impl == "naive":  # 每头完整的 K/V：H·(192+128) 个数/token（V3 尺寸）
            self.register_buffer("k_cache", torch.zeros(B, S, H, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(B, S, H, a.v_head_dim), persistent=False)
        else:  # 只有潜向量与位置键：512+64 个数/token
            self.register_buffer("kv_cache", torch.zeros(B, S, a.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(B, S, a.qk_rope_head_dim), persistent=False)
        self.register_buffer("freqs", rope_frequencies(a.qk_rope_head_dim, S), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        a = self.args
        bsz, seqlen, _ = x.shape
        end_pos = start_pos + seqlen
        freqs = self.freqs[start_pos:end_pos]
        q = self.wq_b(self.q_norm(self.wq_a(x))).view(bsz, seqlen, a.n_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([a.qk_nope_head_dim, a.qk_rope_head_dim], dim=-1)
        q_pe = apply_rope(q_pe.transpose(1, 2), freqs).transpose(1, 2)
        kv, k_pe = self.wkv_a(x).split([a.kv_lora_rank, a.qk_rope_head_dim], dim=-1)
        k_pe = apply_rope(k_pe, freqs)  # [B, T, rope]，所有头共用
        mask = torch.full((seqlen, end_pos), float("-inf")).triu(start_pos + 1)

        if self.attn_impl == "naive":
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv)).view(bsz, seqlen, a.n_heads, -1)
            k_nope, v = kv.split([a.qk_nope_head_dim, a.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.unsqueeze(2).expand(-1, -1, a.n_heads, -1)], dim=-1)
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos])
        else:
            wkv_b = self.wkv_b.weight.view(a.n_heads, -1, a.kv_lora_rank)  # [H, nope+v, rank]
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :a.qk_nope_head_dim])
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos])
                      + torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos]))
        scores = scores * self.softmax_scale + mask.unsqueeze(1)  # [B, T, H, S]
        probs = scores.softmax(dim=-1)

        if self.attn_impl == "naive":
            out = torch.einsum("bsht,bthd->bshd", probs, self.v_cache[:bsz, :end_pos])
        else:
            out = torch.einsum("bsht,btc->bshc", probs, self.kv_cache[:bsz, :end_pos])
            out = torch.einsum("bshc,hdc->bshd", out, wkv_b[:, -a.v_head_dim:])
        return self.wo(out.flatten(2))
# endregion mla


# region gate
class Gate(nn.Module):
    """V3 路由：sigmoid 打分 → 加偏置选组 → 组内选专家 → 用不带偏置的分数加权。"""

    def __init__(self, args: V3Args) -> None:
        super().__init__()
        self.args = args
        self.weight = nn.Parameter(torch.randn(args.n_routed_experts, args.dim) * args.dim**-0.5)
        # 无辅助损失均衡的偏置：只影响“选谁”，训练中按负载在线调整，推理时是常量
        self.register_buffer("bias", torch.zeros(args.n_routed_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.args
        scores = F.linear(x.float(), self.weight.float())
        scores = scores.softmax(-1) if a.score_func == "softmax" else scores.sigmoid()
        original_scores = scores
        scores = scores + self.bias
        if a.n_expert_groups > 1:
            scores = scores.view(x.size(0), a.n_expert_groups, -1)
            group_scores = scores.topk(2, dim=-1).values.sum(dim=-1)  # 组分 = 组内最高两个之和
            keep = group_scores.topk(a.n_limited_groups, dim=-1).indices
            drop = torch.ones_like(group_scores, dtype=torch.bool).scatter_(1, keep, False)
            scores = scores.masked_fill(drop.unsqueeze(-1), float("-inf")).flatten(1)
        indices = scores.topk(a.n_activated_experts, dim=-1).indices
        weights = original_scores.gather(1, indices)
        if a.score_func == "sigmoid":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights * a.route_scale, indices
# endregion gate


class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# region moe
class MoE(nn.Module):
    def __init__(self, args: V3Args) -> None:
        super().__init__()
        self.dim = args.dim
        self.gate = Gate(args)
        self.experts = nn.ModuleList(Expert(args.dim, args.moe_inter_dim)
                                     for _ in range(args.n_routed_experts))
        self.shared_experts = Expert(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=len(self.experts)).tolist()
        for i, expert in enumerate(self.experts):
            if counts[i] == 0:  # 官方代码同样跳过本批没有 token 的专家
                continue
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        return (y + self.shared_experts(x)).view(shape)
# endregion moe


# region fp8_block
FP8_MAX = 448.0  # E4M3 的最大有限值


def block_quant(w: torch.Tensor, block: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """每个 block×block 子块一个 FP32 缩放，使子块的最大绝对值映射到 448。"""
    m, n = w.shape
    tiles = w.view(m // block, block, n // block, block)
    amax = tiles.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12)
    scale = amax / FP8_MAX
    q = (tiles / scale).to(torch.float8_e4m3fn)
    return q.view(m, n), scale.view(m // block, n // block)


def block_dequant(q: torch.Tensor, scale: torch.Tensor, block: int = 128) -> torch.Tensor:
    m, n = q.shape
    tiles = q.float().view(m // block, block, n // block, block)
    return (tiles * scale.view(m // block, 1, n // block, 1)).view(m, n)
# endregion fp8_block


def absorbed_flops_ratio(args: V3Args, context: int) -> float:
    """单个 decode 查询：吸收路径与“每步临时还原 K/V”路径的注意力乘加数之比（每头）。"""
    a = args
    naive = context * (a.kv_lora_rank * (a.qk_nope_head_dim + a.v_head_dim)  # 还原 K/V
                       + (a.qk_nope_head_dim + a.qk_rope_head_dim) + a.v_head_dim)
    absorb = (a.qk_nope_head_dim * a.kv_lora_rank + a.kv_lora_rank * a.v_head_dim  # 吸收的两次
              + context * (a.kv_lora_rank + a.qk_rope_head_dim + a.kv_lora_rank))
    return absorb / naive


def softmax_scale_with_yarn(qk_head_dim: int, rope_factor: float, mscale: float = 1.0) -> float:
    """V3 在 YaRN 扩展后把 softmax 缩放乘以 mscale²，mscale = 0.1·m·ln(factor) + 1。"""
    m = 0.1 * mscale * math.log(rope_factor) + 1.0
    return qk_head_dim ** -0.5 * m * m
