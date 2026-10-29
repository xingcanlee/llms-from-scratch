"""DeepSeek-V4 注意力（DeepSeek-V4-Flash/inference/model.py）的教学移植。

一层 V4 注意力读两类 KV，并在同一个 softmax 中处理：
* 滑动窗口：最近 window 个 token 的原始 KV（环形缓冲区）；
* 压缩 KV：``Compressor`` 把每 m 个 token 用带位置偏置（APE）的 softmax 门控池化成一个条目。
  m = 4（CSA）时窗口重叠：条目 j 汇聚 token [4j-4, 4j+4)，前一半来自另一组投影通道；
  m = 128（HCA）时不重叠。CSA 层再由索引器选 top-k 个压缩条目，HCA 层读全部条目。
每个条目同时充当所有查询头的键和值（共享 KV 的 MQA，头维 512），softmax 分母里额外
加一个每头的可学习“汇聚”logit；输出去掉查询的旋转（逆 RoPE），再经分组低秩投影。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from llms_from_scratch.transformer.model import RMSNorm, apply_rope, rope_frequencies


# region compressor
class Compressor(nn.Module):
    """``forward(x, start_pos)``：prefill（start_pos == 0）一次压缩所有完整的组；
    decode 每步进一个 token，凑满一组才输出一个条目，否则返回 None。"""

    def __init__(self, dim: int, head_dim: int, ratio: int, max_batch_size: int = 2) -> None:
        super().__init__()
        self.ratio, self.head_dim = ratio, head_dim
        self.overlap = ratio == 4
        coff = 1 + self.overlap  # 重叠时两组投影：前 head_dim 通道给“下一组”，后一半给本组
        self.wkv = nn.Linear(dim, coff * head_dim, bias=False)
        self.wgate = nn.Linear(dim, coff * head_dim, bias=False)
        self.ape = nn.Parameter(torch.randn(ratio, coff * head_dim) * 0.1)
        self.norm = RMSNorm(head_dim)
        state = (max_batch_size, coff * ratio, coff * head_dim)
        self.register_buffer("kv_state", torch.zeros(state), persistent=False)
        self.register_buffer("score_state", torch.full(state, float("-inf")), persistent=False)

    def overlap_transform(self, t: torch.Tensor, fill: float) -> torch.Tensor:
        """[B, G, r, 2d] → [B, G, 2r, d]：组 g 的前 r 行取组 g-1 的“前半通道”，后 r 行取自身。"""
        B, G, r, _ = t.shape
        d = self.head_dim
        out = t.new_full((B, G, 2 * r, d), fill)
        out[:, :, r:] = t[:, :, :, d:]
        out[:, 1:, :r] = t[:, :-1, :, :d]
        return out

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor | None:
        bsz, seqlen, _ = x.shape
        r, d = self.ratio, self.head_dim
        kv, score = self.wkv(x), self.wgate(x)
        if start_pos == 0:  # 新序列：清空上一条序列留下的状态（尤其是重叠窗口的前半部分）
            self.kv_state[:bsz] = 0.0
            self.score_state[:bsz] = float("-inf")
            cutoff = seqlen - seqlen % r
            offset = r if self.overlap else 0
            if self.overlap and cutoff >= r:  # 最后一个完整组留作下一组的“前半窗口”
                self.kv_state[:bsz, :r] = kv[:, cutoff - r:cutoff]
                self.score_state[:bsz, :r] = score[:, cutoff - r:cutoff] + self.ape
            rem = seqlen - cutoff
            if rem:  # 不满一组的尾巴存进状态，等 decode 补齐
                self.kv_state[:bsz, offset:offset + rem] = kv[:, cutoff:]
                self.score_state[:bsz, offset:offset + rem] = score[:, cutoff:] + self.ape[:rem]
            if cutoff == 0:
                return None
            kv = kv[:, :cutoff].unflatten(1, (-1, r))
            score = score[:, :cutoff].unflatten(1, (-1, r)) + self.ape
            if self.overlap:
                kv, score = self.overlap_transform(kv, 0.0), self.overlap_transform(score, float("-inf"))
            out = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            slot = start_pos % r
            score = score + self.ape[slot]
            row = r + slot if self.overlap else slot
            self.kv_state[:bsz, row] = kv.squeeze(1)
            self.score_state[:bsz, row] = score.squeeze(1)
            if (start_pos + 1) % r:
                return None
            if self.overlap:
                kv_s = torch.cat([self.kv_state[:bsz, :r, :d], self.kv_state[:bsz, r:, d:]], dim=1)
                sc_s = torch.cat([self.score_state[:bsz, :r, :d], self.score_state[:bsz, r:, d:]], 1)
                out = (kv_s * sc_s.softmax(dim=1)).sum(dim=1, keepdim=True)
                self.kv_state[:bsz, :r] = self.kv_state[:bsz, r:]  # 本组成为下一组的前半窗口
                self.score_state[:bsz, :r] = self.score_state[:bsz, r:]
            else:
                out = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(1, True)
        return self.norm(out)
# endregion compressor


# region sparse_attn
def sparse_attn(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor,
                topk_idxs: torch.Tensor, scale: float) -> torch.Tensor:
    """官方 TileLang ``sparse_attn`` kernel 的参考语义。

    q: [B, M, H, D]，kv: [B, N, D]（键与值是同一个张量），topk_idxs: [B, M, K]，-1 表示空位。
    分母多一项 exp(attn_sink[h])：它像一个值为 0 的虚拟键，让一行权重之和可以小于 1。
    """
    B, M, H, D = q.shape
    valid = topk_idxs >= 0
    gathered = torch.gather(kv.unsqueeze(1).expand(B, M, -1, D), 2,
                            topk_idxs.clamp_min(0).long().unsqueeze(-1).expand(-1, -1, -1, D))
    logits = torch.einsum("bmhd,bmkd->bmhk", q, gathered) * scale
    logits = logits.masked_fill(~valid.unsqueeze(2), float("-inf"))
    sink = attn_sink.view(1, 1, H, 1).expand(B, M, H, 1)
    probs = torch.cat([logits, sink], dim=-1).softmax(-1)[..., :-1]
    return torch.einsum("bmhk,bmkd->bmhd", probs, gathered)
# endregion sparse_attn


def window_topk_idxs(window: int, seqlen: int) -> torch.Tensor:
    """prefill 时每个查询的滑窗位置 [seqlen, min(seqlen, window)]，序列开始之前记 -1。"""
    end = torch.arange(seqlen).unsqueeze(1)
    idxs = (end - window + 1).clamp(min=0) + torch.arange(min(seqlen, window))
    return torch.where(idxs > end, -1, idxs)


def compress_topk_idxs(ratio: int, seqlen: int, offset: int) -> torch.Tensor:
    """HCA：查询 t 可以读所有已完整结束的组 j（(j+1)·m ≤ t+1），索引整体加上 offset。"""
    n = seqlen // ratio
    j = torch.arange(n)
    visible = (j + 1) * ratio <= torch.arange(seqlen).unsqueeze(1) + 1
    return torch.where(visible, j + offset, -1)


# region attention
class V4Attention(nn.Module):
    """不含索引器的 V4 注意力（相当于 HCA 层或 compress_ratio = 0 的纯滑窗层），只做 prefill。"""

    def __init__(self, dim: int = 64, n_heads: int = 4, head_dim: int = 32, rope_dim: int = 8,
                 q_lora_rank: int = 32, o_groups: int = 2, o_lora_rank: int = 16,
                 window: int = 4, ratio: int = 0, max_len: int = 128) -> None:
        super().__init__()
        self.n_heads, self.head_dim, self.rope_dim = n_heads, head_dim, rope_dim
        self.window, self.ratio, self.o_groups, self.o_lora_rank = window, ratio, o_groups, o_lora_rank
        self.wq_a = nn.Linear(dim, q_lora_rank, bias=False)
        self.q_norm = RMSNorm(q_lora_rank)
        self.wq_b = nn.Linear(q_lora_rank, n_heads * head_dim, bias=False)
        self.wkv = nn.Linear(dim, head_dim, bias=False)  # 只有一个 KV 头：键 = 值
        self.kv_norm = RMSNorm(head_dim)
        self.attn_sink = nn.Parameter(torch.zeros(n_heads))
        # 分组低秩输出：每组 n_heads/o_groups 个头 → o_lora_rank，再统一投影回 dim
        self.wo_a = nn.Parameter(torch.randn(o_groups, o_lora_rank, n_heads * head_dim // o_groups)
                                 * (n_heads * head_dim // o_groups) ** -0.5)
        self.wo_b = nn.Linear(o_groups * o_lora_rank, dim, bias=False)
        self.compressor = Compressor(dim, head_dim, ratio) if ratio else None
        self.register_buffer("freqs", rope_frequencies(rope_dim, max_len), persistent=False)

    def forward(self, x: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        """``pos_offset`` 只平移 RoPE 的绝对位置，用来检验输出只依赖相对位置。"""
        B, T, _ = x.shape
        rd = self.rope_dim
        freqs = self.freqs[pos_offset:pos_offset + T]
        q = self.wq_b(self.q_norm(self.wq_a(x))).unflatten(-1, (self.n_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6)  # 每头再做一次无参 RMS 归一
        q = torch.cat([q[..., :-rd], apply_rope(q[..., -rd:].transpose(1, 2), freqs).transpose(1, 2)],
                      dim=-1)
        kv = self.kv_norm(self.wkv(x))
        kv = torch.cat([kv[..., :-rd], apply_rope(kv[..., -rd:], freqs)], dim=-1)
        idxs = window_topk_idxs(self.window, T).expand(B, -1, -1)
        if self.compressor is not None:
            comp = self.compressor(x, 0)
            if comp is not None:
                # 条目 j 代表组的第一个 token，按位置 j·m 旋转（官方用单独的 compress_rope_theta）
                cf = self.freqs[pos_offset:pos_offset + comp.shape[1] * self.ratio:self.ratio]
                comp = torch.cat([comp[..., :-rd], apply_rope(comp[..., -rd:], cf)], dim=-1)
                idxs = torch.cat([idxs, compress_topk_idxs(self.ratio, T, T).expand(B, -1, -1)], -1)
                kv = torch.cat([kv, comp], dim=1)
        o = sparse_attn(q, kv, self.attn_sink, idxs, self.head_dim**-0.5)
        # 值也带着键的旋转 R_s；输出乘 R_t^{-1} 后只剩 R_{s-t}：值的位置信息变成相对的
        o_rope = apply_rope(o[..., -rd:].transpose(1, 2), freqs.conj()).transpose(1, 2)
        o = torch.cat([o[..., :-rd], o_rope], dim=-1)
        o = o.reshape(B, T, self.o_groups, -1)
        o = torch.einsum("btgd,grd->btgr", o, self.wo_a)  # 块对角：每组只投影自己的头
        return self.wo_b(o.flatten(2))
# endregion attention


def sqrtsoftplus_scores(logits: torch.Tensor) -> torch.Tensor:
    """V4 路由的打分函数 √softplus(z)：非负、无上界，增长比 sigmoid 平缓。"""
    return F.softplus(logits).sqrt()


def hash_route(input_ids: torch.Tensor, tid2eid: torch.Tensor) -> torch.Tensor:
    """V4 前 3 层的哈希路由：专家由 token id 查表决定（训练前固定），与隐藏状态无关。"""
    return tid2eid[input_ids]
