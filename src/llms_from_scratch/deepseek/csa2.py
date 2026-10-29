"""DeepSeek-V4.1 的 CSA2 与因果编码器—解码器（CED），对照 DeepSeek-V4.1-Flash/inference/model.py。

每层注意力 = 本层滑窗 KV + 跨层共享的压缩“全局 KV”中被选中的 top-k 条目。层的模式由 config 静态决定：
    Full    层号 ∈ kv_source_layers：本层压缩出主 KV、投影出索引器键，并运行索引器
    Reindex 层号 ∈ index_source_layers 但不是 kv 源：复用主 KV 与索引器键，用自己的查询重选 top-k
    Reuse   其余压缩层：主 KV 与 top-k 索引都复用最近的源层
CED 不需要新的模块：解码器第一层是 Full 层，它的输入就是编码器最后一层的输出，于是整个
解码器的全局 KV 都由编码器输出投影而来。层间共享的状态放在 ``SharedState`` 中，按层号顺序
“先写后读”，对应官方的 ``SharedAttentionRuntime``。

为了突出数据流，本实现省略 RoPE、MoE、mHC、量化与张量并行；每层只有一个共享的 KV 头。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from llms_from_scratch.deepseek.v4_attention import sparse_attn
from llms_from_scratch.transformer.model import RMSNorm


@dataclass
class CSA2Config:
    vocab_size: int = 64
    dim: int = 32
    n_heads: int = 2
    head_dim: int = 16
    window: int = 4
    # 每层一个压缩率：0 = 只有滑窗，m = 每 m 个 token 压成一个条目
    compress_ratios: tuple[int, ...] = (0, 2, 2, 2, 1, 1, 1, 1)
    kv_source_layers: tuple[int, ...] = (1, 4)
    index_source_layers: tuple[int, ...] = (1, 4, 6)
    n_encoder_layers: int = 4
    index_n_heads: int = 2
    index_head_dim: int = 8
    index_topk: int = 3
    candidate_source_layer: int = -1  # ≥ 0 时启用分层稀疏索引器
    candidate_topk_blocks: int = 2
    candidate_block_size: int = 2
    max_batch_size: int = 2
    max_seq_len: int = 64

    @property
    def n_layers(self) -> int:
        return len(self.compress_ratios)


# region modes
def layer_modes(compress_ratios, kv_source_layers, index_source_layers) -> list[str]:
    """按官方规则推出每层的 CSA2 模式。"""
    modes = []
    for layer, ratio in enumerate(compress_ratios):
        if ratio == 0:
            modes.append("swa")
        elif layer in kv_source_layers:
            modes.append("full")
        elif layer in index_source_layers:
            modes.append("reindex")
        else:
            modes.append("reuse")
    return modes
# endregion modes


@dataclass
class SharedState:
    """层与层之间传递的状态：每种只有一个槽，源层写、后续层读，无需在两次前向之间清空。"""

    compress_kv: torch.Tensor | None = None  # 主 KV 缓存 [B, max_len / m, D]
    index_k: torch.Tensor | None = None  # 索引器键缓存 [B, max_len / m, d_I]
    topk_idxs: torch.Tensor | None = None  # 最近一次索引的结果（未加偏移）
    candidates: torch.Tensor | None = None  # 分层索引器的候选池掩码
    extra: dict = field(default_factory=dict)


# region compressor
class Compressor2(nn.Module):
    """CSA2 的压缩器：不重叠、无位置偏置；m = 1 时只是一次投影。"""

    def __init__(self, dim: int, head_dim: int, ratio: int, max_batch_size: int) -> None:
        super().__init__()
        self.ratio = ratio
        self.wkv = nn.Linear(dim, head_dim, bias=False)
        self.wgate = nn.Linear(dim, head_dim, bias=False) if ratio > 1 else None
        self.norm = RMSNorm(head_dim)
        shape = (max_batch_size, ratio, head_dim)
        self.register_buffer("kv_state", torch.zeros(shape), persistent=False)
        self.register_buffer("score_state", torch.full(shape, float("-inf")), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor | None:
        bsz, seqlen, _ = x.shape
        r = self.ratio
        if r == 1:
            return self.norm(self.wkv(x))
        kv, score = self.wkv(x), self.wgate(x)
        if start_pos == 0:
            cutoff = seqlen - seqlen % r
            self.kv_state[:bsz, :seqlen - cutoff] = kv[:, cutoff:]
            self.score_state[:bsz, :seqlen - cutoff] = score[:, cutoff:]
            if cutoff == 0:
                return None
            kv = kv[:, :cutoff].unflatten(1, (-1, r))
            score = score[:, :cutoff].unflatten(1, (-1, r))
            return self.norm((kv * score.softmax(dim=2)).sum(dim=2))
        out = None
        for i in range(seqlen):  # decode：逐 token 填槽，凑满一组才池化
            pos = start_pos + i
            self.kv_state[:bsz, pos % r] = kv[:, i]
            self.score_state[:bsz, pos % r] = score[:, i]
            if (pos + 1) % r == 0:
                g = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(1, True)
                out = g if out is None else torch.cat([out, g], dim=1)
        return None if out is None else self.norm(out)
# endregion compressor


# region candidates
def select_candidate_blocks(logits: torch.Tensor, compress_lens: torch.Tensor | int,
                            topk_blocks: int, block_size: int) -> torch.Tensor:
    """分层索引器第一级：块分 = 块内最高的位置分，保留 topk_blocks 个块，返回位置级布尔掩码。

    含有查询最新位置的那个块只填了一部分，强制入选（分数置 +inf），否则它可能输给更旧的满块。
    """
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=float("-inf"))
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    n_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(n_blocks) == last, float("inf"))
    values, indices = scores.sort(dim=-1, descending=True, stable=True)  # 平分取较早的块
    k = min(topk_blocks, n_blocks)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, indices[..., :k],
                                                               values[..., :k] > float("-inf"))
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]
# endregion candidates


# region indexer
class Indexer2(nn.Module):
    """CSA2 索引器：只有 Full 层拥有索引器键（由主 KV 潜向量投影得到），其余层读共享缓存。"""

    def __init__(self, cfg: CSA2Config, layer: int) -> None:
        super().__init__()
        self.cfg, self.ratio = cfg, cfg.compress_ratios[layer]
        self.owns_k = layer in cfg.kv_source_layers
        self.is_candidate_source = layer == cfg.candidate_source_layer
        self.uses_candidates = 0 <= cfg.candidate_source_layer < layer
        self.wq = nn.Linear(cfg.dim, cfg.index_n_heads * cfg.index_head_dim, bias=False)
        self.weights_proj = nn.Linear(cfg.dim, cfg.index_n_heads, bias=False)
        if self.owns_k:
            self.wk = nn.Linear(cfg.head_dim, cfg.index_head_dim, bias=False)
            self.k_norm = RMSNorm(cfg.index_head_dim)
            self.register_buffer("k_cache", torch.zeros(
                cfg.max_batch_size, cfg.max_seq_len // self.ratio, cfg.index_head_dim),
                persistent=False)

    def append_keys(self, latent: torch.Tensor, start_pos: int, state: SharedState) -> None:
        k = self.k_norm(self.wk(latent))  # CSA2：索引器键 = 主 KV 条目的投影
        first = start_pos // self.ratio
        self.k_cache[:latent.size(0), first:first + k.size(1)] = k
        state.index_k = self.k_cache

    def forward(self, x: torch.Tensor, start_pos: int, state: SharedState) -> torch.Tensor:
        cfg, r = self.cfg, self.ratio
        bsz, seqlen, _ = x.shape
        end = start_pos + seqlen
        n_comp = end // r
        q = self.wq(x).unflatten(-1, (cfg.index_n_heads, cfg.index_head_dim))
        w = self.weights_proj(x) * (cfg.index_head_dim * cfg.index_n_heads) ** -0.5
        k = state.index_k[:bsz, :n_comp]
        score = (torch.einsum("bshd,btd->bsht", q, k).relu() * w.unsqueeze(-1)).sum(dim=2)
        # 查询 p 只能看见已经结束的组：组 j 可见 ⇔ (j+1)·m ≤ p+1
        compress_lens = (torch.arange(start_pos + 1, end + 1) // r).unsqueeze(-1)
        score = score.masked_fill(torch.arange(n_comp) >= compress_lens, float("-inf"))
        if self.is_candidate_source:
            state.candidates = select_candidate_blocks(
                score, compress_lens, cfg.candidate_topk_blocks, cfg.candidate_block_size)
        elif self.uses_candidates:
            score = score.masked_fill(~state.candidates, float("-inf"))
        # ReLU 让很多分数恰好为 0；稳定排序让平分时取较早的位置，prefill 与 decode 的选择才一致
        # （官方用 topk(sorted=False)，平分时的选择不确定，但头数多时几乎不会出现整行平分）
        idxs = score.sort(dim=-1, descending=True, stable=True).indices
        idxs = idxs[..., :min(cfg.index_topk, n_comp)]
        return torch.where(idxs < compress_lens, idxs, -1)
# endregion indexer


# region attention
class CSA2Attention(nn.Module):
    def __init__(self, cfg: CSA2Config, layer: int) -> None:
        super().__init__()
        self.cfg, self.layer = cfg, layer
        self.ratio = cfg.compress_ratios[layer]
        self.mode = layer_modes(cfg.compress_ratios, cfg.kv_source_layers,
                                cfg.index_source_layers)[layer]
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.wkv = nn.Linear(cfg.dim, cfg.head_dim, bias=False)  # 每层都有自己的滑窗 KV
        self.kv_norm = RMSNorm(cfg.head_dim)
        self.attn_sink = nn.Parameter(torch.zeros(cfg.n_heads))
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)
        self.register_buffer("window_kv", torch.zeros(
            cfg.max_batch_size, cfg.max_seq_len, cfg.head_dim), persistent=False)
        self.compressor = self.indexer = None
        if self.mode == "full":
            self.compressor = Compressor2(cfg.dim, cfg.head_dim, self.ratio, cfg.max_batch_size)
            self.register_buffer("compress_kv", torch.zeros(
                cfg.max_batch_size, cfg.max_seq_len // self.ratio, cfg.head_dim), persistent=False)
        if self.mode in ("full", "reindex"):
            self.indexer = Indexer2(cfg, layer)

    def produce_global_kv(self, x: torch.Tensor, start_pos: int, state: SharedState) -> None:
        """Full 层的职责：压缩主 KV、追加索引器键，并把缓存发布到共享状态。"""
        latent = self.compressor(x, start_pos)
        if latent is not None:
            first = start_pos // self.ratio
            self.compress_kv[:x.size(0), first:first + latent.size(1)] = latent
            self.indexer.append_keys(latent, start_pos, state)
        state.compress_kv, state.index_k = self.compress_kv, self.indexer.k_cache

    def forward(self, x: torch.Tensor, start_pos: int, state: SharedState,
                window_start: int = 0, skip_global: bool = False) -> torch.Tensor:
        """``window_start``：滑窗不早于该位置（有界重放时的截断点）；
        ``skip_global``：全局 KV 已由别处产生（CED 的编码器侧 prefill），本层不再重算。"""
        cfg = self.cfg
        bsz, seqlen, _ = x.shape
        end = start_pos + seqlen
        q = self.wq(x).unflatten(-1, (cfg.n_heads, cfg.head_dim))
        self.window_kv[:bsz, start_pos:end] = self.kv_norm(self.wkv(x))
        kv = self.window_kv[:bsz, :end]
        pos = torch.arange(start_pos, end).unsqueeze(1)
        lo = (pos - cfg.window + 1).clamp(min=window_start)
        idxs = lo + torch.arange(cfg.window)
        idxs = torch.where(idxs > pos, -1, idxs).expand(bsz, -1, -1)
        if self.ratio:
            if self.mode == "full" and not skip_global:
                self.produce_global_kv(x, start_pos, state)
            if self.indexer is not None:
                state.topk_idxs = self.indexer(x, start_pos, state)
            comp = state.topk_idxs  # Reuse 层直接拿最近一次的索引
            kv = torch.cat([kv, state.compress_kv[:bsz, :end // self.ratio]], dim=1)
            idxs = torch.cat([idxs, torch.where(comp >= 0, comp + end, -1)], dim=-1)
        o = sparse_attn(q, kv, self.attn_sink, idxs, cfg.head_dim**-0.5)
        return self.wo(o.flatten(2))
# endregion attention


class Block(nn.Module):
    def __init__(self, cfg: CSA2Config, layer: int) -> None:
        super().__init__()
        self.attn_norm, self.ffn_norm = RMSNorm(cfg.dim), RMSNorm(cfg.dim)
        self.attn = CSA2Attention(cfg, layer)
        self.w1 = nn.Linear(cfg.dim, 2 * cfg.dim, bias=False)
        self.w3 = nn.Linear(cfg.dim, 2 * cfg.dim, bias=False)
        self.w2 = nn.Linear(2 * cfg.dim, cfg.dim, bias=False)

    def forward(self, x, start_pos, state, **kw):
        x = x + self.attn(self.attn_norm(x), start_pos, state, **kw)
        h = self.ffn_norm(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


# region model
class CSA2Model(nn.Module):
    def __init__(self, cfg: CSA2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.state = SharedState()

    def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """普通的逐层前向：prefill（start_pos = 0）或 decode，返回所有位置的 logits。"""
        h = self.embed(tokens)
        for layer in self.layers:
            h = layer(h, start_pos, self.state)
        return self.head(self.norm(h))

    @torch.no_grad()
    def prefill_ced(self, tokens: torch.Tensor, replay: int) -> tuple[torch.Tensor, int]:
        """CED 的 prefill：编码器跑全部 token；解码器只跑最后 ``replay`` 个 token（有界重放）。

        解码器的全局 KV 由第一个解码器层（Full）从编码器输出投影得到，覆盖整段 prompt；
        解码器各层的滑窗 KV 只在重放段内重建，窗口被截断在重放起点。
        返回最后一个位置的 logits 与解码器实际处理的 token 数。
        """
        cfg, T = self.cfg, tokens.size(1)
        h = self.embed(tokens)
        for layer in self.layers[:cfg.n_encoder_layers]:
            h = layer(h, 0, self.state)
        first_dec = self.layers[cfg.n_encoder_layers]
        assert first_dec.attn.mode == "full", "CED 要求第一个解码器层是 Full 模式"
        # 编码器输出 → 整个解码器的全局 KV（与逐层前向时 Full 层看到的输入完全相同）
        first_dec.attn.produce_global_kv(first_dec.attn_norm(h), 0, self.state)
        s = max(T - replay, 0)
        h = h[:, s:]
        for layer in self.layers[cfg.n_encoder_layers:]:
            h = layer(h, s, self.state, window_start=s, skip_global=True)
        return self.head(self.norm(h))[:, -1], T - s
# endregion model


def exact_replay_length(n_decoder_layers: int, window: int) -> int:
    """精确重建解码器最后一个位置所需的重放长度：每层感受野向前延伸 window-1 个位置。"""
    return n_decoder_layers * (window - 1) + 1
