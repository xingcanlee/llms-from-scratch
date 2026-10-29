"""Engram 条件记忆（DeepSeek-V4.1-Flash/inference/engram.py 与 model.py 中的 ``Engram``）。

数据流：token id → 压缩词表 id（大小写、空白归一）→ 对每个位置取以它结尾的 2..N-gram →
乘法—异或滚动哈希 → 每个（n-gram 阶, 哈希头）一个互不相同的素数桶 → 查一张大表 →
投影成 hc_mult 个键与一个共享值 → 用“残差流 · 键”的归一化点积做门控，加回残差流。
哈希只依赖 token 序列，所以可以在前向开始前就算好，并从主机内存预取表行。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def find_next_prime(start: int, seen: set[int]) -> int:
    """大于 start 且尚未被分配的最小素数（官方用 sympy.isprime）。"""
    c = start + 1
    while not is_prime(c) or c in seen:
        c += 1
    return c


# region layout
@dataclass(frozen=True)
class EngramLayout:
    max_ngram_size: int
    n_heads: int
    primes: tuple[tuple[tuple[int, ...], ...], ...]  # [层][n-gram 阶][头] 的桶大小

    @classmethod
    def build(cls, n_layers: int, max_ngram_size: int, n_heads: int, vocab_size: int):
        """所有（层, 阶, 头）的素数依次从 vocab_size 往上找、互不重复，所以桶区间不相交。"""
        primes, seen = [], set()
        for _ in range(n_layers):
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], vocab_size - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(max_ngram_size, n_heads, tuple(primes))

    def num_embeddings(self, layer: int) -> int:
        return sum(p for per_ngram in self.primes[layer] for p in per_ngram)
# endregion layout


def hash_multipliers(layer_ids: tuple[int, ...], max_ngram_size: int,
                     compressed_vocab: int) -> torch.Tensor:
    """每层一组奇数乘子，由 seed = 10007·layer 的 RNG 生成；上界保证 id·乘子 不溢出 int64。"""
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab) // 2)
    rows = []
    for layer in layer_ids:
        rng = np.random.default_rng(10007 * layer)
        values = rng.integers(0, bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


# region hash
DEAD = -1  # 图像 token 等不参与 n-gram 的位置


def ngram_hash_ids(compressed: torch.Tensor, layout: EngramLayout, multipliers: torch.Tensor,
                   pad_id: int) -> torch.Tensor:
    """compressed: [B, T] 压缩词表 id（DEAD 表示断点）→ 表行号 [B, T, n_layers, (N-1)·n_heads]。

    回看越过序列开头或遇到 DEAD 就用 pad_id 填充，所以 n-gram 永远不会跨越断点。
    """
    B, T = compressed.shape
    pos = torch.arange(T).expand(B, T)
    tokens, blocked = [], torch.zeros(B, T, dtype=torch.bool)
    for shift in range(layout.max_ngram_size):
        src = compressed.gather(1, (pos - shift).clamp_min(0))
        blocked = blocked | (pos < shift) | (src == DEAD)
        tokens.append(torch.where(blocked, pad_id, src))
    tokens = torch.stack(tokens, dim=-1)  # [B, T, N]：当前 token、前 1 个、前 2 个……
    primes = torch.tensor(layout.primes)  # [L, N-1, heads]
    flat = primes.flatten(1)
    offsets = torch.cumsum(flat, dim=1) - flat  # 每个（阶, 头）的桶在表内的起点
    products = tokens.unsqueeze(2) * multipliers  # [B, T, L, N]
    rolling, hashes = products[..., 0], []
    for i in range(1, layout.max_ngram_size):
        rolling = torch.bitwise_xor(rolling, products[..., i])  # 此时是 (i+1)-gram 的哈希
        hashes.append(rolling.unsqueeze(-1) % primes[:, i - 1])
    return torch.cat(hashes, dim=-1) + offsets
# endregion hash


# region module
class Engram(nn.Module):
    def __init__(self, num_embeddings: int, n_hash_cols: int, head_dim: int, dim: int,
                 hc_mult: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim, self.hc_mult, self.eps = dim, hc_mult, eps
        self.embed = nn.Embedding(num_embeddings, head_dim)
        self.wkv = nn.Linear(n_hash_cols * head_dim, dim * (hc_mult + 1), bias=False)
        self.qk_weight = nn.Parameter(torch.ones(hc_mult, dim))  # 官方 q_weight·k_weight 的乘积

    def forward(self, x: torch.Tensor, hash_ids: torch.Tensor,
                token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, T, hc, d]（mHC 的多路残差流），hash_ids: [B, T, n_hash_cols]。"""
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.unflatten(-1, (self.hc_mult, self.dim))
        # 每条流各自与自己的键做“双 RMS 归一”的点积，相当于余弦相似度乘以 √d
        rstd = (torch.rsqrt(x.square().mean(-1) + self.eps)
                * torch.rsqrt(key.square().mean(-1) + self.eps))
        dot = (x * self.qk_weight * key).sum(-1) * rstd * self.dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))  # 带符号开方
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        return x + gate.unsqueeze(-1) * value.unsqueeze(-2)  # 同一个值，按流各自的门写入
# endregion module
