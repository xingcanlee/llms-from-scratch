"""DSpark：半自回归草稿 + 置信度调度的验证（V4.1 model.py 的 ``DSparkBlock.forward_head``）。

草稿模块一次前向为 block_size 个位置并行给出“基础 logits”；Markov 头再按刚采样出的上一个
草稿 token 给下一位置加一个低秩偏置（embed: V→r，head: r→V），补上块内 token 间的依赖。
置信度头为每个位置预测“在前缀都被接受的条件下，本位置被接受”的概率 c_i，前缀存活概率
S_j = Π_{i≤j} c_i 决定验证几个 token 最划算。

调度器在这里被简化为：给定每个请求的 c 和一条“一步验证 n 个 token 所需时间”的曲线，
选每个请求的验证长度使整批的期望吞吐最大。论文的完整调度策略见参考文献。
"""

from __future__ import annotations

from collections.abc import Callable

import torch


# region draft
def markov_draft(base_logits: torch.Tensor, first_token: torch.Tensor, markov_embed: torch.Tensor,
                 markov_head: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """base_logits: [B, k, V]（并行算出），first_token: [B]。贪心解码 k 个草稿 token。

    返回草稿 [B, k] 和每个位置的 Markov 嵌入 [B, k, r]（供置信度头使用）。
    """
    B, k, _ = base_logits.shape
    tokens = [first_token]
    embeds = []
    for i in range(k):  # 只有这一个廉价的低秩循环是串行的
        e = markov_embed[tokens[-1]]  # [B, r]
        logits = base_logits[:, i] + e @ markov_head.T  # 依赖上一个草稿 token 的偏置
        embeds.append(e)
        tokens.append(logits.argmax(-1))
    return torch.stack(tokens[1:], dim=1), torch.stack(embeds, dim=1)
# endregion draft


# region schedule
def prefix_survival(conf: torch.Tensor) -> torch.Tensor:
    """S_j = Π_{i≤j} c_i：前 j 个草稿全部被接受的概率。"""
    return torch.cumprod(conf, dim=-1)


def expected_tokens(conf: torch.Tensor, verify_len: int) -> float:
    """验证前 verify_len 个草稿时，一步得到的期望 token 数 = 1（目标模型自己的一个）+ Σ_{j≤L} S_j。"""
    return 1.0 + prefix_survival(conf)[:verify_len].sum().item()


def schedule_verify_lengths(confs: list[torch.Tensor],
                            step_time: Callable[[int], float]) -> list[int]:
    """为一批请求选验证长度，最大化 期望 token 数 / 一步耗时。

    固定总验证 token 数 n 时，最优解是把 n 个“名额”依次给边际收益 S 最大的位置（每个请求的
    S_j 单调递减，所以贪心最优）；再在所有 n 中选吞吐最高的。每个请求至少占 1 个 token。
    """
    gains = sorted(((s.item(), r) for r, c in enumerate(confs) for s in prefix_survival(c)),
                   reverse=True)
    lengths = [0] * len(confs)
    best, best_lengths = len(confs) / step_time(len(confs)), list(lengths)
    total_tokens, expected = len(confs), float(len(confs))
    for gain, r in gains:
        lengths[r] += 1
        total_tokens += 1
        expected += gain
        rate = expected / step_time(total_tokens)
        if rate > best:
            best, best_lengths = rate, list(lengths)
    return best_lengths
# endregion schedule
