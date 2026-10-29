"""DeepSeek-V4.1 的优化器改动（技术报告 §2.5 与算法 1）与按模态的无辅助损失均衡。

* ``sinkhorn_balanced_update``：嵌入表、Engram 表与输出头的更新。与 Muon 同样只需一个动量
  缓冲，但把 Newton–Schulz 正交化换成交替的行、列 ℓ2 归一化，使更新矩阵的行 RMS 与列 RMS 都约为 1。
* ``headwise_orthogonalize``：按头切开 Q/K 权重，每个头单独做 Newton–Schulz（按头 Muon）。
* ``update_modal_bias``：文本 token 与图像 token 各一组路由偏置，各按本模态的专家负载调整。
* ``effort_rewards``：可控推理强度（1–100）的长度惩罚与按子组计算的优势（报告 §5.1.4）。
"""

from __future__ import annotations

import math

import torch

from llms_from_scratch.pretraining.optimizers import newton_schulz


# region sinkhorn_update
def sinkhorn_balance(G: torch.Tensor, K: int = 11, tau: float = 1e-3,
                     eps: float = 1e-20) -> torch.Tensor:
    """算法 1 的第 3–17 行：G 为 m×n（m = 词表或 n-gram 数，n = 隐藏维）。"""
    assert K % 2 == 1, "K 为奇数，保证最后一步是行归一化"
    m, n = G.shape
    rho = G.norm(dim=1)
    U = torch.where((rho <= tau * rho.mean()).unsqueeze(1), torch.zeros_like(G), G)  # 屏蔽近零行
    for k in range(1, K + 1):
        if k % 2:  # 奇数步：每行除以行 ℓ2 范数
            U = U / (U.norm(dim=1, keepdim=True) + eps)
        else:  # 偶数步：每列除以列 ℓ2 范数
            U = U / (U.norm(dim=0, keepdim=True) + eps)
    return n**0.5 * U  # 行 ℓ2 = 1 → 行 RMS = 1


def sinkhorn_balanced_update(W: torch.Tensor, grad: torch.Tensor, momentum: torch.Tensor,
                             lr: float, beta: float = 0.95, gamma: float = 0.18,
                             K: int = 11) -> None:
    """原地更新 W 与动量（Nesterov 动量，无权重衰减）。"""
    momentum.mul_(beta).add_(grad, alpha=1 - beta)
    nesterov = beta * momentum + (1 - beta) * grad
    W.sub_(gamma * lr * sinkhorn_balance(nesterov, K))
# endregion sinkhorn_update


# region headwise
def headwise_orthogonalize(G: torch.Tensor, n_heads: int) -> torch.Tensor:
    """G: [n_heads·d_h, d]（Linear 布局的 Q 或 K 权重梯度）→ 每个头的 [d_h, d] 块各自正交化。"""
    blocks = G.unflatten(0, (n_heads, -1))
    return torch.stack([newton_schulz(b) for b in blocks]).flatten(0, 1)
# endregion headwise


# region modal_bias
def update_modal_bias(bias: dict[str, torch.Tensor], indices: torch.Tensor,
                      modality: torch.Tensor, speed: float = 1e-3) -> None:
    """bias: {"text": [E], "image": [E]}；indices: [N, k] 选中的专家；modality: [N] bool（图像为真）。

    与 V3 的规则相同：负载高于本模态平均的专家偏置减 speed，低于平均的加 speed。
    """
    for name, mask in (("text", ~modality), ("image", modality)):
        if not mask.any():
            continue
        load = torch.bincount(indices[mask].flatten(), minlength=bias[name].numel()).float()
        bias[name] -= speed * torch.sign(load - load.mean())
# endregion modal_bias


# region effort
def effort_penalty_coef(b: float, k0: float, b_min: float, tau: float) -> float:
    """可控推理强度（V4.1 报告式 (10)）：k(b) = k0 · exp(-(b - b_min) / τ)，b 每增加 τ 惩罚乘 e^-1。"""
    return k0 * math.exp(-(b - b_min) / tau)


def effort_rewards(task_reward: torch.Tensor, lengths: torch.Tensor, effort: torch.Tensor,
                   k0: float, b_min: float, tau: float, l_norm: float, c_max: float) -> torch.Tensor:
    """r = 任务奖励 - min(C_max, k(b)·ℓ/L_norm)，再在同一 (prompt, b) 子组内减均值得到优势。

    task_reward / lengths / effort 都是 [N]，同一 prompt 的回答；不同 b 的回答互不比较。
    """
    coef = torch.tensor([effort_penalty_coef(b.item(), k0, b_min, tau) for b in effort])
    reward = task_reward - torch.minimum(torch.tensor(c_max), coef * lengths / l_norm)
    adv = torch.zeros_like(reward)
    for b in effort.unique():
        m = effort == b
        adv[m] = reward[m] - reward[m].mean()
    return adv
# endregion effort
