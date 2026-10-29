"""DeepSeek-V4.1 的 KV 精度与持久化缓存：FP4 主 KV、MXFP4 索引器键、SWA 有界重放的代价。

FP4 E2M1 只有 8 个非负取值 {0, 0.5, 1, 1.5, 2, 3, 4, 6}，必须配合块缩放使用：
* 主 KV（V4.1）：每 16 个通道一个 E4M3 缩放（NVFP4 去掉第二级全局缩放），缩放本身有 3 位尾数；
* 索引器 Q/K（V4 起）：每 32 个通道一个 E8M0 缩放（OCP MXFP4），缩放只能是 2 的幂。
这里用 PyTorch 模拟“量化—反量化”，数值与真实 FP4 存储一致，只是不做位打包。
"""

from __future__ import annotations

import torch

E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MAX, E4M3_MAX = 6.0, 448.0


# region fp4
def round_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """把 |x| ≤ 6 的数舍入到最近的 E2M1 取值（保留符号）。"""
    mag = x.abs().clamp(max=E2M1_MAX)
    nearest = (mag.unsqueeze(-1) - E2M1_GRID).abs().argmin(-1)
    return E2M1_GRID[nearest] * x.sign()


def fp4_e4m3_roundtrip(x: torch.Tensor, block: int = 16) -> torch.Tensor:
    """V4.1 主 KV 格式：块内 amax 映射到 6，缩放用 E4M3 存储。"""
    tiles = x.unflatten(-1, (-1, block))
    scale = tiles.abs().amax(-1, keepdim=True) / E2M1_MAX
    scale = scale.clamp(max=E4M3_MAX).to(torch.float8_e4m3fn).float()  # 缩放本身也被量化
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    # 缩放被舍入后可能略小于 amax/6，先夹紧再舍入，避免越界
    return (round_to_e2m1((tiles / scale).clamp(-E2M1_MAX, E2M1_MAX)) * scale).flatten(-2)


def mxfp4_roundtrip(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """索引器格式：缩放向上取整到 2 的幂（E8M0），保证 amax / scale ≤ 6。"""
    tiles = x.unflatten(-1, (-1, block))
    amax = tiles.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    scale = torch.exp2(torch.ceil(torch.log2(amax / E2M1_MAX)))
    return (round_to_e2m1(tiles / scale) * scale).flatten(-2)
# endregion fp4


def max_representable_without_global_scale() -> float:
    """去掉 NVFP4 的第二级缩放后，单级 E4M3 缩放 × E2M1 最大值 = 448 × 6。"""
    return E4M3_MAX * E2M1_MAX


def kv_latent_bound(head_dim: int = 512) -> float:
    """RMSNorm（权重 ≤ 1）后向量的 L2 范数 ≤ √d，于是任一通道的绝对值 ≤ √d；RoPE 保持范数。"""
    return head_dim**0.5


# region replay
def swa_replay_tokens(n_layers: int, window: int, bounded: bool) -> int:
    """恢复全部 SWA 状态要重放的 token 数：精确重建需 L·n_win，V4.1 的有界重放只需 n_win。"""
    return window if bounded else n_layers * window


def persistent_cache_bytes(seq_len: int, global_per_token: float, swa_entry_bytes: float,
                           n_layers: int, window: int, swa_points: int) -> float:
    """一条会话写进持久化缓存（SSD）的字节数：全局 KV 全量 + 在 swa_points 个位置各存一份滑窗。

    V4：swa_points = 2（prompt 末尾与输出末尾）；V4.1：滑窗不再持久化，swa_points = 0。
    """
    return seq_len * global_per_token + swa_points * n_layers * window * swa_entry_bytes
# endregion replay
