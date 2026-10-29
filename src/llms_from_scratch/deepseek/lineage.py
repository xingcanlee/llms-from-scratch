"""DeepSeek 开源模型谱系与“全局 KV Cache 账本”。

``RELEASES`` 收录 deepseek-ai 在 Hugging Face 上发布的主要开源权重（截至 2026-10-30），
``global_kv_bytes_per_token`` 按各代官方 config 与推理代码中的缓存格式，逐层累加每个 token
需要长期保存的 KV 字节数。滑动窗口 KV 的大小与序列长度无关，不计入“全局 KV”。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class Release:
    date: str  # Hugging Face 仓库创建日期
    repo: str  # huggingface.co/deepseek-ai/<repo>
    family: str
    total_params: str  # 官方模型卡给出的总参数（MoE 为总参数 / 激活参数）
    note: str = ""


RELEASES: tuple[Release, ...] = (
    Release("2023-10-28", "deepseek-coder-33b-base", "Coder", "1.3B/6.7B/33B", "稠密，16K 上下文"),
    Release("2023-11-29", "deepseek-llm-67b-base", "LLM", "7B/67B", "V1：稠密，67B 用 GQA"),
    Release("2024-01-08", "deepseek-moe-16b-base", "MoE", "16.4B / 2.8B", "细粒度专家 + 共享专家"),
    Release("2024-02-05", "deepseek-math-7b-rl", "Math", "7B", "首次提出 GRPO"),
    Release("2024-03-07", "deepseek-vl-7b-chat", "VL", "1.3B/7B", "混合视觉编码器"),
    Release("2024-04-22", "DeepSeek-V2", "V2", "236B / 21B", "MLA + DeepSeekMoE"),
    Release("2024-05-15", "DeepSeek-V2-Lite", "V2", "15.7B / 2.4B", "可单卡运行的 MLA 模型"),
    Release("2024-06-14", "DeepSeek-Coder-V2-Instruct", "Coder", "236B / 21B", "基于 V2 继续训练"),
    Release("2024-07-04", "ESFT-vanilla-lite", "ESFT", "15.7B / 2.4B", "只微调相关专家"),
    Release("2024-08-15", "DeepSeek-Prover-V1.5-RL", "Prover", "7B", "Lean 4 定理证明"),
    Release("2024-09-05", "DeepSeek-V2.5", "V2", "236B / 21B", "Chat 与 Coder 合并"),
    Release("2024-10-18", "Janus-1.3B", "Janus", "1.3B", "理解与生成解耦的视觉编码"),
    Release("2024-12-13", "deepseek-vl2", "VL", "27.5B / 4.5B", "MoE 视觉语言模型"),
    Release("2024-12-25", "DeepSeek-V3", "V3", "671B / 37B", "无辅助损失均衡、MTP、FP8 训练"),
    Release("2025-01-20", "DeepSeek-R1", "R1", "671B / 37B", "大规模 RL 推理模型；另有 6 个蒸馏模型"),
    Release("2025-01-26", "Janus-Pro-7B", "Janus", "1B/7B", ""),
    Release("2025-03-24", "DeepSeek-V3-0324", "V3", "671B / 37B", ""),
    Release("2025-04-30", "DeepSeek-Prover-V2-671B", "Prover", "671B / 37B", ""),
    Release("2025-05-28", "DeepSeek-R1-0528", "R1", "671B / 37B", ""),
    Release("2025-08-21", "DeepSeek-V3.1", "V3.1", "671B / 37B", "思考/非思考混合"),
    Release("2025-09-22", "DeepSeek-V3.1-Terminus", "V3.1", "671B / 37B", "V3.2 的起点检查点"),
    Release("2025-09-29", "DeepSeek-V3.2-Exp", "V3.2", "671B / 37B", "DSA 稀疏注意力"),
    Release("2025-10-17", "DeepSeek-OCR", "OCR", "3B / 0.57B", "光学上下文压缩"),
    Release("2025-11-27", "DeepSeek-Math-V2", "Math", "685B", "自验证的定理证明"),
    Release("2025-12-01", "DeepSeek-V3.2", "V3.2", "685B / 37B", "另有 V3.2-Speciale"),
    Release("2026-01-27", "DeepSeek-OCR-2", "OCR", "3B", ""),
    Release("2026-04-22", "DeepSeek-V4-Flash", "V4", "284B / 13B", "CSA/HCA、mHC、1M 上下文"),
    Release("2026-04-22", "DeepSeek-V4-Pro", "V4", "1.6T / 49B", ""),
    Release("2026-06-27", "DeepSeek-V4-Flash-DSpark", "V4", "284B / 13B", "附加 DSpark 投机解码模块"),
    Release("2026-07-31", "DeepSeek-V4-Flash-0731", "V4", "284B / 13B", "V4-Flash 正式版"),
    Release("2026-08-13", "DeepSeek-V4-Pro-0813", "V4", "1.6T / 49B", "V4-Pro 正式版"),
    Release("2026-08-31", "DeepSeek-V4-Flash-Vision-Exp", "V4", "284B / 13B", "实验性多模态"),
    Release("2026-09-10", "DeepSeek-V4.1-Flash", "V4.1", "552B / 8B·16B", "CED、CSA2；另有 196B Engram 参数"),
    Release("2026-10-17", "DeepSeek-V4.1-Flash-1017", "V4.1", "552B / 8B·16B", "V4.1-Flash 小版本更新"),
)


# region kv_formats
BF16, FP8 = 2.0, 1.0


def fp8_bytes(dim: int, block: int) -> float:
    """FP8 E4M3 数值 + 每 block 个通道一个 1 字节缩放（UE8M0 或 E4M3）。"""
    return dim * FP8 + ceil(dim / block)


def fp4_bytes(dim: int, block: int) -> float:
    """FP4 E2M1 数值（两个数一字节）+ 每 block 个通道一个 1 字节缩放。"""
    return dim / 2 + ceil(dim / block)
# endregion kv_formats


# region ledger
def gqa_bytes(n_layers: int, n_kv_heads: int, head_dim: int) -> float:
    """标准 K/V 缓存：每层 2 · n_kv_heads · head_dim 个 BF16。"""
    return n_layers * 2 * n_kv_heads * head_dim * BF16


def mla_bytes(n_layers: int, kv_rank: int = 512, rope_dim: int = 64,
              index_dim: int = 0) -> float:
    """V2/V3：每层一份潜向量 c_kv 与解耦位置键 k_R（BF16）。

    ``index_dim > 0`` 时按 V3.2 的部署格式计：潜向量 FP8（每 128 维一个 FP32 缩放），
    k_R 保持 BF16，再加上索引器键（FP8 + 一个 FP32 缩放）。
    """
    if not index_dim:
        return n_layers * (kv_rank + rope_dim) * BF16
    main = kv_rank * FP8 + 4 * (kv_rank // 128) + rope_dim * BF16
    return n_layers * (main + index_dim * FP8 + 4)


def v4_bytes(compress_ratios: list[int], head_dim: int = 512, rope_dim: int = 64,
             index_dim: int = 128) -> float:
    """V4：每个压缩层存 1/m 个共享 KV 条目；比率为 4 的 CSA 层还带 FP4 索引器键。

    主 KV 的非 RoPE 部分 FP8（每 64 维一个 UE8M0 缩放），RoPE 部分 BF16。
    """
    entry = fp8_bytes(head_dim - rope_dim, 64) + rope_dim * BF16
    total = 0.0
    for m in compress_ratios:
        if m:
            total += (entry + (fp4_bytes(index_dim, 32) if m == 4 else 0)) / m
    return total


def v41_bytes(compress_ratios: list[int], kv_source_layers: list[int], head_dim: int = 512,
              index_dim: int = 128) -> float:
    """V4.1 CSA2：只有 Full 模式层（kv_source）产生主 KV 与索引器键，其余层复用。

    主 KV 为 FP4 E2M1 + 每 16 维一个 E4M3 缩放，索引器键为 MXFP4（每 32 维一个 UE8M0）。
    """
    entry = fp4_bytes(head_dim, 16) + fp4_bytes(index_dim, 32)
    return sum(entry / compress_ratios[layer] for layer in kv_source_layers)
# endregion ledger


V4_FLASH_RATIOS = [0, 0] + [4, 128] * 20 + [4]  # 43 层；官方 config 第 44 项属于 MTP 层
V41_FLASH_RATIOS = [0, 0] + [2] * 18 + [1] * 20  # 40 层（不含 3 个 DSpark 层）
V41_KV_SOURCES = [2, 8, 14, 20]


def global_kv_bytes_per_token() -> dict[str, float]:
    """各代代表模型每 token 的全局 KV 字节数（单个序列、全部层）。"""
    return {
        "DeepSeek-V1 67B": gqa_bytes(95, 8, 128),
        "DeepSeek-V2": mla_bytes(60),
        "DeepSeek-V3": mla_bytes(61),
        "DeepSeek-V3.2": mla_bytes(61, index_dim=128),
        "DeepSeek-V4-Flash": v4_bytes(V4_FLASH_RATIOS),
        "DeepSeek-V4.1-Flash": v41_bytes(V41_FLASH_RATIOS, V41_KV_SOURCES),
    }
