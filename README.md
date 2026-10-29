# LLMs From Scratch

一本层层递进的中文技术书：从链式法则与反向传播出发，亲手实现 Transformer，沿 2017—2026 年的架构演进走到 DeepSeek-V4，再讲清预训练、分布式训练、后训练与对齐，再深入 GPU/CUDA 编程与 vLLM 式推理系统，最后对照官方源码完整拆解 DeepSeek V2—V4.1 的核心技术。路线参考 Stanford CS336《Language Modeling from Scratch》。

**在线阅读：<https://xingcanlee.github.io/llms-from-scratch/>**

| 部分 | 内容 |
| --- | --- |
| 一、数学基础 | 张量运算、梯度与链式法则、反向传播、从零实现自动微分、概率与交叉熵、浮点数值、SVD 与低秩 |
| 二、PyTorch 与资源核算 | 张量的存储与视图、autograd、模块与训练循环、FLOPs/显存/MFU 估算、第一个神经语言模型 |
| 三、从零实现 Transformer | BPE、注意力、多头与掩码、RoPE、归一化与 SwiGLU、组装与训练 GPT、采样与 KV Cache |
| 四、现代架构：2017—2026 | 架构演进、GQA 与 MLA、长上下文、MoE、稀疏与线性注意力、mHC、MTP、多模态 |
| 五、预训练 | 数据管线、AdamW 与 Muon、训练稳定性、BF16/FP8 混合精度、缩放定律、评估 |
| 六、分布式训练 | 集合通信、DDP/ZeRO/FSDP、张量并行、流水线并行与 DualPipe、专家并行、并行组合 |
| 七、后训练与对齐 | SFT、LoRA、RLHF 与 PPO、DPO、GRPO 与推理模型、大规模 RL、蒸馏 |
| 八、GPU 与 CUDA 编程 | GPU 架构、CUDA kernel、访存优化、归约与 GEMM、Triton、算子融合、FlashAttention、性能分析 |
| 九、推理系统 | prefill/decode 成本模型、PagedAttention、连续批处理、CUDA Graph、vLLM、量化、投机解码、分布式推理、迷你推理引擎 |
| 十、DeepSeek 技术全景 | 开源模型谱系与 KV 账本、V3 官方代码（MLA/分组路由/FP8）、V3.2 DSA、V4 压缩注意力、V4.1 CSA2 与因果编码器—解码器、FP4 KV 与有界重放、Engram 与 Single-Pass mHC、DSpark、训练配方 |

## 仓库结构

```text
site/src/content/book/<part>/   全书正文（MDX），每个部分一个目录，index.mdx 为导读
src/llms_from_scratch/<part>/   各部分的参考实现；transformer/model.py 是全书共用的 GPT
tests/<part>/                   配套测试（CPU 即可运行；GPU 测试在无硬件时跳过）
site/                           Astro 站点：布局、图元组件、样式
docs/BOOK_GUIDE.md              写作、组件、图与代码规范
```

## 运行代码

需要 Python 3.11+ 与 [uv](https://docs.astral.sh/uv/)。默认环境使用 CPU 版 PyTorch：

```sh
uv sync --locked --group dev
uv run pytest
uv run ruff check src tests
```

GPU 章节的 CUDA 与 Triton 示例需要 NVIDIA GPU；建议使用独立环境安装与驱动匹配的 PyTorch：

```sh
uv venv .venv-gpu
uv pip install --python .venv-gpu/bin/python -e '.[gpu]'
```

## 本地站点

```sh
cd site
pnpm install
pnpm dev      # 本地预览
pnpm check    # 类型检查
pnpm build    # 构建并检查全部站内链接
```

`<CodeFile>` 组件在构建时调用系统 `python3`（仅标准库）从源码摘录函数与类。修改正文前请阅读 [写作与排版规范](docs/BOOK_GUIDE.md)。
