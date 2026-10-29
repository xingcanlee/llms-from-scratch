export type PartId =
  | 'math'
  | 'pytorch'
  | 'transformer'
  | 'architecture'
  | 'pretraining'
  | 'distributed'
  | 'post-training'
  | 'gpu'
  | 'inference'
  | 'deepseek'

export type Part = {
  id: PartId
  /** 第一部分 … 第十部分 */
  numeral: string
  label: string
  summary: string
}

/** Reading order of the whole book. A part's chapters live in src/content/book/<id>/. */
export const parts: Part[] = [
  {
    id: 'math',
    numeral: '第一部分',
    label: '数学基础',
    summary: '向量与矩阵、导数与链式法则、反向传播与自动微分、概率与交叉熵、浮点数值与低秩分解——后面每一章都会用到的工具。',
  },
  {
    id: 'pytorch',
    numeral: '第二部分',
    label: 'PyTorch 与资源核算',
    summary: '张量的形状与存储、autograd、模块与训练循环；学会估算 FLOPs 与显存，并训练第一个神经语言模型。',
  },
  {
    id: 'transformer',
    numeral: '第三部分',
    label: '从零实现 Transformer',
    summary: '从 BPE 分词、注意力、位置编码到完整的 GPT 式 Decoder，亲手训练一个小模型并用 KV Cache 生成文本。',
  },
  {
    id: 'architecture',
    numeral: '第四部分',
    label: '现代架构：2017—2026',
    summary: '从原始 Transformer 到 LLaMA、DeepSeek-V3/V4 与 Qwen3：GQA 与 MLA、MoE、长上下文、稀疏与线性注意力、mHC 与多模态。',
  },
  {
    id: 'pretraining',
    numeral: '第五部分',
    label: '预训练',
    summary: '数据管线、优化器（AdamW 到 Muon）、训练稳定性、BF16/FP8 混合精度、缩放定律与评估。',
  },
  {
    id: 'distributed',
    numeral: '第六部分',
    label: '分布式训练',
    summary: '集合通信、数据并行与 ZeRO/FSDP、张量并行、流水线并行与 DualPipe、专家并行，以及如何组合它们。',
  },
  {
    id: 'post-training',
    numeral: '第七部分',
    label: '后训练与对齐',
    summary: 'SFT、LoRA、RLHF 与 PPO、DPO、GRPO 与 DeepSeek-R1 式推理强化学习，以及蒸馏。',
  },
  {
    id: 'gpu',
    numeral: '第八部分',
    label: 'GPU 与 CUDA 编程',
    summary: 'GPU 硬件与执行模型、CUDA kernel、访存优化、归约与 GEMM、Triton、算子融合、FlashAttention 与性能分析。',
  },
  {
    id: 'inference',
    numeral: '第九部分',
    label: '推理系统',
    summary: 'prefill 与 decode 的成本模型、PagedAttention、连续批处理、CUDA Graph、vLLM 架构、量化、投机解码与分布式推理。',
  },
  {
    id: 'deepseek',
    numeral: '第十部分',
    label: 'DeepSeek 技术全景',
    summary: '对照官方推理代码读懂 DeepSeek 从 V2 到 V4.1 的全部关键设计：MLA 与分组路由、DSA、压缩注意力、CSA2 与因果编码器—解码器、FP4 KV 与有界重放、Engram、mHC、DSpark 与训练配方。',
  },
]

export function getPart(id: string): Part {
  const part = parts.find((item) => item.id === id)
  if (!part) throw new Error(`Unknown part "${id}"`)
  return part
}
