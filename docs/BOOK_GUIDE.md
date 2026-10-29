# 写作与排版规范

本书是一本层层递进的中文技术书，参考 Stanford CS336《Language Modeling from Scratch》的路线：数学 → PyTorch 与资源核算 → 从零实现 Transformer → 现代架构 → 预训练 → 分布式训练 → 后训练 → GPU 编程 → 推理系统 → DeepSeek 技术全景（对照官方源码的综合案例）。正文唯一来源是 `site/src/content/book/<part>/<slug>.mdx`；参考实现在 `src/llms_from_scratch/<part>/`，测试在 `tests/<part>/`。

## 1. 全书结构

| 目录 | 部分 | Python 子包 |
| --- | --- | --- |
| `math` | 第一部分 数学基础 | `llms_from_scratch.math` |
| `pytorch` | 第二部分 PyTorch 与资源核算 | `llms_from_scratch.pytorch` |
| `transformer` | 第三部分 从零实现 Transformer | `llms_from_scratch.transformer` |
| `architecture` | 第四部分 现代架构：2017—2026 | `llms_from_scratch.architecture` |
| `pretraining` | 第五部分 预训练 | `llms_from_scratch.pretraining` |
| `distributed` | 第六部分 分布式训练 | `llms_from_scratch.distributed` |
| `post-training` | 第七部分 后训练与对齐 | `llms_from_scratch.post_training` |
| `gpu` | 第八部分 GPU 与 CUDA 编程 | `llms_from_scratch.gpu` |
| `inference` | 第九部分 推理系统 | `llms_from_scratch.inference` |
| `deepseek` | 第十部分 DeepSeek 技术全景 | `llms_from_scratch.deepseek` |

- 部分的元数据在 `site/src/lib/parts.ts`。章节顺序只由 frontmatter 的 `order` 决定；全书章号（第 1 章……第 68 章）、节号（3.2）、图号（图 3-2）由站点自动生成。
- 每个部分的 `index.mdx`（`order: 0`）是导读：这一部分解决什么问题、与前后部分的关系、阅读建议，末尾放 `<ChapterList part="…" />`。
- `src/llms_from_scratch/transformer/model.py` 是全书共用的 GPT（RMSNorm + RoPE + GQA + SwiGLU + KV Cache）。后续部分直接 import 它，不要复制一份。它的公开接口写在模块 docstring 中，修改时必须保持兼容并让 `tests/transformer/test_model.py` 通过。

## 2. Frontmatter

```yaml
---
title: 链式法则与计算图          # 短标题，用于 h1、侧栏、目录
description: 一两句话说明本章讲什么、读完能做什么。   # 页面导语与 meta
order: 3
code:                            # 本章对应的实现与测试（仓库相对路径）
  - src/llms_from_scratch/math/autodiff.py
  - tests/math/test_autodiff.py
draft: false                     # 写完后删掉这一行或设为 false
---
```

## 3. 章节写法：技术书，而不是讲义碎片

**层层递进。** 每章开头用一两段话把读者从上一章带过来：上一章解决了什么，留下了什么问题，本章怎样回答。结尾的小结同样指向下一章。全书只在第一次出现时完整推导一个结论，之后链接回去。

**每章的骨架**（标题按内容命名，不要照抄这些词）：

1. 引言段落（没有标题）：动机、要回答的问题、先修。
2. 若干个二级标题的主体小节。每个核心概念按 **直觉 → 定义/推导 → 手算小例子 → PyTorch 实现 → 验证** 推进。推导写完整，关键一步不跳；公式前后都有文字说明符号与含义。
3. `## 小结`：3–6 条要点。
4. `## 练习`：3–6 题，用 `<details>` 折叠答案；答案给出推理过程。
5. `## 参考文献`：论文、官方文档、源码链接（优先 arXiv、官方博客、GitHub 源码的固定行）。

**标题**：二级标题是名词短语或“名词：问题”，**不写编号**（页面自动编号）。三级标题用于小节内部的具体问题，不要四级以下。

**篇幅**：一章通常 5000–12000 汉字（不含代码），配 2–6 幅图、若干代码片段。宁可拆节，不要堆砌列表；正文以完整段落为主，列表只用于真正并列的条目。

**准确性**：涉及 2024—2026 年的模型与系统（DeepSeek-V3/V3.2/V4、Qwen3/Qwen3-Next、Kimi K2、Muon、GRPO 变体、vLLM V1、FlashAttention-3 等），必须先检索一手资料（论文、技术报告、官方仓库）再写，并在参考文献中列出。数字（参数量、FLOPs、带宽、显存）要给出来源或推导过程。不确定的内容明确写出不确定，不要编造。

**语言**：简体中文。中文与 `**加粗**` 相邻时无需加空格（站点启用了 CJK 友好的强调解析），术语首次出现写“中文（English）”，之后用约定写法；API 与代码标识符保留原文。避免口号式表述和“我们可以看到”式填充。

## 4. 数学记号

- 标量、向量小写，矩阵大写，默认列向量；批量输入 $X\in\mathbb R^{B\times d}$ 按行存放样本。线性层 $y = xW^\top + b$ 与 `torch.nn.Linear` 一致；注意力沿用 $Q = XW_Q$ 的论文写法时要说明布局。
- 张量形状写作 $(B, T, d)$ 或代码风格 `[B, T, D]`，同一章保持一致。常用维度：批 $B$、序列 $T$、模型宽度 $d$（`d_model`）、头数 $h$、头维 $d_h$、词表 $V$、层数 $L$。
- 损失：单样本 $\ell$，批平均 $\mathcal L$；学习率 $\eta$；$\log$ 为自然对数。
- 梯度与参数同形状；雅可比 $J_{ij}=\partial y_i/\partial x_j$，反向传播为 $J^\top g$（VJP）。
- 概率 $P(\cdot)$、密度 $p(\cdot)$、期望 $\mathbb E[\cdot]$。KaTeX 以 `strict` 模式渲染：不要在数学中写中文（用 `\text{}` 包裹），不要用 KaTeX 不支持的宏。

## 5. 组件

以下组件无需 import，直接在 MDX 中使用。

### 文本组件

| 组件 | 用途 |
| --- | --- |
| `<Callout tone="note \| tip \| warn \| deep" title="…">` | note 背景说明；tip 实践建议；warn 陷阱与边界；deep 选读的深入内容 |
| `<Definition term="…" kind="定义 \| 定理 \| 命题" source="url">` | 需要读者停下来记住的定义或定理 |
| `<KeyEq label="…">` + 空行 + `$$…$$` + 空行 + `</KeyEq>` | 一章围绕的 1–3 个关键公式 |
| `<CodeFile path="src/llms_from_scratch/…py" symbol="Name" />` | 从仓库摘录真实源码；`symbol` 可为数组或 `Class.method`；也可用 `region="name"` 摘录 `# region name` … `# endregion` 标记块（CUDA 用 `//`）。完整文件自动折叠在下方 |
| `<Steps label="…" items={[{ title, detail }]} />` | 横向的少量步骤 |
| `<StatGrid items={[{ value, label }]} />` | 少量关键数字 |
| `<Panels>` + `<Panel title="…" tone="accent \| accent2 \| info">` | 并列对比 |
| `<ChapterList part="…" />` | 部分导读中的章节列表 |
| `<details>` / `<summary>` | 练习答案与可选细节；两个标签在 MDX 中分行写，内容前后留空行 |

### 代码块

````md
```python title="attention.py" showLineNumbers {3-4}
def attention(q, k, v):
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(q.shape[-1])   # 高亮行
    return scores.softmax(-1) @ v
```
````

- `title="…"` 给代码块加文件名标题；`showLineNumbers` 显示行号；`{3-4}` 高亮行。
- 行尾注释 `# [!code highlight]`、`# [!code ++]`、`# [!code --]`、`# [!code focus]` 分别高亮、标记增删、聚焦。
- 正文中的短片段可以直接写代码块；凡是“完整实现”，放进 `src/llms_from_scratch/` 并用 `<CodeFile>` 摘录，保证书中代码就是被测试的代码。

### 图：统一用图元组件绘制

**不要使用 Mermaid，不要手写独立的 SVG 文件，不要用图片。** 所有图用下面的组件绘制，颜色自动跟随明暗主题，风格全书一致。

画布坐标为 SVG 用户单位，按宽度 **720** 绘制（约等于正文栏 1:1 像素）；窄屏自动横向滚动。每幅图都必须有 `label`（无障碍描述）和 `caption`（中文图注：怎样读这幅图），图后正文要解释图。

```mdx
<Diagram width={720} height={200} label="两层网络的计算图">
  <Box x={20} y={70} w={100} h={56} label="x" sub="(B, d)" />
  <Box x={200} y={70} w={130} h={56} label="z = xWᵀ + b" tone="accent" mono />
  <Box x={580} y={70} w={120} h={56} label="ℓ" tone="strong" />
  <Arrow from={[120, 98]} to={[200, 98]} label="前向" />
  <Arrow from={[640, 126]} via={[[640, 170], [80, 170]]} to={[80, 126]} tone="accent2" dashed label="梯度" labelSide="below" />
  <Fragment slot="caption">实线是前向依赖，虚线是反向传播返回梯度的路径。</Fragment>
</Diagram>
```

| 图元 | 主要参数 |
| --- | --- |
| `<Diagram width height label minWidth? maxWidth?>` | 画布与编号图注（`<Fragment slot="caption">`） |
| `<Box x y w h label? sub? tone? size? mono? title?>` | 矩形节点；`label` 可用 `\n` 换行；`sub` 是第二行小字（如形状）；`title` 把标签放到左上角作容器 |
| `<Arrow from to via? tone? dashed? head="end\|both\|none" label? labelSide? curve?>` | 直线/折线/曲线箭头 |
| `<Text x y label? anchor? size? tone? mono? bold? rotate?>` | 自由文字 |
| `<Matrix x y rows cols cell? values? highlight? rowLabels? colLabels? title? gap? digits?>` | 矩阵、注意力图、内存行；`highlight={[{ rows:[0], tone:'accent' }, { cells:[[1,2]], tone:'accent2' }, { upper:true, tone:'muted' }]}` |
| `<Region x y w h title? tone?>` | 分组容器（虚线），如“GPU 0”“SM”“Encoder” |
| `<Legend x y items={[{ label, tone }]} gap?>` | 图例 |
| `<Plot label x={[lo,hi]} y={[lo,hi]} series={[{ fn:(t)=>…, label, tone?, dashed? } \| { points:[[x,y],…] }]} markers? xLabel? yLabel? logX? logY? width? height?>` | 函数图与数据曲线（独立图，带编号） |
| `<Lanes label lanes={[…]} items={[{ lane, start, end, label?, tone? }]} unit? tick? span?>` | 时间线：流水线并行、CUDA stream、调度 |
| `<Bars label items={[{ label, value? , segments?:[{ value, label, tone? }], note? }]} unit? max?>` | 横向（堆叠）条形图：显存构成、吞吐比较 |

**色调语义**（`tone`）：

- `neutral`（默认）普通节点；`accent` 主数据通路或本图焦点；`info` 次要通路、存储、缓存；`accent2` 梯度/反向、需要警惕的部分、对比的另一方；`muted` 不活跃、被掩码、被跳过；`strong` 最终结果或唯一重点（每幅图最多一两个）；`ghost` 虚线框，表示可选或假想的部分。
- 字号用 `size="xs|sm|md|lg"`，默认 `md`（14）。框内中文每个字约 14 单位宽，框宽要留够。
- 构图：从左到右或从上到下的单一阅读方向；对齐到 10 的倍数网格；节点间距一致；箭头不要穿过节点。画完后用截图检查（见第 8 节）。

## 6. 链接

- 站内链接从站点根写：`[链式法则与计算图](/math/chain-rule/)`。用章节标题作链接文字，**不要写“第 N 章”**（章号会随结构变化）。
- 只链接到章节页面本身；跨部分不要链接到 `#锚点`。同一章内可以链接锚点。
- `pnpm build` 会检查全部站内链接，断链会让构建失败。

## 7. 交互

本书不使用 React 交互实验：机制用图元图配合手算例与可运行代码讲清楚。确有必要的交互需求，先向维护者提出共享层方案。

## 8. 代码、测试与验证

- 实现：`src/llms_from_scratch/<part>/<module>.py`，模块名对应章节主题（如 `math/autodiff.py`、`transformer/bpe.py`、`gpu/flash_attention_triton.py`）。代码要短小、可读、有中文 docstring 与关键注释，用类型标注；不要为了“通用”引入多余抽象。
- 测试：`tests/<part>/test_<module>.py`，pytest 风格，CPU 上每个文件数秒内完成。核对数学结论（数值梯度、与 PyTorch 参考实现一致、缓存等价、并行切分与单卡一致等）。
- GPU 代码（CUDA/Triton）：本机没有 GPU。kernel 源码写入 `src/llms_from_scratch/gpu/`（`.cu` 或 Triton `.py`），测试用 `pytest.mark.skipif(not torch.cuda.is_available(), ...)` 跳过；同时提供一个 CPU 上可运行的参考实现或模拟器，测试其算法正确性（分块、在线 softmax、归约顺序等）。正文中的 GPU 性能数字必须注明来源（论文、官方博客或明确的推导），不能虚构实测结果。
- 分布式代码：用 `torch.distributed` 的 gloo 后端在 CPU 上以多进程（`torch.multiprocessing.spawn`，world_size 2–4）验证与单进程结果一致。
- 依赖：只用 `torch`、`numpy` 与标准库（已在 `pyproject.toml` 中）。需要新依赖时在汇报中说明，不要自行添加。

验证命令（在仓库根目录）：

```sh
uv sync --group dev
uv run pytest tests/<part>
uv run ruff check src tests
cd site && pnpm install --frozen-lockfile && pnpm build   # 必须通过：KaTeX 严格模式 + 断链检查
```

截图检查版式（构建后；`$SHOT` 是维护者提供的 Playwright 截图脚本路径）：

```sh
cd site && pnpm astro preview --port <端口> &
node $SHOT http://localhost:<端口>/llms-from-scratch/<part>/<slug>/ out.png 1440       # 浅色
node $SHOT http://localhost:<端口>/llms-from-scratch/<part>/<slug>/ out-dark.png 1440 dark
```

## 9. 样式

站点只用 Tailwind CSS 一套样式系统（token 在 `site/src/styles/global.css` 的 `@theme`）。章节作者不写 CSS、不写 `style` 属性、不引入颜色值；需要新的版式能力时向维护者提出，而不是在章节里绕过。
