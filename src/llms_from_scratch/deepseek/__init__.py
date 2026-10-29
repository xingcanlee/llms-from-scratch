"""第十部分：DeepSeek 技术全景的参考实现。

各模块对应一章，均以 DeepSeek 官方推理代码（DeepSeek-V3、V3.2-Exp、V4-Flash、V4.1-Flash 的
``inference/model.py``）为事实依据，保留数据流与数学，去掉张量并行、FP8/FP4 kernel 等工程外壳：
``lineage``（开源模型谱系与 KV 账本）、``v3``（MLA 官方布局与分组路由）、``dsa``（V3.2 闪电索引器）、
``v4_attention``（V4 压缩器与稀疏注意力）、``csa2``（V4.1 的 CSA2 与因果编码器—解码器）、
``kv_cache``（FP4 KV 与 SWA 有界重放）、``engram`` 与 ``mhc``（条件记忆与 Single-Pass mHC）、
``dspark``（置信度调度的投机解码）、``optim``（Sinkhorn 均衡更新与按头 Muon）。
"""
