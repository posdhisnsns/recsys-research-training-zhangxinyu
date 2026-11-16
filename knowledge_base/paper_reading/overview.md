# 推荐系统笔记总览

本页按当前 `knowledge_base/paper_reading` 目录维护，现有 **30 篇独立笔记**。`README.md` 是逐篇索引；本页保留主题导航、方法主线和复现状态。论文数值均来自论文原表，除非明确标注“本地”，不应当当作本地复现结果。

## 主题导航

| 主题 | 笔记 |
|---|---|
| 扩散推荐与去噪 | ADARec、InfoDCL、M³BSR、TDRec、SID4SRec |
| 自适应增强与对比学习 | ACLSRec、AsarRec、DuoRec、ICSRec、IERec、IOCLRec、MCLRec、MuPP、SSS、TADA、TCLA |
| 长尾与语义对齐 | FCRLLM、MELT、TADA、FAERec、LLM-EDT |
| 意图与子序列建模 | ICSRec、IERec、IOCLRec、MCLRec、Predictability-Aware、SSS |
| 测试时计算与结构诊断 | AdaTTA、Residual Dominance |
| 数据、关系与鲁棒性 | RaSR、FLAME、HERec、MIMAR-SRec、DrunkAgent |
| 基础架构与多模态 | Transformer、DuAF-MAT |

## 方法主线对照

| 方法 | 改动发生在 | 训练时新增模块 | 测试时是否保留 |
|---|---|---|---|
| DuoRec | 表示空间 | 双 Dropout、同目标 hard positive、对比正则 | 否 |
| SSS | 数据读取 | 子序列切分、目标策略、损失组合 | 间接改变评测协议 |
| ICSRec | 子序列意图 | CICL、FICL、意图 prototype | 否 |
| IOCLRec | 子序列意图 | fine/single/multi-intent ICL | 否 |
| FCRLLM | 长尾表示对齐 | LLM 语义流、FlipClass、ST/CT 对齐 | 否 |
| MELT | 用户/物品长尾表示 | 用户分支、物品分支、课程学习 | 否 |
| Predictability-Aware | 子序列与损失 | 可预测性标记、边界发现、加权 CE | 否 |
| Residual Dominance | 结构诊断 | 范数分解、推理残差缩放 | 是，作为诊断干预 |
| AdaTTA | 测试输入 | PPO 策略、TTA 操作 | 是，增加推理次数 |

## 近期复现状态

| 笔记 | 本地结果 | 当前限制 |
|---|---|---|
| DuoRec | Beauty（稳定运行）test NDCG@10=0.0450；Clothing=0.0138；Sports=0.0251；ML-1M（Dropout=0.1）=0.1733；Yelp=0.0384；相对论文 +1.58% / −6.76% / −4.20% / +3.15% / −0.52% | Beauty 首次运行出现 0.6231 异常高分；其余为单 seed，消融和多 seed 未做 |
| MuPP | Beauty test NDCG@10/MRR@10=0.0495/0.0409；Video Games=0.0651/0.0493 | 论文 PDF/主表仍未核对，暂无法计算相对百分比；消融、多 seed 和 RepPad 留空 |
| MELT | Beauty/Music/Sports 的 MELT+SASRec 相对 SASRec NDCG@10 分别 +10.05%/+8.53%/+7.01%（+0.0300/+0.0319/+0.0224） | 使用 100 个随机负例；Sports 有两次输出，需多次运行确认 |
| AsarRec | 已标注复现失败并放弃 | 增强器 NaN、空视图和预热阶段问题，不能填入论文式成绩 |
| SID4SRec | 完成一次 Beauty 本地运行和代码核查 | 本地负例过滤与论文语义不一致，不能直接对齐论文表 |

本地和论文结果必须同时记录数据切分、候选集、负例策略、骨干、随机种子和最佳轮。论文均值与单次本地运行不能混写；未完成实验保持空白。

## 阅读与复现实验顺序

1. 先读方法数据流和公式，确认增强改变的是输入、表示、负例还是测试时计算。
2. 核对论文主表的数据集、骨干、切分和候选范围；SSS 还要明确 split、target 和 loss。
3. 本地先做与论文最接近的主实验，论文结果和本地结果分栏记录，并同时给出绝对差值和相对百分比（`(本地 − 对照) / 对照 × 100%`）。
4. 主实验稳定后再做消融、参数和多随机种子实验；尚未运行的格子保持空白。
