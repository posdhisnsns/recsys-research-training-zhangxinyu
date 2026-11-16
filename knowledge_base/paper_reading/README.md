# 文献阅读笔记索引

最后更新：2026-11-13。按当前目录实际文件核对，共 **30 篇独立论文笔记**（不计本文件和 `overview.md`）。本次新增 DuoRec、FCRLLM、SSS、ICSRec、IOCLRec、Residual Dominance、Predictability-Aware；删除 Zotero 中不存在的 BASRec、ReaRec。论文结果、本地复现结果、未完成实验分开记录，未进行的实验保留空白。

笔记顶部统一使用：论文（英文名、来源、DOI）／代码／更新记录。图片只保留与内容对应的裁剪图，公式使用 `$...$` 或 `$$...$$`。

本地结果与论文或基线的对比，一律在绝对差值之外同时给出相对百分比：`(本地 − 对照) / 对照 × 100%`，正值为提升、负值为下降，并写明“相对论文”还是“相对基线（如 SASRec）”。论文主表未核对的项留空，不编造百分比。

论文与本地实验结果的直接对照统一使用五列长表：`数据集名 | 指标名 | 论文宣称效果 | 本地实测效果 | 差异百分比`。同一数据集有多个本地运行时，将运行标记写入数据集名；论文或本地缺少对应值时使用 `—`，不计算差异。

| 论文/模型 | 笔记 | 方向 |
|---|---|---|
| ACLSRec | [ACLSRec - 通过自适应扰动与恢复保持序列对比语义](<ACLSRec - 通过自适应扰动与恢复保持序列对比语义.md>) | 自适应扰动与恢复保持序列对比语义 |
| ADARec | [ADARec - 通过自适应扩散与混合专家缓解意图坍塌](<ADARec - 通过自适应扩散与混合专家缓解意图坍塌.md>) | 自适应扩散与混合专家缓解意图坍塌 |
| AdaTTA | [AdaTTA - 通过强化学习自适应选择测试时增强](<AdaTTA - 通过强化学习自适应选择测试时增强.md>) | 强化学习按序列选择测试时增强 |
| AsarRec | [AsarRec - 通过自适应序列增强提升自监督推荐鲁棒性](<AsarRec - 通过自适应序列增强提升自监督推荐鲁棒性.md>) | 自适应序列增强提升自监督推荐鲁棒性 |
| DrunkAgent | [DrunkAgent - 通过隐蔽记忆污染攻击推荐智能体](<DrunkAgent - 通过隐蔽记忆污染攻击推荐智能体.md>) | 推荐智能体安全 |
| DuAF-MAT | [DuAF-MAT - 通过时间感知融合与缺失补全建模多模态兴趣](<DuAF-MAT - 通过时间感知融合与缺失补全建模多模态兴趣.md>) | 时间感知融合与缺失模态补全 |
| DuoRec | [DuoRec - 通过模型级对比学习缓解表示退化](<DuoRec - 通过模型级对比学习缓解表示退化.md>) | 模型级 Dropout 与同目标正样本 |
| FAERec | [FAERec - 通过语义融合与双层对齐改善尾部物品推荐](<FAERec - 通过语义融合与双层对齐改善尾部物品推荐.md>) | LLM 语义融合与尾部物品推荐 |
| FCRLLM | [FCRLLM - 通过翻转课堂式对齐改善长尾序列推荐](<FCRLLM - 通过翻转课堂式对齐改善长尾序列推荐.md>) | 协同/语义双流与长尾对齐 |
| FLAME | [FLAME - 将集成多样性压缩到单网络实现高效序列推荐](<FLAME - 将集成多样性压缩到单网络实现高效序列推荐.md>) | 集成多样性压缩与单网络推理 |
| HERec | [HERec - 利用双曲表征平衡推荐的探索与利用](<HERec - 利用双曲表征平衡推荐的探索与利用.md>) | 双曲层级表示与探索利用 |
| ICSRec | [ICSRec - 通过跨子序列意图对比学习建模用户意图](<ICSRec - 通过跨子序列意图对比学习建模用户意图.md>) | 跨子序列粗细粒度意图对比 |
| IERec | [IERec - 通过兴趣熵与互检索缓解对比语义偏差](<IERec - 通过兴趣熵与互检索缓解对比语义偏差.md>) | 兴趣熵路由与互检索增强 |
| InfoDCL | [InfoDCL - 将辅助语义融入扩散噪声构造对比视图](<InfoDCL - 将辅助语义融入扩散噪声构造对比视图.md>) | 辅助语义引导扩散对比视图 |
| IOCLRec | [IOCLRec - 面向多意图的对比学习序列推荐](<IOCLRec - 面向多意图的对比学习序列推荐.md>) | 单意图/多意图对比学习 |
| LLM-EDT | [LLM-EDT - 通过大模型增强与双阶段训练改善跨域推荐](<LLM-EDT - 通过大模型增强与双阶段训练改善跨域推荐.md>) | 大模型增强与跨域训练 |
| M³BSR | [M³BSR - 通过条件扩散去噪建模多模态多行为偏好](<M³BSR - 通过条件扩散去噪建模多模态多行为偏好.md>) | 条件扩散建模多模态多行为偏好 |
| MCLRec | [MCLRec - 通过元优化增强对比学习](<MCLRec - 通过元优化增强对比学习.md>) | 元优化数据视图与模型视图 |
| MELT | [MELT - 通过长尾用户与物品互增强改善序列推荐](<MELT - 通过长尾用户与物品互增强改善序列推荐.md>) | 长尾用户与物品互增强 |
| MIMAR-SRec | [MIMAR-SRec - 结合多粒度意图与对抗训练增强序列推荐](<MIMAR-SRec - 结合多粒度意图与对抗训练增强序列推荐.md>) | 多粒度意图与对抗鲁棒学习 |
| MuPP | [MuPP - 通过多偏好填充增强短序列](<MuPP - 通过多偏好填充增强短序列.md>) | 多偏好填充利用 padding 位 |
| Predictability-Aware | [Predictability-Aware - 通过可预测性感知子序列建模序列推荐](<Predictability-Aware - 通过可预测性感知子序列建模序列推荐.md>) | 可预测性驱动子序列建模 |
| RaSR | [RaSR - 利用多关系图增强与再生成改进训练数据](<RaSR - 利用多关系图增强与再生成改进训练数据.md>) | 多关系图增强与关系感知再生成 |
| Residual Dominance | [Residual Dominance - 因果自注意力中的残差主导性](<Residual Dominance - 因果自注意力中的残差主导性.md>) | 残差主导性与末项依赖诊断 |
| SID4SRec | [SID4SRec - 结合扩散增强与双层负例过滤改善序列推荐](<SID4SRec - 结合扩散增强与双层负例过滤改善序列推荐.md>) | 扩散增强与双层负例过滤 |
| SSS | [SSS - 关注序列切分对序列推荐的影响](<SSS - 关注序列切分对序列推荐的影响.md>) | 子序列切分对评测的影响 |
| TADA | [TADA - 通过尾部感知序列增强改善长尾推荐](<TADA - 通过尾部感知序列增强改善长尾推荐.md>) | 尾部感知增强与长尾推荐 |
| TCLA | [TCLA - 三元组对比学习与可学习序列增强](<TCLA - 三元组对比学习与可学习序列增强.md>) | 三元组对比学习与可学习增强 |
| TDRec | [TDRec - 在扩散过程中显式建模序列时间演化](<TDRec - 在扩散过程中显式建模序列时间演化.md>) | 扩散过程中的时间演化建模 |
| Transformer | [Transformer - 基于自注意力建模序列并对比四种编码解码架构](<Transformer - 基于自注意力建模序列并对比四种编码解码架构.md>) | 自注意力序列建模基础 |

## 按主题快速查找

| 主题 | 相关笔记 |
|---|---|
| 扩散推荐与去噪 | ADARec、InfoDCL、M³BSR、TDRec、SID4SRec |
| 自适应增强与对比学习 | ACLSRec、AsarRec、DuoRec、ICSRec、IERec、IOCLRec、MCLRec、MuPP、SSS、TADA、TCLA |
| 长尾与语义对齐 | FCRLLM、MELT、TADA、FAERec、LLM-EDT |
| 意图与子序列建模 | ICSRec、IERec、IOCLRec、MCLRec、Predictability-Aware、SSS |
| 测试时计算与结构诊断 | AdaTTA、Residual Dominance |
| 数据、关系与鲁棒性 | RaSR、FLAME、HERec、MIMAR-SRec、DrunkAgent |
| 基础架构与多模态 | Transformer、DuAF-MAT、SID4SRec |

[总览](overview.md)
