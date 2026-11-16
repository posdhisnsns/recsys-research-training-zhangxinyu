# 2026 年 CCF-A 会议 · 序列推荐 × 数据增强 论文清单

> 检索日期：2026-10-22
> 口径：2026 年召开/出版的 **CCF-A 类会议主会**；不含 workshop、findings、期刊、CCF-B/C 会议。
> 每条均已用**官方源**核实（AAAI OJS 文章页 / ACM Crossref 元数据 + doi.org）。arXiv、Semantic Scholar、Google Scholar 仅用于发现，不作为官方链接。

---

## 一、核心命中（10 篇）

| # | 标题 | 会议 | 方法要点 | 官方链接 |
|---|---|---|---|---|
| 1 | De-collapsing User Intent: Adaptive Diffusion Augmentation with Mixture-of-Experts for Sequential Recommendation (ADARec) | AAAI-26 | 不把扩散模型当黑盒生成器，而是用整条去噪轨迹重建稀疏序列的**意图层级**；每个序列自适应决定增强深度，MoE 解耦粗/细粒度意图 | https://doi.org/10.1609/aaai.v40i17.38481 |
| 2 | BLADE: A Behavior-Level Data Augmentation Framework with Dual Fusion Modeling for Multi-Behavior Sequential Recommendation | AAAI-26 | 增强作用在**行为序列**而非 item 序列，设计三种行为级算子保语义一致；配合 item–behavior 双融合 | https://doi.org/10.1609/aaai.v40i18.38543 |
| 3 | Data-Centric Sequential Recommendation with Relation-Augmented Generation (RaSR) | AAAI-26 | 用 LLM 语义表示 + 多关系图**生成增强数据集并再生数据**，不改模型结构（data-centric 路线） | https://doi.org/10.1609/aaai.v40i18.38540 |
| 4 | Tail-Aware Data Augmentation for Long-Tail Sequential Recommendation (TADA) | WWW-26 | 长尾感知：T-Substitute 换头尾、T-Insert 按共现扩展序列；增强序列与原序列在**表征层 mix**，兼顾头部性能 | https://doi.org/10.1145/3774904.3792171 |
| 5 | Adaptive Contrastive Learning in Sequential Recommendation based on Perturbation and Restoration Networks (ACLSRec) | WWW-26 | 用**可学习的扰动 + 复原网络**替代随机噪声/neuron dropout，保证增强视图与原序列语义一致 | https://doi.org/10.1145/3774904.3792207 |
| 6 | Interest Entropy: Rethinking Contrastive Learning for Sequential Recommendation with Interest Uncertainty (IERec) | KDD-26 | 用**兴趣熵**度量序列在扰动下的稳定性，只对低/高熵序列做增强，检索语义互补序列拼接成正视图 | https://doi.org/10.1145/3770854.3780333 |
| 7 | AsarRec: Adaptive Sequential Augmentation for Robust Self-supervised Sequential Recommendation | SIGIR-26 | 面向鲁棒自监督的**自适应序列增强**（增强算子按样本自适应选择/组合） | https://doi.org/10.1145/3805712.3809629 |
| 8 | MuPP: Multi-Preference Padding for Sequential Recommendation | SIGIR-26 | 用多偏好层从每个 item 生成多样子偏好来**填充 padding 位**，替代"重复同一 item"的 padding 式增强 | https://doi.org/10.1145/3805712.3809950 |
| 9 | Pay Attention to Sequence Split: Uncovering the Impacts of Sub-Sequence Splitting on Sequential Recommendation Models | SIGIR-26 | 系统实证**子序列切分**（一种数据层增强）对评测的干扰：很多 SOTA 模型在数据读取阶段默默用了它 | https://doi.org/10.1145/3805712.3808548 |
| 10 | Semantic-Aware Data Augmentation for Sequential Recommendation (SADA) | AAAI-26（Undergraduate Consortium，3 页短文） | LLM 生成语义嵌入 + 语义引导 mixup，缓解长尾表示缺失、避免随机混合破坏语义结构 | https://doi.org/10.1609/aaai.v40i48.42328 |

页码参考：ADARec 40(17):14630–14638；BLADE 40(18):15189–15197；RaSR 40(18):15162–15170；SADA 40(48):41525–41527；TADA pp.5886–5897；IERec pp.1566–1577 (KDD Vol.1)；AsarRec pp.2297–2307；MuPP pp.3854–3859。

## 二、边界条目（增强是组件或邻近任务，未计入上表）

| 标题 | 会议 | 说明 | 链接 |
|---|---|---|---|
| Adaptive Diffusion-based Augmentation for Recommendation (ADAR) | AAAI-26 | 扩散合成**难负样本**（负采样增强），对 CF 与序列推荐均适用；不是"序列增强" | https://doi.org/10.1609/aaai.v40i18.38535 |
| Capturing Dynamic User Interests Under Modality Imbalance for Multimodal Sequential Recommendation (DuAF-MAT) | AAAI-26 | 多模态序列推荐，含"虚拟序列增强"作为辅助组件 | https://doi.org/10.1609/aaai.v40i18.38544 |
| HiST: Hierarchical Semantic Tree Augmentation for Generative Recommendation | KDD-26 (Vol.2) | 生成式推荐的语义树增强，非序列增强主线 | https://doi.org/10.1145/3770855.3817719 |

## 三、空集记录（省得重复检索）

| 会议 | 结论 |
|---|---|
| IJCAI-26 | 无。序列推荐相关仅 ConFM / PDFlow / LLM 层级兴趣演化等，均非增强 |
| ACM MM 2026 | 客观不存在：会期 2026-11-10~14，会议录尚未出版 |
| ICML 2026 | 无。全库提 "sequential recommendation" 仅 3 篇（CRAMER / Conformal Subgraphs / SIRIUS），均非增强 |
| ICLR 2026 | 无。提该短语 12 篇中仅 1 篇接收（CollectiveKV，纯 KV cache 效率），其余撤稿/拒稿 |
| NeurIPS 2026 | 无法核实：proceedings 页面 404，OpenReview 该组 0 条 |
| ACL 2026 | 无。SR 相关仅 UCGRec（图学习），增强类论文均非推荐 |

## 四、期刊补充（CCF-A 期刊，非"会"，但同方向必读）

- **Data Augmentation for Sequential Recommendation: A Survey** — TKDE 38(8):4938–4957, 2026. https://doi.org/10.1109/TKDE.2026.3692969
  （该方向的分类学基准；作者组与 TADA / 序列切分 / TPAMI 下文同源）
- **Exploring and Tailoring the Test-Time Augmentation for Sequential Recommendation** — TPAMI 48(6):6999–7016, 2026. https://doi.org/10.1109/TPAMI.2026.3663633
- **PALRec: LLM-Based Sequential Recommendation with Parameter-Preserving Augmentation** — TOIS, 2026. https://doi.org/10.1145/3842668

## 五、脉络观察

1. **从"固定算子"到"自适应/选择性增强"**：AsarRec、IERec、ACLSRec 都在回答"这条序列该不该增强、增强多少"，而不是统一套 crop/mask/reorder。ADARec 的"自适应增强深度"同理。
2. **增强的载体从 item 序列上移**：BLADE 在行为序列上做，TADA 在表征层 mix，RaSR 直接生成数据集——纯 item 级启发式算子的空间已接近饱和。
3. **语义/生成式增强成为主线**：扩散（ADARec、ADAR）与 LLM（RaSR、SADA）两条生成路线，核心矛盾都是"生成的数据如何不偏离用户真实意图"。
4. **长尾与稀疏是主要动机**：TADA（长尾）、MuPP（短序列）、ADARec（稀疏）都以此立论。
5. **该方向在 ML 三大会近乎缺席**，主战场是 SIGIR / KDD / WWW / AAAI 四个会。

## 六、建议阅读顺序

1. TKDE 综述（建立分类框架）
2. ADARec（已在复现，对照它的 related work 定位）
3. TADA → BLADE（算子设计的两条不同抽象层）
4. AsarRec → IERec（自适应/选择性增强这一新趋势）
5. SSS（评测陷阱，做实验前必读）
