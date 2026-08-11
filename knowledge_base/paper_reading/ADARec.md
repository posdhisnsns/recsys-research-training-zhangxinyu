# ADARec 文献阅读及代码复现

> 文献：ADARec（AAAI 2026，自适应扩散增强 + 混合专家 MoE 的序列推荐框架）
> 源码：`F:\recsys-research-training-zhangxinyu\experiment\ADARec\src\`

---
## 文献阅读
### 0 符号约定

| 符号    | 含义                    | CLI 默认值                                   |
| ----- | --------------------- | ----------------------------------------- |
| B     | batch_size            | 256                                       |
| T     | max_seq_length        | 50（若 shorten_seq_to 则 = shorten+3）        |
| D     | hidden_size           | 128                                       |
| H     | num_attention_heads   | 2                                         |
| L     | head_size = D/H       | 64                                        |
| K     | num_intent_clusters   | 512(Toys/Sports) / 256(Yelp/Beauty/ml-1m) |
| M     | controller num_levels | 2（硬编码）                                    |
| T_max | diffusion_t_max       | 50                                        |

---

### A. 研究问题

#### A.1 研究对象、任务、输入和预测目标是什么？

- **研究对象**：**序列推荐**（Sequential Recommendation, SR）领域，聚焦于“**数据稀疏**（data sparsity）场景下**用户意图层次建模**”这一子问题。
- **任务**：基于用户的历史行为序列，预测用户的下一个交互动作（predict users' next action）。
- **输入**：用户的历史交互记录（historical behavior / historical interactions），在 ADARec 中具体表现为**单条稀疏序列（single sparse sequence）**——即用户仅有少数历史交互。
- **预测目标**：用户下一个可能交互的物品（next item），即给定历史序列预测未来动作。

【原文依据：P1 摘要 “Sequential recommendation (SR) aims to predict users' next action based on their historical behavior”；P1-2 引言 “用户活动遵循长尾分布，导致数据稀疏问题”；P1 摘要 “reconstruct a user's intent hierarchy from a single sparse sequence”】

#### A.2 为什么这个问题值得研究？

- SR 模型的性能依赖丰富的交互数据（“The performance of SR models relies on rich interaction data”）。
- 真实场景中大量用户仅有少量历史交互 → 数据稀疏（“many users only have a few historical interactions, leading to the problem of data sparsity”）。
- 数据稀疏带来两个直接后果：
  1. 模型在稀疏序列上**过拟合**（“model overfitting on sparse sequences”）；
  2. 阻碍模型捕获用户意图的**底层层次结构**（“hinders the model's ability to capture the underlying hierarchy of user intents”）。
- 最终结果：误解用户真实意图、推荐无关物品（“misinterpreting the user's true intents and recommending irrelevant items”）。
- 用户活动服从**长尾分布**（long-tail distribution），进一步加剧稀疏，严重削弱模型学习过程。

【原文依据：P1 摘要；P1-2 引言 “用户活动遵循长尾分布，导致数据稀疏问题，严重削弱模型学习过程”】

#### A.3 已有方法采用什么路线，具体不足是什么？

**已有路线（现有数据增强方法）**：

- 核心策略：通过生成**相关且多样**（relevant and varied）的数据来缓解过拟合。
- 两种增强方向：生成**相似物品**（保持相关性 relevance）与生成**多样物品**（增加多样性 diversity）。
- 实例：ELCRec 会过拟合稀疏物品——将“能量棒”简单归类为零食、将“登山靴”归为鞋类，错误地推断用户兴趣局限于这些宽泛类别。

**具体不足**：

- 忽视了**重建用户意图层次**（reconstructing the user's intent hierarchy）这一关键问题，而该层次在稀疏数据中已经**丢失**（“lost in sparse data”）。
- 增强的多样性**缺乏用户高层意图的引导**（“lacks the guidance of user's high-level intent”），导致增强数据往往**偏离用户真实意图**（“often fails to align with the user's true intents”）。
- 相关工作综述指出：现有数据增强的核心是“通过生成更多物品来丰富训练集”，而非建模意图结构，最终未能解决“数据稀疏导致的意图层次表征不完整”这一根本问题。

【原文依据：P1 摘要 “Existing data augmentation methods attempt to mitigate overfitting by generating relevant and varied data. However, they overlook the problem of reconstructing the user's intent hierarchy...”；P1-2 引言 ELCRec 例子与 “多样性缺乏用户高层意图的引导”；P3 相关工作 “现有数据增强核心策略：通过生成更多物品来丰富训练集，而非建模意图结构，最终未能解决数据稀疏导致的意图层次表征不完整这一根本问题”】

#### A.4 作者提出什么核心假设？

- **核心论点（核心假设）**：数据增强的关键**不在于盲目权衡相似性与多样性**（“blindly trade-off between relevance and diversity”），**而在于构建用户意图的层次结构**（“constructing the hierarchy of user intents”）。
- **方法学假设**：扩散模型的**逐步去噪轨迹**（step-wise denoising trajectory）会迫使模型在更高抽象层次上推断用户意图；轨迹中每一步都产生**从细粒度到粗粒度**（fine- to coarse-grained）的丰富意图层次。

【原文依据：P1-2 引言 “核心论点：数据增强的关键不在于盲目权衡相似性与多样性，而在于构建用户意图的层次结构”；P1-2 “扩散模型的去噪过程迫使模型在更高抽象层次上推断用户意图，每一步去噪轨迹产生从细粒度到粗粒度的丰富意图层次”；P1 摘要 “we use its entire step-wise denoising trajectory to reconstruct a user's intent hierarchy from a single sparse sequence”】

#### A.5 问题成立需要哪些场景、数据或前提？

- **场景前提**：真实世界场景（real-world scenarios），用户交互呈**长尾分布**，大量用户**仅有少数历史交互**（“only have a few historical interactions”）。
- **数据前提**：输入为**单一稀疏序列**（single sparse sequence）——用户历史交互记录稀少，无法从中直接学到完整的意图层次，需被**重建**（“reconstruct”）。
- **前提假设**：意图层次在稀疏数据中会**丢失**（“lost in sparse data”），且稀疏数据下聚类/层次表征效果差（Fig.2 的 t-SNE 可视化）。
- **评价前提**：方法须同时在**标准基准**（standard benchmarks）与**稀疏序列**（sparse sequences）上优于已有方法，方能证明问题成立与方案有效。

【原文依据：P1 摘要；P1-2 引言；P2 Fig.1、Fig.2；P1 摘要 “Experiments show ADARec outperforms state-of-the-art methods on standard benchmarks and on sparse sequences”】

---

### B. 解决方法

#### B.1 输入与输出：模型接收什么，最终产生什么？

**输入**：

- 用户历史交互序列 → 经骨干编码器（ELCRec）得到**序列嵌入** $E^{0}_{u}\in R^{L\times D}$（数据形状为序列长度 L × 隐维度 D）。
- 序列长度信息：归一化的**序列长度嵌入向量**  $V_{l_u}$（length embedding）。
- 预定义的**最大扩散深度** $T_{max}$（控制器可选深度的上限）。

**输出**：

- 经 HP-MoE 融合得到的**最终用户表征** $h_u$（对用户意图进行重构后得到的序列）
- 将 $h_u$ 送入预测层，得到**所有候选物品的概率分数** $y_u$（用于排序并推荐下一个物品）。

【原文依据：P3-4 “使用ELCRec作为骨干编码器获得序列嵌入 $E_u$”；P3-4 ADC 输入 “$E^0_u$（用户历史序列嵌入，$R^{L×D}$)和 $v_{l_u}$（归一化序列长度嵌入向量）”；P5 “最终融合的用户表征 $h_u$ 送入预测层计算所有候选物品的概率分数 $y_u$”】

#### B.2 方法数据流：原始数据如何经过模块形成预测？

**(a) 序列嵌入（前置）**：用户历史序列输入 ELCRec 骨干编码器 → 得到序列嵌入 $E^0_u$（$R^{L×D}$）；同时编码序列长度得到归一化长度嵌入 $V_{l_u}$。

**(b) ADC（自适应深度控制器）**：以 $E^0_u$ 与 $v_{l_u}$ 为输入，拼接后送入 GRU 捕获时序依赖，池化后经线性层得到跨深度（0…T_max）的 $logits$，再用 Gumbel-Softmax 估计器在 $T_{max}+1$ 个离散选择中选出该序列的离散深度 $T_u$。辅助探索损失 $L_{expl}$ 防止 ADC 坍缩到只选浅深度。ADC的作用是为每条序列**自适应确定增强深度**。

**(c) HDA（分层扩散增强）**：给定 $E^0_u$ 与 $T_u$，前向扩散逐步加高斯噪声生成噪声表征 ${E^0_u,…,E^{T_u}_u}（Eq.4）$；再由可学习去噪网络 $D_θ$ 反向去噪，重建各步去噪表征 $\hat{E}^k_u（Eq.5)$; 迭代得到意图层次集合 $A_u={\hat{E}^{T_u}_u,…,\hat{E}^0_u}$ (高 k 元素=粗粒度意图，低 k 元素=细粒度意图)。HDA 输出是整个结构化集合 $A_u$，而非单一向量。

**(d) HP-MoE（分层解析混合专家）**：细粒度专家 $E_{fin}$（Transformer，受 $L_{rec}$ 驱动）捕获细粒度意图；粗粒度专家 $E_{coar}$（受排斥性对比损失 $L_{cont}$ 驱动）学习用户不变本质。两级门控：第一级对每个层次 $k$ 用内容感知门控 $g_k$ 动态加权$（Eq.7）$；第二级按 $g_k$ 聚合全层次专家输出，得聚合细粒度表征 $z_{fin}$ 与聚合粗粒度表征 $z_{coar}（Eq.8、9）$；最后由 $z_{coar}$ 算出最终门控 $gate_u（Eq.10）$，加权融合得最终用户表征 $h_u（Eq.11）$。

**(e) Prediction（预测层）**：将 $h_u$ 送入预测层，计算所有候选物品概率分数 $y_u$，按分数排序产出“下一个物品”推荐。

【原文依据：P3-4 ADC；P4 HDA；P4 HP-MoE；P5 Prediction；P1-2 “三个关键创新：ADC、HDA、HP-MoE”】

#### B.3 新增与继承：哪些是作者提出的，哪些来自已有工作？

**继承（来自已有工作）**：

- **骨干编码器 ELCRec**：作为现成骨干（backbone）获得序列嵌入 $E^0_u$。
- **扩散基础 DDPM**：前向加噪采用 DDPM 的线性方差调度（linear variance schedule），去噪网络 $D_θ$ 为已有扩散框架组件。
- **意图学习范式**：层次化意图建模思路继承自现有意图学习方法（intent learning）。

**作者新提出（ADARec 三个关键创新）**：

- **ADC（自适应深度控制器）**：自适应确定每条序列所需的增强深度，含 Gumbel-Softmax 选择 + 探索损失 $L_{expl}$。
- **HDA（分层扩散增强）**：利用**整条逐步去噪轨迹**从单条稀疏序列重建粗/细粒度意图层次（核心区别于“仅生成相似/多样物品”的增强）。
- **HP-MoE（分层解析混合专家）**：解耦粗/细粒度意图的专家 + 内容感知两级门控融合。
- **总体框架论点**：用扩散去噪轨迹重建意图层次，而非盲目权衡相似性与多样性。

【原文依据：P3-4 “使用ELCRec作为骨干编码器”；P4 “采用DDPM的线性方差调度”；P1-2 “三个关键创新：ADC（自适应深度控制器）、HDA（分层扩散增强）、HP-MoE（分层解析混合专家）”；P3 相关工作 “意图学习”】

#### B.4 关键公式：符号、形状（张量维度）、运算、输出和作用

> 格式：符号 / 形状 / 运算 / 输出 / 作用 / 原文依据（Equations）。凡原文未明确给出的形状，均标注“原文未明确”或“推断”，不臆造维度。

**① ADC — Eq.(1) 上下文编码**

- 符号：$E^0_u$（用户输入序列嵌入，$R^{L×D}$）；$V_{l_u}$（归一化序列长度嵌入，**原文未给明确形状**，仅称向量）；GRU 捕获时序依赖；Pool 池化。
- 形状：$h_c$ 为 GRU 池化后的上下文向量（原文未给维度，推断为 $R^D$ 或 controller 隐藏维）。
- 运算：拼接 $[E^0_u; V_{l_u}]$ → GRU → Pool。
- 输出：$h_c$（序列级上下文表征）。
- 作用：为自适应深度选择提供序列级上下文。
- 【原文依据：P3-4 $Eq.(1) h_c = Pool(GRU([E^0_u; v_{l_u}]))$

**② ADC — Eq.(2) 深度 logits**

- 符号：$W_σ$、$b_σ$ 为可学习参数；$h_c$ 为①输出。
- 运算：线性变换 $logits_u = W_σ·h_c + b_σ$ 。
- 输出：$logits_u$（跨 0…T_max 共 $T_{max}+1$ 个可能深度的 $logits$）。
- 作用：为深度选择提供各深度打分。
- 【原文依据：P3-4 Eq.(2)】

**③ ADC — Eq.(3) Gumbel-Softmax 深度选择**

- 符号：$g$ 为来自 Gumbel(0,1) 的 i.i.d. 样本；$τ_{gs}$ 为 softmax 温度。
- 运算：$p_u = Softmax((logits_u + g)/τ_{gs})$；前向取 argmax 得离散深度 $T_u$，反向用软概率 $p_u$（Gumbel-Softmax 估计器）。
- 输出：$p_u$（$T_{max}+1$ 维深度选择概率分布）；$T_u$（选定的离散深度）。
- 作用：在 $T_{max}+1$ 个离散深度中软/硬选择增强深度，实现自适应深度控制且前向可微。
- 【原文依据：P3-4 Eq.(3)】

**④ HDA — Eq.(4) 前向加噪**

- 符号：$\bar{\alpha } _k$ 为 DDPM 累积保留系数（线性方差调度）；$E^0_u$（干净序列嵌入 $R^{L×D}$）；$\epsilon_k～N(0,I)$（与 $E^0_u$ 同形状的标准高斯噪声）。
- 形状：$E^k_u$ 与 $E^0_u$ 同形，$R^{L×D}$。
- 运算：闭式前向加噪 $E^k_u = \sqrt{\bar{\alpha}_k}·E^0_u + \sqrt{(1-\bar{\alpha}_k)}·\epsilon_k$ 。
- 输出：$E^k_u$（第 k 步噪声表征）。
- 作用：构造从干净到高度噪声的扩散视图，为去噪/意图层次提供多噪声水平输入。
- 【原文依据：P4 Eq.(4)】

**⑤ HDA — Eq.(5) 反向去噪**

- 符号：$ε_θ$ 为可学习去噪网络（原文指明实现为 MLP）；$k$ 为噪声水平（同时作条件输入）。
- 形状：$\hat{E}^k_u$ 与 $E^k_u$ 同形，$R^{L×D}$。
- 运算：一步反向去噪 $\hat{E}^k_u = \sqrt{(1-\bar{\alpha}_k)}·\epsilon_θ(E^k_u,k)) / \sqrt{\bar{\alpha}_k}$。
- 输出：$\hat{E}^k_u$（第 k 步去噪表征）。
- 作用：从各噪声步重建意图表征——高 k→粗粒度意图（信息被压缩，$I(\hat{E}_k;E_u)≪H(E_u)）$；低 k→细粒度意图（信息近全保留，$I(\hat{E}_k;E_u)≈H(E_u)）$；整条轨迹构成意图层次 $A_u$。
- 【原文依据：P4 Eq.(5)；P4 高/低噪声水平段落】

**⑥ HP-MoE — Eq.(6) 粗粒度对比损失**

- 符号：$z_{u,k}$（粗粒度表征）；$z_{u,k+1}$ 为正样本；$B_k$ 含 $z_{u,k+1}$ 与同批其他用户的粗粒度表征（负样本）；$τ$ 温度；$k$ 范围 $⌊T_u/2⌋…T_u-1$。
- 运算：基于 InfoNCE 的排斥性对比损失（repulsive contrastive loss）$L_{cont} = Σ_{k=⌊T_u/2⌋}^{T_u-1} -log[exp(sim(z_{u,k},z_{u,k+1})/τ) / Σ_{z_j∈B_k} exp(sim(z_{u,k},z_j)/τ)]$。
- 输出：标量损失。
- 作用：强制粗粒度专家学习“用户不变本质”（user-invariant essence），使不同粗粒度层次一致、并与同批其他用户区分。
- 【原文依据：P4 Eq.(6)】

**⑦ HP-MoE — Eq.(7) 内容感知门控**

- 符号：$Pool(\hat{E}^k_u)$（第 k 层去噪表征的池化，即内容表征）；$emb(k)$（噪声水平嵌入）；MLP 多层感知机。
- 形状：$g_k$ **原文未明确给出**（推断为逐层次标量或向量，取值 0~1）。
- 运算：$g_k = Sigmoid(MLP([Pool(\hat{E}^k_u); emb(k)]))$。
- 输出：$g_k$（第 k 层门控概率）。
- 作用：内容感知路由——动态决定每个层次 k 偏向细粒度专家（$g_k$ 大）还是粗粒度专家（$g_k$ 小）。
- 【原文依据：P4 Eq.(7)】

**⑧ HP-MoE — Eq.(8) 聚合细粒度表征**

- 符号：$g_k$（⑦输出）；$E_{fin}$（细粒度专家，Transformer）；$\hat{E}^k_u$（HDA 输出）；$ε$ 数值稳定项。
- 运算：$z_{fin} = (Σ_{k=0}^{T_u} g_k·E_{fin}(\hat{E}^k_u)) / (Σ_{k=0}^{T_u} g_k + ε)$。
- 输出：$z_{fin}$（聚合细粒度表征）。
- 作用：按门控对各层细粒度专家输出加权聚合，强调内容相关层次。
- 【原文依据：P4 Eq.(8)】

**⑨ HP-MoE — Eq.(9) 聚合粗粒度表征**

- 符号：同⑧，权重用 $(1-g_k)$；$E_{coar}$（粗粒度专家）。
- 运算：$z_{coar} = (Σ_{k=0}^{T_u} (1-g_k)·E_{coar}(\hat{E}^k_u)) / (Σ_{k=0}^{T_u} (1-g_k) + ε)$。
- 输出：$z_{coar}$（聚合粗粒度表征）。
- 作用：按反向门控聚合各层粗粒度专家输出。
- 【原文依据：P4 Eq.(9)】

**⑩ HP-MoE — Eq.(10)(11) 最终融合**

- 符号：$gate_u = Sigmoid(W_g·z_{coar} + b_g)$（Eq.10，由粗粒度聚合表征算最终门控）；$z_{fin}$、$z_{coar}$（⑧⑨输出）；⊙ 逐元素乘。
- 运算：$h_u = gate_u ⊙ z_{fin} + (1-gate_u) ⊙ z_{coar}$（Eq.11）。
- 输出：$h_u$（最终融合用户表征，送预测层）。
- 作用：平衡稳定的长期主题（粗粒度，$z_{coar}$）与瞬时细节（细粒度，$z_{fin}$），得到综合用户表征。
- 【原文依据：P4 Eq.(10)、Eq.(11)】

**⑪ 总损失 — Eq.(12)(13)**

- 符号：$L_{rec}$（推荐损失，Eq.13）；$L_{deno}$（去噪 MSE）；$L_{cont}$（Eq.6 对比）；$L_{expl}$（探索负熵）；$λ$ 加权系数。
- 运算：$L_{tot} = L_{rec} + λ_{deno}·L_{deno} + λ_{cont}·L_{cont} + λ_{expl}·L_{expl}$（Eq.12）；$L_{rec} = -Σ_{i∈I} p(i) log(y_{u,i})$（Eq.13，标准交叉熵）。
- 输出：标量总损失。
- 作用：联合优化推荐精度、意图层次重建质量、粗粒度不变性、深度探索多样性。
- 【原文依据：P5 Eq.(12)、Eq.(13)】

#### B.5 训练目标：模型实际优化什么，是否对应最终任务？

**实际优化的目标**是联合损失（Eq.12）：

$L_{tot} = L_{rec} + λ_{deno}·L_{deno} + λ_{cont}·L_{cont} + λ_{expl}·L_{expl}$

各分项及其与最终任务的关系：

- **$L_{rec}$（Eq.13，标准交叉熵）**：直接对应**最终推荐任务**——预测用户下一个交互物品的概率分布 $y_u$，是任务主监督信号。✓ **直接对应最终任务**。
- **$L_{deno}$（去噪 MSE：$‖ε_θ(E^k_u,k) - ε_k‖²$）**：优化可学习去噪网络 $D_θ$ 准确估计噪声，支撑 HDA 意图层次重建质量（间接服务推荐）。
- **$L_{cont}$（Eq.6 排斥性对比损失）**：强制粗粒度专家学习用户不变本质，提升粗粒度意图判别性。
- **$L_{expl}$（负熵探索损失：$-Σ p_{u,i} log p_{u,i}）$**：防止 ADC 坍缩到只选浅深度，鼓励深度选择多样性。

**是否对应最终任务**：主损失 $L_{rec}$ 直接对应“下一个物品预测”这一最终推荐任务；其余三项为辅助正则/重构损失，服务于“从稀疏序列重建完整意图层次”这一研究问题的解法质量，间接提升最终推荐效果。论文在**标准基准与稀疏序列上均优于 SOTA**（P1 摘要），验证了训练目标与最终任务的一致性。

【原文依据：P4-5 Eq.(12)、Eq.(13)；P3-4 $L_{expl}$（“引入辅助探索损失 $L_{expl}$ 防止 ADC 坍缩”）；P4 $L_{deno}$（“去噪网络 $D_θ$ 使用标准均方误差损失优化”）、$L_{cont}$（Eq.6）；P1 摘要 “outperforms state-of-the-art methods on standard benchmarks and on sparse sequences”】

---

## 代码复现
### 1. 代码体系总览

| 文件                     | 行数  | 核心职责                                                      | 关键类/函数                                                                                                 |
| ---------------------- | --- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `modules.py`           | 313 | 基础算子与网络积木                                                 | `LayerNorm/Embeddings/SelfAttention/Intermediate/Layer/Encoder/NCELoss/PCLoss/AdaptiveDepthController` |
| `models.py`            | 218 | SASRec 主干 + KMeans 意图聚类 + 双分支前向                           | `KMeans/SASRecModel`                                                                                   |
| `datasets.py`          | 207 | 序列截断/补零/增强视图/正负样本；**存在未定义变量 bug**                         | `RecWithContrastiveLearningDataset`                                                                    |
| `data_augmentation.py` | 66  | 对比学习增强算子                                                  | `Random/Crop/Mask/Reorder`                                                                             |
| `diffusion_utils.py`   | 21  | DDPM 噪声调度与前向加噪                                            | `get_noise_schedule/forward_diffusion`                                                                 |
| `trainers.py`          | 932 | 训练/评测调度；单专家与双专家                                           | `Trainer/ELCRecTrainer/DualExpertTrainer`                                                              |
| `utils.py`             | 494 | 种子/负采样/稀疏矩阵/评测指标/早停                                       | `EarlyStopping/neg_sample/get_metric/ndcg_k/...`                                                       |
| `main.py`              | 210 | argparse 全量超参 + 按数据集覆盖 + 数据集/DataLoader/trainer 装配 + 训练循环 | `main()`                                                                                               |

---

### 2. 六步映射体系（核心公式映射表）

> 每张表六列：①论文符号/公式　②张量含义　③代码定位(文件:函数:行)　④变量名　⑤运行时 shape　⑥实现补充 / 人工核验

#### 2.1 ADC（自适应深度控制器）公式 (1)-(3)

**论文意图**：用 GRU 编码序列 + 长度嵌入，输出每个序列在若干扩散深度上的选择概率分布 π_u（论文摘要称 Gumbel-Softmax 选深度）。

| ① 论文符号/公式                          | ② 张量含义               | ③ 代码定位                                                   | ④ 变量名                     | ⑤ 运行时 shape | ⑥ 实现补充 / 人工核验                                                                                  |
| ---------------------------------- | -------------------- | -------------------------------------------------------- | ------------------------- | ----------- | ---------------------------------------------------------------------------------------------- |
| (1) `h_u = GRU(E_u)_last`          | 序列的 GRU 末态表示（压缩序列语义） | `modules.py:AdaptiveDepthController.forward` (~L270-280) | `sequence_representation` | (B, D)      | GRU `batch_first=True`，取 `last_hidden.squeeze(0)`；`sequence_emb` 为 (B,T,D)                     |
| (2) `f_u = [h_u ; ℓ(len_u)]`       | 序列表示拼接长度嵌入           | 同上 forward                                               | `fused_representation`    | (B, 2D)     | `length_embedding` 表大小 513，长度 `clamp(max=512)`；拼接在 dim=-1                                      |
| (3) `π_u = softmax(W_π·f_u)` ∈ Δ^M | 扩散深度选择分布（M=2）        | 同上 forward                                               | `probs`                   | (B, M=2)    | ⚠️ **论文称 Gumbel-Softmax，代码实为 `F.softmax(logits, dim=-1)`（普通 softmax，无 Gumbel 噪声）**。详见 §3 / §8。 |
| `logits = W_π·f_u`                 | 深度 logits            | 同上                                                       | `logits`                  | (B, 2)      | `output_layer = Linear(2D, num_levels=2)`，num_levels 硬编码 2                                     |
| `len_u`（输入）                        | 序列真实长度               | `models.py:SASRecModel.forward` 传 `sequence_lengths`     | `sequence_lengths`        | (B,)        | 由 `datasets.__getitem__` 末尾 `(torch.tensor(sequence_len),)` 提供                                 |

**映射结论**：ADC 把“深度选择”建模为对 `diffusion_levels`（e5=[4,5]，e10=[9,10]）中每个深度的一个可学习 softmax 权重。代码用普通 softmax，`num_levels=2` 与 `diffusion_levels` 长度一致。

#### 2.2 HDA（分层扩散增强）公式 (4)-(5)

**论文意图**：DDPM 式前向加噪，从干净嵌入 E_0 一步采样到噪声嵌入 E_t，构造分层（粗/细粒度）增强视图。

| ① 论文符号/公式                             | ② 张量含义        | ③ 代码定位                                  | ④ 变量名            | ⑤ 运行时 shape      | ⑥ 实现补充 / 人工核验                                                                                 |
| ------------------------------------- | ------------- | --------------------------------------- | ---------------- | ---------------- | --------------------------------------------------------------------------------------------- |
| (4) `ᾱ_t = ∏_{s=1..t} α_s, α_s=1-β_s` | 累积保留系数调度表     | `diffusion_utils.py:get_noise_schedule` | `alphas_cumprod` | (T_max,) = (50,) | `betas=linspace(1e-4, 0.02, 50)`，`alphas=1-betas`，`cumprod(axis=0)`                           |
| (4) `β_s = linspace(β_0,β_T,T)`       | 线性方差调度        | 同上                                      | `betas`          | (50,)            | 默认 `beta_start=0.0001, beta_end=0.02`                                                         |
| (5) `E_t = √ᾱ_t·E_0 + √(1-ᾱ_t)·ε`     | 第 t 步加噪嵌入（闭式） | `diffusion_utils.py:forward_diffusion`  | `E_t`            | (B, T, D)        | `ε = randn_like(E_0)`；`√ᾱ_t` reshape 为 (B,1,1) 广播                                             |
| `t`（步数）                               | 每条序列各自的扩散步    | `models.py:forward` / `trainers`        | `t`              | (B,)             | `t = full(B, depth-1)`；depth∈{4,5}→t∈{3,4}，e10→{8,9}                                          |
| `E_0`（干净嵌入）                           | 物品+位置嵌入       | `models.py:add_position_embedding`      | `sequence_emb`   | (B, T, D)        | 注意：adaptive 分支 forward_diffusion 用 **未 detach** 的 sequence_emb（梯度经此回传）；仅 controller 输入 detach |

**映射结论**：⚠️ **代码只实现 DDPM 前向（加噪）过程，未实现可学习的反向去噪网络**（无 ε_θ）。“反向”由下游 encoder + `L_deno` 隐式承担（见 §2.4）。这与“前向/反向加噪”的措辞不符，属证据边界——论文可能训练了独立去噪器，但本仓库代码未见。

#### 2.3 HP-MoE（分层解析混合专家）公式 (6)-(11)

**论文意图**：用门控把不同扩散深度（粗/细粒度意图层次）的专家输出融合。本代码有 (a) **专家内深度门控**（controller 的 π_u 对多深度加权求和）与 (b) **专家间融合**（e5/e10 两个独立模型在推理时等权平均）。

| ① 论文符号/公式                                     | ② 张量含义                  | ③ 代码定位                                                        | ④ 变量名                      | ⑤ 运行时 shape    | ⑥ 实现补充 / 人工核验                                                        |
| --------------------------------------------- | ----------------------- | ------------------------------------------------------------- | -------------------------- | -------------- | -------------------------------------------------------------------- |
| (6) `z_u = W_g·f_u`                           | 门控 logits（= ADC logits） | `modules.py:AdaptiveDepthController.output_layer`             | `logits`                   | (B, 2)         | 门控与深度选择共用同一 `output_layer`                                           |
| (7) `G_u = softmax(z_u)`                      | 门控权重（= π_u）             | 同上                                                            | `probs`                    | (B, 2)         | 经 `probs_for_diffusion_normalized` 在非零深度上再归一化                        |
| (8) `E_t^{(k)} = forward_diffusion(E_0, t_k)` | 第 k 个深度的加噪视图            | `models.py:forward` 循环                                        | `diffused_emb`             | (B, T, D)      | 仅遍历 `non_zero_depth_indices`                                         |
| (9) `Ē_u = Σ_k G_{u,k}·E_t^{(k)}`             | 专家内深度融合嵌入               | `models.py:forward` (`sum(weighted_embs)`)                    | `final_sequence_emb`       | (B, T, D)      | `weight = probs[:,i].unsqueeze(1).unsqueeze(2)`，逐深度加权后求和             |
| (10a) `H_c = Encoder(Ē_u; θ_c)`               | 粗粒度专家（浅扩散）输出            | `models.py:forward` → `adaptive_item_encoded_layers[-1]` (e5) | `adaptive_sequence_output` | (B, T, D)      | e5 的 `diffusion_levels=[4,5]`，共享 `item_encoder`                      |
| (10b) `H_f = Encoder(Ē_u; θ_f)`               | 细粒度专家（深扩散）输出            | 同上 (e10)                                                      | `adaptive_sequence_output` | (B, T, D)      | e10 的 `diffusion_levels=[9,10]`                                      |
| (11) `ŷ_u = (g(H_c)+g(H_f))/2`                | 专家间融合预测                 | `trainers.py:DualExpertTrainer.iteration`（eval）               | `rating_pred`              | (B, num_items) | ⚠️ **专家间融合是等权平均 `(pred_e5+pred_e10)/2`，无专门门控网络**；`g(·)=predict_full` |

**映射结论**：HP-MoE 在代码中被拆解为两层简单融合——专家内用 controller 的 softmax 门控（π_u）对多扩散深度加权；专家间用推理时等权平均。论文声称的“专家混合门控”在此实现为上述近似，未见跨专家的可学习 gating 参数（证据边界）。

#### 2.4 总损失函数公式 (12)-(13)

**论文意图**：联合优化推荐损失 + 自适应重构损失 + 探索（负熵）损失 + 对比损失 + 簇内/簇间距离损失。

| ① 论文符号/公式                                                            | ② 张量含义        | ③ 代码定位                                            | ④ 变量名                    | ⑤ 运行时 shape | ⑥ 实现补充 / 人工核验                                                                                                         |
| -------------------------------------------------------------------- | ------------- | ------------------------------------------------- | ------------------------ | ----------- | --------------------------------------------------------------------------------------------------------------------- |
| (12) `L_rec = L_BPR(raw) + ρ·L_BPR(adapt)`                           | 双分支推荐损失       | `trainers.py:DualExpertTrainer._train_one_expert` | `total_rec_loss`         | 标量          | `ρ = diffusion_aug_rate`(0.5)；`raw_rec_loss` 用 `raw_sequence_output`，`adaptive_rec_loss` 用 `adaptive_sequence_output` |
| `L_BPR = Σ -log σ(pos)·m - log(1-σ(neg))·m / Σm`                     | BPR 交叉熵       | `trainers.py:Trainer.cross_entropy`               | `loss`                   | 标量          | `istarget=(pos_ids>0).view(-1)`；分母 `Σ istarget`（非 B×T）                                                                |
| (12) `L_deno = MSE(Ĥ⊙m, H⊙m)/‖m‖`                                    | 自适应表示对原始表示的重构 | `_train_one_expert`                               | `reconstruction_loss`    | 标量          | `input_mask=(input_ids>0).unsqueeze(-1)`；`reduction='sum'/mask.sum()`；`target=raw.detach()`                           |
| (12) `L_expl = -E_u[Σ_k G_{u,k}log G_{u,k}]`                         | 负熵（探索）损失      | 同上                                                | `exploration_loss`       | 标量          | `entropy=-Σ p log p`；`exploration_loss=-mean(entropy)`；最小化即最大化熵（鼓励深度多样）                                               |
| (12) `L_cl = w_cf·L_Inst + w_int·L_Intent`                           | 对比损失          | `_instance_...` / `_pcl_...`                      | `cl_losses`              | 标量列表        | `w_cf=cf_weight`(0.1)，`w_int=intent_cf_weight`(0.3)                                                                   |
| `L_Inst = InfoNCE(slice0, slice1)`                                   | 实例级对比         | `modules.py:NCELoss`                              | `nce_loss`               | 标量          | logits (2B,2B)，labels arange(2B)；同意图对置 -inf                                                                           |
| `L_Intent = PCL(samples, intents)`                                   | 意图级对比         | `modules.py:PCLoss`                               | `mean_pcl_loss`          | 标量          | `intents`=(K,B,D)；逐意图与两视角对比                                                                                           |
| (12) `L_dist = α(d_intra + d_inter)`                                 | 簇内/簇间距离       | `DualExpertTrainer` Hybrid 分支                     | `sample_distance_losses` | 标量列表        | `α=trade_off`；`d_intra=mean(‖x-c*‖²)`，`d_inter=-mean(‖c_i-c_j‖²)`                                                     |
| (13) `Joint = w_rec·L_rec + λ_d·L_deno + λ_e·L_expl + L_cl + L_dist` | 联合目标          | `_train_one_expert` 末尾                            | `joint_loss`             | 标量          | `w_rec=rec_weight`(1)；`λ_d=deno_weight`(1e-4)；`λ_e=expl_weight`(1e-4)                                                 |

**映射结论**：双专家路径的 `joint_loss` 同时含 `L_deno`(仅自适应专家) 与 `L_expl`(仅当 `probs` 非空)。权重极小（1e-4），重构/探索项在实践中是弱正则。

---

### 3. 核验问题（Slide 15-19, 24）

#### Slide 15 — Padding 与掩码 / 参数表

**Q1：padding 在哪一步被清零？**

- **嵌入层**：`item_embeddings = nn.Embedding(..., padding_idx=0)` → padding 位（id=0）恒为零向量（梯度恒 0）。
- **注意力**：`extended_attention_mask = (1.0 - mask) * -10000.0` → padding/未来位置分数 -10000，softmax≈0。
- **损失**：`cross_entropy` 中 `istarget=(pos_ids>0)` 屏蔽 padding；`reconstruction_loss` 用 `input_mask=(input_ids>0)` 屏蔽。
- **结论**：padding 在“嵌入(零向量)→注意力(-10000)→loss(istarget 掩蔽)”三层分别处理，并非单一清零点。

**Q2：公式中的 M / P 分别对应哪张参数表？**

- **P（Projection 参数表）**：`SelfAttention` 的 `query/key/value/dense` 四个 `Linear(D,·)`（即 W^Q,W^K,W^V,W^O），以及 `Embeddings` 的 `item_embeddings`/`position_embeddings` 参数表。
- **M（Mask 矩阵）**：`extended_attention_mask`（shape (B,1,T,T)，值 0/-10000），由 `padding_mask (input_ids>0)` 与 `subsequent_mask (torch.triu)` 逐元素乘积得到。**注意 M 不是可学习参数，是运行时构造的掩码张量。**

#### Slide 16 — 缩放维度 / Softmax 维度

**Q3：缩放使用哪个维度？Softmax 在哪个维度执行？**

- **缩放维度**：`attention_scores / math.sqrt(self.attention_head_size)`（即 L=head_size=D/H），**不是** hidden_size 也不是 seq_len。
- **SelfAttention Softmax 维度**：`nn.Softmax(dim=-1)` → 最后一维 = key/seq 维 (B,H,T,T) 中最后一个 T（对每个 query 在所有 key 上归一化）。
- **ADC Softmax 维度**：`F.softmax(logits, dim=-1)` → 沿 `num_levels`（深度维）归一化。

#### Slide 17 — 仅下三角 Mask 能否同时处理 Padding 的 Key 和 Query？

**Q4：只使用下三角 Mask 能否同时处理 Padding 作为 Key 和 Query？**

- **不能。** `torch.triu` 仅保证因果性（屏蔽未来位置）。padding 必须靠额外的 `(input_ids>0)` 逐元素乘积掩码处理。
- **代码正确性**：`extended_attention_mask = (input_ids>0).unsqueeze(1).unsqueeze(2) * subsequent_mask`，再 `(1-mask)*-10000`。二者乘积同时把 (a) padding 作为 **Key**（对应列 -10000）、(b) 未来位置作为 **Query 可见范围** 屏蔽。
- **边界**：padding 作为 Query 的位置，其所有 Key 权重被压到≈0，但该位置输出本身不进入 loss（`istarget=0`），故无影响。

#### Slide 18 — 当前 Block 中 Q/K/V 来源

**Q5：当前 Block 中哪个张量作为 Q，哪个作为 K/V？**

- 在 `SelfAttention.forward`：`mixed_query_layer = self.query(input_tensor)`、`key/value` 同理，**三者均来自同一 `input_tensor`**（自注意力）。
- `input_tensor` 在 raw 分支 = `sequence_emb`（来自 `add_position_embedding`）；在 adaptive 分支 = `final_sequence_emb`（加权求和的加噪嵌入）。
- 即：Q=K=V=当前分支输入；经 `transpose_for_scores` 重排为 (B,H,T,L) 后做 scaled dot-product。

#### Slide 19 — 哪些位置不进入 loss？为何分母不能直接用 B×T？

**Q6：哪些位置不进入 loss，分母为什么不能直接用 B×T？**

- **不进入 loss 的位置**：(1) `input_ids==0` 的 padding 位（`istarget=0`）；(2) `NCELoss` 中对角线（自身）与同意图对（置 -inf）；(3) `reconstruction_loss` 中 padding（被 `input_mask` 屏蔽）。
- **分母**：`cross_entropy` 用 `torch.sum(istarget)`（有效位置数），而非 B×T，因为 padding 位无真实标签，若用 B×T 会把 0 标签的“虚假负样本”计入，且尺度随 padding 比例漂移，破坏梯度量级。
- `reconstruction_loss` 分母同理为 `input_mask.sum()`（非 B×T×D）。

#### Slide 24 — rank+2 的来源

**Q7：rank=0 在公式中为什么用 rank+2？**

- 评测指标 `get_metric`/`ndcg_k` 中 `NDCG += 1.0/log2(rank+2.0)`。
- `rank` 为 **0-indexed** 排名（rank=0 = 第 1 位）。DCG 标准公式对 1-indexed 位置 `pos` 用 `1/log2(pos+1)`。
- 0-indexed → 1-indexed：`pos = rank+1`；再代入公式：`log2((rank+1)+1) = log2(rank+2)`。
- 故 `rank+2` = 0-indexed 排名映射到 DCG 位置增益分母。`idcg_k` 同样用 `log(i+2,2)`（i 从 0 起）保持口径一致。

---

### 4. 代码阅读三问（Slide 10）

#### 4.1 程序按什么顺序执行？

`main()` 顺序：

1. `argparse` 解析全部超参 → `set_seed` → `check_path(output_dir)`。
2. 按 `data_name` **覆盖** `num_intent_clusters / trade_off / prototype`（CLI 默认值被覆盖！见 §7）。
3. `get_user_seqs` 读序列 + 构造 valid/test 稀疏矩阵 → 设 `item_size = max_item+2`、`mask_id`。
4. 构造 4 个 `RecWithContrastiveLearningDataset`（cluster/train/eval/test）+ 对应 `DataLoader`（train 用 `RandomSampler`，其余 `SequentialSampler`）。
5. 若 `dual_expert`：建 `model_e5(SASRecModel, levels=[4,5])`、`model_e10(levels=[9,10])` + 两个 `EarlyStopping` + `DualExpertTrainer`；否则 `SASRecModel` + `ELCRecTrainer`。
6. 训练循环：`for epoch: trainer.train_epoch(epoch)` → `trainer.valid_epoch(epoch)`（各自早停，双停则 break）。
7. 切 `train_matrix = test_rating_matrix` → 载入最优权重 → `iteration(..., expert_to_eval='fused')` 测试评测。

#### 4.2 每个模块/函数承担什么职责？

- **`modules.py`**：纯网络积木。`Embeddings`(item+pos→(B,T,D))；`SelfAttention`(scaled dot-product + 残差)；`Intermediate`(FFN)；`Encoder`(多层堆叠，返回所有层)；`NCELoss/PCLoss`(对比)；`AdaptiveDepthController`(GRU+长度嵌入→深度 softmax)。
- **`models.py`**：`KMeans`(FAISS 聚类，train 提取 centroids、query 返回簇与中心)；`SASRecModel`(主干 + 双分支 forward + 掩码构造 + 权重初始化)。
- **`datasets.py`**：`__getitem__` 按 data_type 切 input/target/answer，构造正负样本与多视图增强对。
- **`diffusion_utils.py`**：噪声调度与单步前向加噪（闭式）。
- **`trainers.py`**：`Trainer`(基类：聚类器/projection/NCE/PCL/BPR/指标)；`ELCRecTrainer`(单专家，三种对比 + 距离损失)；`DualExpertTrainer`(双专家，`_train_one_expert` 内含 L_deno/L_expl，训练/评测分离)。
- **`utils.py`**：种子/负采样/稀疏矩阵/评测指标/早停。

#### 4.3 参数、数据和输出从哪里进入或离开？

- **参数进入**：CLI(`main.py`) → `args` 命名空间 → 各 `nn.Module.__init__(args)`；优化器在 Trainer 内以 `model.parameters()` 创建。
- **数据进入**：磁盘 txt → `get_user_seqs` → `RecWithContrastiveLearningDataset` → `DataLoader` → `batch=(rec_batch, cl_batches, seq_class_label_batches)`。
  - `rec_batch = (user_id, input_ids, target_pos, target_neg, _, sequence_len)`（train/valid）；test 多一个 `sample_negs`。
- **模型输出**：`SASRecModel.forward` → `(raw_sequence_output, adaptive_sequence_output, probs)`，各 (B,T,D) 与 (B,2)。
- **输出离开**：`joint_loss`(标量) → `backward` → `optimizer.step`；评测 `rating_pred`(B,num_items) → argpartition top20 → `get_full_sort_score` → 日志文件 + 返回 `[HIT@5,NDCG@5,HIT@20,NDCG@20]`。

---

### 5. Debug 观察清单（Slide 12）

> 每阶段给出：断点位置 / 观察内容（shape、数值范围、切片）/ 关键变量名。

#### 5.1 数据阶段（datasets.py）

| 断点                                                                       | 观察                                                                                 | 关键变量                                              |
| ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- | ------------------------------------------------- |
| `RecWithContrastiveLearningDataset.__getitem__` 末尾 `return`              | `input_ids` 长度==T；左侧补零（前 `pad_len` 个应为 0）；`target_pos/target_neg` 与 `input_ids` 对齐 | `input_ids, target_pos, target_neg, sequence_len` |
| `_data_sample_rec_task` 测试分支 `test_samples = self.test_neg_items[index]` | ⚠️ `index` 未定义 → 若 `test_neg_items is not None` 触发 `NameError`                     | `index`（**bug，见 §8**）                             |
| `_one_pair_data_augmentation` 返回前                                        | 增强后长度==T；`assert len==max_len`                                                     | `augmented_seqs`                                  |

#### 5.2 模型阶段（models.py / modules.py）

| 断点                                        | 观察                                                                                  | 关键变量                                                                       |
| ----------------------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `SASRecModel.get_attention_mask` return   | mask shape (B,1,T,T)；有效位 0、屏蔽位 -10000；padding 与未来位均为 -10000                         | `extended_attention_mask`                                                  |
| `SASRecModel.forward` 双分支处                | `probs` 每行和≈1（已归一化）；`final_sequence_emb` 数值范围（应含噪声，方差随 t 增大）；`raw` vs `adaptive` 差异 | `probs, final_sequence_emb, raw_sequence_output, adaptive_sequence_output` |
| `AdaptiveDepthController.forward` return  | `probs` ∈ (B,2)，非负、和为 1；`logits` 是否饱和                                               | `logits, probs, length_emb`                                                |
| `SelfAttention.forward` `attention_probs` | (B,H,T,T) 每行和=1；padding 行/列应≈0                                                      | `attention_probs, attention_scores`                                        |

#### 5.3 训练阶段（trainers.py）

| 断点                                         | 观察                                                                                                         | 关键变量                                             |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| `DualExpertTrainer._train_one_expert` 分支判断 | `is_adaptive` 应为 True（专家含 controller）；否则走 fixed 分支报错                                                       | `is_adaptive, model.controller`                  |
| `reconstruction_loss` 计算后                  | 非负、量级（因 mask 归一化）；`target_representation` 已 detach                                                         | `reconstruction_loss, input_mask`                |
| `exploration_loss` 计算后                     | `entropy≥0` → `exploration_loss≤0`；最小化应使其更负（熵更大）                                                           | `entropy, exploration_loss`                      |
| `joint_loss.backward()` 前                  | 各分项（`total_rec_loss, reconstruction_loss, exploration_loss, cl_losses, sample_distance_losses`）量级与 NaN/Inf | `joint_loss` 各加数                                 |
| `ELCRecTrainer.iteration` Hybrid 距离分支      | ⚠️ 存在一行被丢弃的重塑计算（见 §8 dead code）；`sample_distance_loss`/`center_distance_loss` 符号                           | `sample_center_distance, center_center_distance` |

#### 5.4 评价阶段（trainers.py / utils.py）

| 断点                                                                          | 观察                                         | 关键变量                                    |
| --------------------------------------------------------------------------- | ------------------------------------------ | --------------------------------------- |
| `DualExpertTrainer.iteration` fused 分支 `rating_pred = (pred_e5+pred_e10)/2` | `rating_pred`(B,num_items) 有限；训练集交互位是否已置 0 | `rating_pred`                           |
| `rating_pred[train_matrix>0]=0` 之后                                          | 置零后 top20 不应含已交互物品                         | `ind, arr_ind_argsort, batch_pred_list` |
| `get_full_sort_score` 返回                                                    | `[HIT@5,NDCG@5,HIT@20,NDCG@20]` 合理（0~1）    | `recall, ndcg`                          |
| `get_metric` 内 `1/log2(rank+2)`                                             | rank=0 → 分母 log2(2)=1（满增益）；验证 rank+2 口径    | `rank`                                  |

---

### 6. 数据流链路图（Slide 11）

```
[磁盘 .txt]
   │  get_user_seqs()
   ▼
[user_seq, valid/test rating_matrix]  ──main.py──► args.item_size, args.train_matrix
   │
   ▼  RecWithContrastiveLearningDataset(user_seq, data_type)
[Dataset 实例]  ── __getitem__(index) ──►  (_sample 等价物)
   │    · 切 input_ids/target_pos/answer (train 去末3 / valid 去末2 / test 去末1)
   │    · _data_sample_rec_task → 正负样本 + sequence_len
   │    · _one_pair_data_augmentation × nCr(n_views,2) → 增强视图对
   │    · _process_sequence_label_signal → seq_class_label
   ▼  返回 (rec_tensors, cf_tensors_list, seq_class_label)
[DataLoader(collate_fn=默认)]  ──► batch = (rec_batch, cl_batches, seq_class_label_batches)
   │
   ▼  DualExpertTrainer.train_epoch(epoch)
   │    for (rec_batch, cl_batches, seq_class_label_batches):
   │       rec_batch → .to(device); seq_class_label_batches → .to(device)
   │
   ▼  _train_one_expert(model_e5/e10, optim, rec_batch, cl_batches, seq_class_label, is_coarse_expert=True)
   │    · model(input_ids, sequence_lengths=seq_len)  ──► (raw, adaptive, probs)   [model.forward]
   │    · cross_entropy(raw,...) + diffusion_aug_rate*cross_entropy(adaptive,...)  → total_rec_loss
   │    · L_deno = MSE(adaptive⊙m, raw.detach()⊙m)/‖m‖
   │    · L_expl = -mean(entropy(probs))
   │    · InstanceCL/IntentCL + 距离损失 → cl_losses / sample_distance_losses
   │
   ▼  joint_loss = rec_weight*total_rec_loss + deno_weight*L_deno + expl_weight*L_expl + Σcl + Σdist
   │       [_train_one_step 等价：前向→各分项→求和]
   ▼  optimizer.zero_grad(); joint_loss.backward(); optimizer.step()
   │
   ▼  valid_epoch(epoch): 分别对 e5/e10 调 iteration(train=False, expert_to_eval=...) 
   │       → predict_full → 置训练集位0 → argpartition top20 → get_full_sort_score
   │       → early_stopper_e5/e10 (依据 NDCG@20)
   │       → fused = (pred_e5+pred_e10)/2 评测（仅日志）
   │
   ▼  双停 → break；否则下一 epoch
   ▼  测试：train_matrix=test_rating_matrix；载最优权重；iteration(expert_to_eval='fused')
   ▼  get_full_sort_score → [HIT@5,NDCG@5,HIT@20,NDCG@20] → 写日志
```

> 注：本仓库无显式 `__iter__/_sample/collate_fn`，上图以 `Dataset.__getitem__` 充当 `_sample`、PyTorch 默认 `collate_fn` 充当 `collate_fn`、`_train_one_expert` 充当 `_train_one_step`。

---

### 7. 证据边界说明（Slide 22 / 25）

#### 7.1 四套配置 ≠（务必区分）

| 维度         | 内容                    | 本仓库实际                                                                                                                             |
| ---------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| 论文设置       | 官方超参/公式（未持 PDF）       | 未知，以摘要为准                                                                                                                          |
| README 命令  | 仓库运行命令                | 未提供（本环境无 README 运行示例）                                                                                                             |
| CLI 默认值    | `main.py` argparse 默认 | hidden=128, heads=2, layers=2, lr=1e-3, batch=256, cf_weight=0.1, intent_cf_weight=0.3, deno/exp_weight=1e-4, diffusion_t_max=50… |
| **本次实际配置** | `data_name` 触发的覆盖     | Toys/Sports→`num_intent_clusters=512,trade_off=1`；Yelp→256/0.1；Beauty→256/10/`prototype=shift`；其余 concat                          |

⚠️ **关键不一致**：CLI 默认 `num_intent_clusters="256"`，但 `main.py` 在 `data_name∈{Toys,Sports}` 时**覆盖为 512**；`trade_off`/`prototype` 同样被按数据集覆盖。因此“CLI 默认值”不可直接当作“实际配置”。

#### 7.2 三层结论（Slide 25）

1. **实验事实（代码可验证）**：
   - ADC 用普通 softmax（非 Gumbel-Softmax）；num_levels=2 硬编码。
   - 仅实现 DDPM 前向加噪，无独立反向去噪网络。
   - 双专家 = 两个独立 `SASRecModel`，推理融合为等权平均。
   - `L_deno/L_expl` 权重仅 1e-4，属弱正则。
   - `SelfAttention` 缩放用 `head_size`、softmax dim=-1；padding 经嵌入零向量 + -10000 掩码 + istarget 三层处理。
2. **作者解释（论文摘要/注释声称）**：
   - ADC “Gumbel-Softmax 选深度”；HDA “分层粗/细粒度意图”；HP-MoE “专门处理不同层级表征”；总损失联合重构/探索/对比/距离。
3. **个人判断（待原文确认）**：
   - 代码的“HP-MoE”弱于论文措辞：专家间融合无门控、专家内门控退化为 2 深度 softmax，更接近“双深度扩散增强 + 平均集成”而非完整 MoE 门控。
   - 论文的 Gumbel-Softmax / 反向去噪器可能为附图细节或后续版本实现，本仓库未体现。

---

### 8. 关键 Bug / 风险清单（综合人工核验）

| #   | 位置                                                 | 问题                                                                                                                                                                                 | 严重度          | 说明                                                                                                                          |
| --- | -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------- |
| 1   | `datasets.py:_data_sample_rec_task` 测试分支           | `test_samples = self.test_neg_items[index]` 中 **`index` 未定义**（函数形参为 `user_id, items, input_ids, target_pos, answer`，无 `index`）                                                     | **高**        | 仅当 `test_neg_items is not None`（sample 模式评测）触发 `NameError`；full_sort 路径不触发。`__getitem__` 中 `index=user_id`，但此处作用域无 `index`。 |
| 2   | `modules.py:AdaptiveDepthController.forward`       | 论文称 Gumbel-Softmax，代码为 `F.softmax(logits, dim=-1)`（无 Gumbel 噪声/温度）                                                                                                                 | 中            | 行为差异：无随机直通，深度选择确定性。影响“探索”语义（探索损失仍作用于此分布）。                                                                                   |
| 3   | `trainers.py:ELCRecTrainer.iteration`（IntentCL 分支） | `for cluster in self.model.cluster_centers:` 遍历 nn.Parameter 的行（每行为 1D 张量），随后 `cluster.query(...)` → 张量无 `query` 方法                                                                | **高（单专家路径）** | 单专家 IntentCL/Hybrid 路径会 `AttributeError`。双专家路径不依赖此循环（改用 `self.distance`+`argmin`），故默认 `dual_expert` 实验不受影响。                 |
| 4   | `trainers.py:ELCRecTrainer.iteration` Hybrid       | 存在死代码：`center_center_distance.flatten()[:-1].view(...)[...,1:].flatten()` 计算后被丢弃，未赋值                                                                                               | 低            | 实际损失用 `center_distance_loss = -center_center_distance.mean()`（含对角线 0）。该重塑行疑似调试遗留，建议删除。                                      |
| 5   | `trainers.py:Trainer.__init__`                     | `self.projection`（Linear(T*D,512)→BN→ReLU→Linear(512,D)）**定义但从未被调用**                                                                                                               | 低            | CL 分支直接 flatten 后送 `NCELoss`，未过 projection 头。死代码。                                                                           |
| 6   | `trainers.py:Trainer.__init__` 多粒度聚类               | `args.num_intent_clusters` 支持 `"4,8"` 多粒度，但 warm-up 循环对每个 cluster 反复 `model.cluster_centers.data = cluster.centroids`，**仅最后一个簇的 centroids 存活**                                     | 低            | 多粒度未真正落地；`model.cluster_centers` 为单 (K,D) 参数表。                                                                              |
| 7   | `main.py` dual_expert 装配                           | 双专家依赖 `args.enable_adaptive_diffusion=True` 才会建 controller；但 `main.py` **未自动置位**（需用户同时传 `--enable_adaptive_diffusion`），否则 `model.forward(..., sequence_lengths=)` 因无 controller 失败 | 中            | 配置依赖易错；README 应明示。                                                                                                          |
| 8   | `trainers.py:DualExpertTrainer.__init__`           | `self.diffusion_depth_e5=5` / `self.diffusion_depth_e10=5` 属性实际未使用（自适应专家走 `is_adaptive` 分支，不读固定深度）                                                                                 | 低            | 残留字段，易误导。                                                                                                                   |
| 9   | `models.py:SASRecModel.forward` adaptive           | `forward_diffusion(sequence_emb, ...)` 用**未 detach** 的 `sequence_emb`，梯度经加噪路径回传 encoder；仅 controller 输入 detach                                                                     | 设计项          | 符合“自适应分支可训练”意图，但需注意加噪路径梯度噪声较大。                                                                                              |
| 10  | `trainers.py:DualExpertTrainer` InstanceCL         | 对比学习使用 `model(...)` 返回的 **adaptive（加噪）** 表示，而非 raw 表示                                                                                                                              | 设计项          | 即对比视图本身带扩散噪声，与“自适应扩散增强”一致，但需确认是否预期。                                                                                         |

---

### C. 实验证据（主张—实验—边界 · 对应 PPT Slide 22）

#### C.1 研究问题的实验映射

| 研究问题                  | 对应实验                              | 数据来源 |
| --------------------- | --------------------------------- | ---- |
| RQ1：ADARec 是否优于 SOTA？ | Table 1（全序列）+ Table 3（骨干组合）       | P5-6 |
| RQ2：关键组件的贡献？          | Table 4（消融实验）                     | P6-7 |
| RQ3：计算效率如何？           | Table 5（训练/推理时间）                  | P7   |
| RQ4：超参数敏感性？           | Figure 4（λ分析）+ Figure 5（专家数量）     | P7   |
| RQ5：能否从稀疏数据构建层次化意图？   | Table 2（稀疏序列 ≤5）+ Figure 6（t-SNE） | P6-7 |

---

#### C.2 RQ1 — 主结果（Table 1，Page 5–6）

**实验条件**：Beauty / Sports / Toys / Yelp（遵循 ELCRec pipeline）；指标：Hit Rate @5/20（HR@K）、NDCG @5/20（N@K）；候选集 = 100 负样本 + 1 正样本；11 个基线。

#### Beauty

| 方法         | HR@5         | N@5          | HR@20        | N@20         |
| ---------- | ------------ | ------------ | ------------ | ------------ |
| Caser      | 0.0251       | 0.0145       | 0.0643       | 0.0298       |
| SASRec     | 0.0374       | 0.0241       | 0.0901       | 0.0387       |
| BERT4Rec   | 0.0360       | 0.0216       | 0.0984       | 0.0391       |
| IOCRec     | 0.0408       | 0.0245       | 0.0916       | 0.0444       |
| ICLRec     | 0.0495       | 0.0326       | 0.1072       | 0.0491       |
| ELCRec     | 0.0529       | 0.0355       | 0.1079       | 0.0509       |
| PDRec      | 0.0569       | 0.0380       | 0.1145       | 0.0541       |
| BASRec     | 0.0551       | 0.0385       | 0.1135       | 0.0539       |
| GlobalDiff | 0.0563       | 0.0382       | 0.1129       | 0.0531       |
| DiffDiv    | 0.0575       | 0.0375       | 0.1152       | 0.0548       |
| **ADARec** | **0.0600\*** | **0.0403\*** | **0.1187\*** | **0.0568\*** |

#### Sports

| 方法         | HR@5         | N@5          | HR@20        | N@20         |
| ---------- | ------------ | ------------ | ------------ | ------------ |
| Caser      | 0.0154       | 0.0114       | 0.0399       | 0.0178       |
| SASRec     | 0.0206       | 0.0135       | 0.0497       | 0.0216       |
| BERT4Rec   | 0.0217       | 0.0143       | 0.0604       | 0.0251       |
| IOCRec     | 0.0246       | 0.0162       | 0.0641       | 0.0280       |
| ICLRec     | 0.0263       | 0.0173       | 0.0630       | 0.0276       |
| ELCRec     | 0.0286       | 0.0185       | 0.0648       | 0.0286       |
| PDRec      | 0.0325       | 0.0218       | 0.0701       | 0.0328       |
| BASRec     | 0.0328       | 0.0220       | 0.0728       | 0.0331       |
| GlobalDiff | 0.0321       | 0.0215       | 0.0719       | 0.0325       |
| DiffDiv    | 0.0315       | 0.0211       | 0.0714       | 0.0318       |
| **ADARec** | **0.0341\*** | **0.0228\*** | **0.0745\*** | **0.0341\*** |

#### Toys

| 方法         | HR@5         | N@5          | HR@20        | N@20         |
| ---------- | ------------ | ------------ | ------------ | ------------ |
| Caser      | 0.0166       | 0.0107       | 0.0420       | 0.0179       |
| SASRec     | 0.0463       | 0.0306       | 0.0941       | 0.0441       |
| BERT4Rec   | 0.0274       | 0.0174       | 0.0688       | 0.0291       |
| IOCRec     | 0.0311       | 0.0197       | 0.0781       | 0.0330       |
| ICLRec     | 0.0586       | 0.0397       | 0.1130       | 0.0550       |
| ELCRec     | 0.0585       | 0.0403       | 0.1138       | 0.0560       |
| PDRec      | 0.0635       | 0.0433       | 0.1230       | 0.0615       |
| BASRec     | 0.0648       | 0.0431       | 0.1245       | 0.0618       |
| GlobalDiff | 0.0642       | 0.0435       | 0.1219       | 0.0613       |
| DiffDiv    | 0.0645       | 0.0428       | 0.1225       | 0.0605       |
| **ADARec** | **0.0691\*** | **0.0474\*** | **0.1298\*** | **0.0646\*** |

#### Yelp

| 方法         | HR@5         | N@5          | HR@20        | N@20         |
| ---------- | ------------ | ------------ | ------------ | ------------ |
| Caser      | 0.0142       | 0.0080       | 0.0406       | 0.0166       |
| SASRec     | 0.0160       | 0.0101       | 0.0443       | —            |
| BERT4Rec   | 0.0196       | 0.0121       | 0.0564       | —            |
| IOCRec     | 0.0222       | 0.0137       | 0.0640       | —            |
| ICLRec     | 0.0233       | 0.0146       | 0.0645       | —            |
| ELCRec     | 0.0248       | 0.0153       | 0.0667       | —            |
| PDRec      | 0.0253       | 0.0159       | 0.0688       | —            |
| BASRec     | 0.0255       | 0.0153       | 0.0681       | —            |
| GlobalDiff | 0.0251       | 0.0161       | 0.0685       | —            |
| DiffDiv    | 0.0258       | 0.0155       | 0.0691       | —            |
| **ADARec** | **0.0264\*** | **0.0167\*** | **0.0712\*** | **0.0291\*** |

**RQ1 关键数值**：

- Beauty HR@5 = 0.0600，相对次优 DiffDiv（0.0575）**+4.35%**；N@5 **+4.68%**；全表最高相对增益 Toys N@5 **+9.08%**
- Yelp 提升最小（HR@5 +2.33%），意图类方法在 Yelp 上普遍增幅有限（P6 原文解释：Yelp 数据集综合性强，意图解耦更困难）
- 标注 `*` 表示统计显著性；原文**未说明显著性水平**（p-value）和多重假设校正方法

> 【图表来源】：Table 1, P5-6

---

#### C.3 RQ1 补充 — ADARec 与不同意图感知骨干的组合（Table 3，Page 6）

| 数据集    | 骨干     | 骨干 HR@20 | +ADARec HR@20 | 提升          | 骨干 N@20 | +ADARec N@20 | 提升          |
| ------ | ------ | -------- | ------------- | ----------- | ------- | ------------ | ----------- |
| Beauty | IOCRec | 0.0916   | 0.1053        | +14.96%     | 0.0444  | 0.0501       | +12.84%     |
| Beauty | ICLRec | 0.1072   | 0.1165        | +8.68%      | 0.0491  | 0.0558       | +13.65%     |
| Beauty | ELCRec | 0.1079   | 0.1187        | +10.01%     | 0.0509  | 0.0568       | +11.59%     |
| Toys   | IOCRec | 0.0781   | 0.1098        | **+40.59%** | 0.0330  | 0.0515       | **+56.06%** |
| Toys   | ICLRec | 0.1130   | 0.1271        | +12.48%     | 0.0550  | 0.0628       | +14.18%     |
| Toys   | ELCRec | 0.1138   | 0.1298        | +14.06%     | 0.0560  | 0.0646       | +15.36%     |

- ADARec 叠加在 IOCRec 骨干上提升最显著（尤其 Toys，HR@20 +40.59%），证实框架通用性
- 仅报告了 Beauty 和 Toys，未测试 Sports / Yelp 的骨干组合

> 【图表来源】：Table 3, P6

---

#### C.4 RQ2 — 关键组件贡献（Table 4，Page 6–7）

**组件重要性排序（以 Beauty HR@20 下降幅度为例）**：

| 变体          | Beauty HR@20 | Beauty Δ   | Sports HR@20 | Sports Δ   | Toys HR@20 | Toys Δ     | Yelp HR@20 | Yelp Δ     |
| ----------- | ------------ | ---------- | ------------ | ---------- | ---------- | ---------- | ---------- | ---------- |
| Full        | 0.1187       | —          | 0.1298       | —          | 0.0745     | —          | 0.0712     | —          |
| −ADC        | 0.1141       | −3.29%     | 0.1246       | −4.01%     | 0.0715     | −4.03%     | 0.0690     | −3.09%     |
| −HDA        | 0.1105       | **−6.91%** | 0.1189       | **−8.40%** | 0.0681     | **−8.59%** | 0.0672     | **−5.62%** |
| −ADC−HDA    | 0.1075       | −9.44%     | 0.1132       | −12.79%    | 0.0645     | −13.42%    | 0.0661     | −7.16%     |
| −HP-MoE     | 0.1128       | −4.97%     | 0.1225       | −5.62%     | 0.0702     | −5.77%     | 0.0702     | −4.07%     |
| −C-A Router | 0.1172       | −1.26%     | 0.1281       | −1.31%     | 0.0736     | −1.21%     | 0.0705     | −0.98%     |
| −Lcont      | 0.1158       | −2.44%     | 0.1260       | −2.93%     | 0.0725     | −2.68%     | 0.0697     | −2.11%     |

**RQ2 结论**：

- **HDA 贡献最大**：移除后所有数据集一致下降 5.6%~8.6%，是最关键组件
- **HP-MoE 次之**：移除后下降 4.1%~5.8%，证实双专家分粒度解析有效
- **ADC 贡献稳定**：所有数据集一致下降 3.0%~4.0%，自适应深度控制有正向作用
- **C-A Router 影响最小**（约 −1%），内容感知路由在当前设置下作用有限
- −ADC−HDA 的下降幅度不等于 −ADC + −HDA 之和（Beauty: −9.44% ≠ −3.29%−6.91%=−10.20%），暗示存在非加性交互效应，原文**未分析该现象**

> 【图表来源】：Table 4, P6-7

---

#### C.5 RQ3 — 计算效率（Table 5，Page 7）

| 方法         | Beauty 训练(s) | Beauty 推理(ms) | Sports 训练(s) | Sports 推理(ms) | Toys 训练(s) | Toys 推理(ms) | Yelp 训练(s) | Yelp 推理(ms) |
| ---------- | ------------ | ------------- | ------------ | ------------- | ---------- | ----------- | ---------- | ----------- |
| SASRec     | 2.15         | 3.1           | 6.51         | 9.5           | 4.32       | 6.8         | 5.25       | 7.9         |
| ELCRec     | 3.81         | 5.9           | 11.28        | 16.1          | 7.88       | 11.5        | 9.00       | 13.2        |
| DiffDiv    | 7.55         | 14.8          | 20.10        | 31.5          | 15.60      | 22.4        | 18.23      | 26.1        |
| GlobalDiff | 8.90         | 21.2          | 23.50        | 45.8          | 18.95      | 33.6        | 21.75      | 38.5        |
| **ADARec** | **6.21**     | **10.5**      | **17.95**    | **24.9**      | **13.52**  | **18.7**    | **15.88**  | **19.3**    |

**RQ3 结论**：

- ADARec 训练时间**低于 GlobalDiff 约 24%~32%**，推理延迟**低于 GlobalDiff 约 50%~61%**
- ADARec 训练时间**高于 DiffDiv 约 18%~21%**（Beauty 例外 6.21 vs 7.55），推理延迟**低于 DiffDiv 约 29%~45%**
- ADARec **慢于非扩散基线**（ELCRec / SASRec），"高效"是相对扩散基线而言

> 【图表来源】：Table 5, P7；原文未报告 GPU 显存占用

---

#### C.6 RQ4 — 超参数敏感性（Page 7，Figure 4 & 5）

**λexpl 和 λdeno**（Figure 4）：

- 最优组合：λexpl = 0.01，λdeno = 0.1
- 曲线在最优值附近相对平坦（鲁棒性），即 λexpl 在 0.003~0.03、λdeno 在 0.05~0.15 范围内性能变化不剧烈
- 原文**未提供具体数值**，仅描述定性趋势

**专家数量**（Figure 5）：

- 扩展到 3 或 4 个专家**仅带来边际 HR@20 增益**（原文未给具体数值），但显著增加训练时间
- 2 个专家最优平衡效率和效果（支持论文默认双专家架构）

> 【图表来源】：Figure 4 & 5, P7

---

#### C.7 RQ5 — 稀疏数据构建层次化意图（Table 2 + Figure 6，Page 6–7）

#### 稀疏序列主结果（序列长度 ≤5）

| 数据集    | 指标    | BASRec | GlobalDiff | DiffDiv | ADARec       | 相对次优提升      |
| ------ | ----- | ------ | ---------- | ------- | ------------ | ----------- |
| Beauty | HR@5  | 0.0441 | 0.0428     | 0.0455  | **0.0531\*** | **+16.70%** |
| Beauty | N@5   | 0.0285 | 0.0272     | 0.0295  | **0.0353\*** | **+19.66%** |
| Beauty | HR@20 | 0.0910 | 0.0897     | 0.0925  | **0.1066\*** | **+15.24%** |
| Beauty | N@20  | 0.0421 | 0.0409     | 0.0427  | **0.0504\*** | **+17.48%** |
| Sports | HR@5  | 0.0261 | 0.0256     | 0.0265  | **0.0305\*** | **+15.09%** |
| Sports | N@5   | 0.0180 | 0.0174     | 0.0179  | **0.0199\*** | **+10.56%** |
| Sports | HR@20 | 0.0592 | 0.0585     | 0.0590  | **0.0659\*** | **+11.32%** |
| Sports | N@20  | 0.0270 | 0.0266     | 0.0269  | **0.0298\*** | **+10.37%** |
| Toys   | HR@5  | 0.0557 | 0.0550     | 0.0560  | **0.0618\*** | **+10.36%** |
| Toys   | N@5   | 0.0389 | 0.0387     | 0.0392  | **0.0431\*** | **+9.95%**  |
| Toys   | HR@20 | 0.1051 | 0.1043     | 0.1050  | **0.1169\*** | **+11.23%** |
| Toys   | N@20  | 0.0530 | 0.0518     | 0.0522  | **0.0586\*** | **+10.57%** |
| Yelp   | HR@5  | 0.0194 | 0.0189     | 0.0190  | **0.0200\*** | **+3.09%**  |
| Yelp   | N@5   | 0.0121 | 0.0112     | 0.0119  | **0.0124\*** | **+2.48%**  |
| Yelp   | HR@20 | 0.0528 | 0.0506     | 0.0522  | **0.0559\*** | **+5.87%**  |
| Yelp   | N@20  | 0.0217 | 0.0204     | 0.0209  | **0.0225\*** | **+3.69%**  |

**RQ5 关键发现**：

- Beauty 的稀疏提升最显著（HR@5 +16.70%、N@5 +19.66%）
- **Yelp 不支持"稀疏提升最显著"的主张**（仅 +3.09%/+2.48%），远低于 Beauty/Sports/Toys
- 仅测试了 ≤5 的极端稀疏情况，5~10、10~20 等中间段落**未报告**

**意图表征可视化**（Figure 6）：

- DiffDiv 在稀疏 Yelp 上意图坍缩到中心，**无法形成清晰聚类**
- ADARec 成功学习**清晰、分离良好的意图聚类**
- 原文**未提供其他数据集的可视化**，也**未报告聚类指标**（如轮廓系数）

> 【图表来源】：Table 2, P6；Figure 6, P7

---

#### C.8 证据边界（Conditions & Limitations）

**结果适用条件**：

1. 四个数据集（Beauty / Sports / Toys / Yelp），**未在其他领域**（新闻、视频、音乐推荐）验证
2. Top-K 隐式反馈推荐，**无用户满意度或多样性指标**
3. 仅测试了序列长度 ≤5 的稀疏子集，更细粒度分段（≤3、≤10）**未报告**
4. Table 3 仅测试了三种意图感知骨干，**与 Caser / SASRec 等非意图感知方法的组合效果未知**
5. 标注 `*` 的显著性**原文未说明 p-value 和多重假设校正方法**

**原文未证明的部分**：

1. **跨时间段泛化性**：未在不同时间段的同一平台数据上验证
2. **意图数量的可扩展性**：Figure 5 仅测试了 1~4 个专家
3. **专家分工的实证**：两个专家实际负责何种粒度，**原文无分析**
4. **消融交互效应**：−ADC−HDA ≠ −ADC + −HDA 之和，暗示非加性交互，原文**未分析**
5. **意图的语义标签**：Figure 6 可视化**未说明每个聚类的语义含义**

---

### D. 批判性核验（主张 vs 证据 vs 未证明 · 对应 PPT Slide 25）

> 按第六次课 PPT（Slide 25）方法论，对以下六个核心主张逐一区分**直接证据**、**作者解释**和**未证明部分**。结论标注"原文依据"或"推断"。

---

#### D.1 主张："数据稀疏导致意图坍缩"

**直接证据**：

- Figure 6（t-SNE）：DiffDiv 在稀疏 Yelp（序列长度 ≤5）上意图嵌入**坍缩到中心**，无法形成清晰聚类（P7）
- Table 2：DiffDiv 在 Beauty 稀疏子集 HR@5 = 0.0455，ADARec = 0.0531，说明稀疏条件下两类方法绝对性能均较低（P6）

**作者解释**：

- 缺乏足够的用户行为数据使意图表征在嵌入空间中趋同
- 扩散模型去噪轨迹在高噪声下无法充分学习多样化意图模式（P7）

**未证明部分**：

1. **因果性未建立**：仅展示"稀疏 → DiffDiv 意图坍缩"的相关性，**无消融实验**证明"增加数据量 → 坍缩消失"
2. **"稀疏"的定义唯一**：仅用序列长度 ≤5 定义，未探讨用户级别稀疏或其他稀疏度量
3. **阈值未知**：序列长度多少时开始出现意图坍缩？DiffDiv 在 Toys 稀疏子集 HR@5 仍有 0.0560，坍缩程度是否与 Beauty 同样严重？
4. **防坍缩机制的直接证据缺失**：Figure 6 仅在 Yelp 上展示，**未在其他数据集上验证**，也**无消融可视化**（如单独移除 ADC 后意图可视化）

> 【原文依据】：Fig.6, P7；Table 2, P6

---

#### D.2 主张："扩散去噪轨迹产生意图层次"

**直接证据**：

- Table 4（−HDA 消融）：移除 HDA 后 Beauty HR@20 从 0.1187 下降到 0.1105（−6.91%），Sports 下降 −8.40%，说明 HDA 是关键组件（P6-7）
- 论文方法描述：不同噪声水平 k 对应不同粒度意图——高噪声（k→Tu）→ 粗粒度，低噪声（k→0）→ 细粒度（P4）

**作者解释**：

- 去噪轨迹中的不同时间步隐式编码了从粗到细的意图层次
- 高噪声水平对应粗粒度意图（如品类偏好），低噪声水平对应细粒度意图（如品牌/价格偏好）（P4）

**未证明部分**：

1. **轨迹层次是学到的还是假设的**：**无直接证据**显示去噪轨迹中间步骤确实编码了层次化语义信息；**无 t-SNE 可视化中间去噪步骤的嵌入分布**
2. **层次数量无依据**：实验固定 2 个专家（对应 2 层意图），但"为什么是 2 层"没有消融证明
3. **与其他层次化方法的比较缺失**：未与 HAC 或多任务学习方法比较，无法证明扩散轨迹本身产生了有意义的层次结构
4. **意图层次的语义标签缺失**：Figure 6 展示了意图分离，但**未说明每个聚类对应什么语义**（如意图1="品类偏好"，意图2="价格敏感"）

> 【原文依据】：Table 4, P6-7；方法描述, P4

---

#### D.3 主张："HDA 利用完整去噪轨迹构建意图层次"

**直接证据**：

- Table 4（−HDA 和 −ADC−HDA 对比）：Beauty −HDA: 0.1105 → −ADC−HDA: 0.1075（额外下降 0.0030）；Sports −HDA: 0.1189 → −ADC−HDA: 0.1132（额外下降 0.0057）——说明 ADC 和 HDA 存在协同效应，但单独移除 HDA 的下降幅度已最大（P6-7）
- T_max 从 {4,6,8,10,12} 搜索——暗示使用了完整轨迹（P5）

**作者解释**：

- HDA 对齐了扩散前向过程与多粒度意图学习的对应关系（P4）
- 不同噪声水平映射到不同粒度的意图表示

**未证明部分**：

1. **"完整"轨迹的必要性未证明**：**没有测试部分轨迹**（如只用 T_max 的前 50% 或后 50% 步）是否足够
2. **T_max 选择依据缺失**：T_max ∈ {4,6,8,10,12} 的选择**未分析敏感性**
3. **与其他扩散步策略的比较缺失**：没有与常见策略（如只用最后一步或均匀采样）对比
4. **协同效应机制未阐明**：−ADC−HDA ≠ −ADC + −HDA 之和，暗示非加性交互，原文**未分析**

> 【原文依据】：Table 4, P6-7；实现细节, P5

---

#### D.4 主张："ADC 的自适应深度选择是高效的"

**直接证据**：

- Table 5：ADARec 训练时间 6.21~17.95s/epoch，推理延迟 10.5~24.9ms，**低于 GlobalDiff**（训练 8.90~23.50s，推理 21.2~45.8ms）（P7）
- Table 4（−ADC）：Beauty HR@20 下降 −3.29%，说明 ADC 对性能有贡献（P6-7）

**作者解释**：

- 自适应深度选择让简单序列使用更少步数，复杂序列使用更多步数，平衡效率和效果（P2）

**未证明部分**：

1. **自适应决策分布未透明化**：有多少比例的序列被分配到浅层 vs 深层？"简单/复杂"的区分标准是什么？**原文无报告**
2. **效率增益量化不完整**：Table 5 比较的是 ADARec 整体与 GlobalDiff，**没有单独剥离 ADC 的效率贡献**
3. **自适应 vs 固定深度的公平比较缺失**：**未报告固定 T_max（如 T_max=8）版本的训练/推理时间**作为对照
4. **与 ELCRec 的比较**：ADARec 仍比 ELCRec 慢（Beauty: 6.21s vs 3.81s），**"高效"若以 ELCRec 为参照则不成立**

> 【原文依据】：Table 5, P7；Table 4, P6-7

---

#### D.5 主张："HP-MoE 有效解耦粗细粒度意图"

**直接证据**：

- Table 4（−HP-MoE）：Beauty HR@20 下降 −4.97%，Sports 下降 −5.62%，所有数据集一致下降约 5%（P6-7）
- Table 3：IOCRec 骨干叠加 ADARec 后 Toys HR@20 +40.59%，暗示 HP-MoE 对底层意图感知能力弱的骨干有更大补偿作用（P6）

**作者解释**：

- HP-MoE 通过混合专家架构，将粗/细粒度意图分别分配给不同专家网络，实现解耦学习（P4）
- 路由器（C-A Router）根据用户意图类别动态分配计算（P4）

**未证明部分**：

1. **专家分工的实证缺失**：**无专家激活分布分析或专家输出可视化**——专家1是否真的专门负责粗粒度？
2. **"解耦"的量化指标缺失**：没有使用 DCI 分数或意图纯度等指标量化 HP-MoE 是否真的实现了比单一模型更好的意图解耦
3. **C-A Router 与 HP-MoE 的关系**：Table 4 单独移除 C-A Router 仅下降 −1.26%，但 C-A Router 是 HP-MoE 的路由组件——**如果路由不重要，HP-MoE 的动态分配功能是否真的被充分利用？**
4. **与标准 MoE 的比较缺失**：没有将 HP-MoE 与不含层次化先验的标准 MoE（如 Switch Transformer）比较，**无法判断"层次化先验"部分的独立贡献**

> 【原文依据】：Table 4, P6-7；Table 3, P6

---

#### D.6 主张："稀疏序列（≤5）提升最显著（+15%~+19%）"

**直接证据**：

- Table 2（Beauty）：HR@5 = 0.0531（相对 DiffDiv 0.0455）→ **+16.70%**；N@5 = 0.0353 → **+19.66%**（P6）
- Table 2（Sports）：HR@5 = 0.0305 → **+15.09%**（P6）
- Table 2（Toys）：HR@5 = 0.0618 → **+10.36%**（P6）

**作者解释**：

- HDA 利用完整去噪轨迹构建的意图层次提供了额外监督信号，弥补了数据不足
- ADC 的自适应深度选择让稀疏序列也能获得足够质量的增强（P1-2）

**未证明部分 ⚠️**：

1. **Yelp 不支持该主张**：Yelp 稀疏序列 HR@5 +3.09%、N@5 +2.48%，**远低于 +15%~+19% 的声称**
2. **"最显著"缺乏统计检验**：原文**未报告 p-value 或置信区间**，Beauty +16.70% vs Toys +10.36% 是否存在统计显著性差异未知
3. **中等稀疏序列（6~10）未测试**：5~10、10~20 段落的性能提升未知，**无法判断"越稀疏越显著"的线性关系**
4. **消融证据缺失**：Table 2 只比较了 ADARec 与扩散基线，**未报告稀疏子集上移除 HDA 或 ADC 后的性能**，无法证明"提升最显著"的直接原因
5. **相对 vs 绝对增益的混淆**：Beauty 绝对增益 0.0531−0.0455 = 0.0076（HR@5），Sports 绝对增益 0.0305−0.0265 = 0.0040——**相对提升大但绝对增益小，业务意义存疑**

> 【原文依据】：Table 2, P6；Table 4, P6-7

---

### E. 综合评估

| 主张              | 直接证据强度            | 主要漏洞                    |
| --------------- | ----------------- | ----------------------- |
| D.1 数据稀疏→意图坍缩   | ★★★（t-SNE 可视化）    | 缺乏因果消融，仅 Yelp 一个数据集     |
| D.2 扩散轨迹→意图层次   | ★★（HDA 消融）        | 中间轨迹无可视化，层次数量无依据        |
| D.3 HDA 利用完整轨迹  | ★★（消融下降幅度）        | 无部分轨迹对比，T_max 未分析敏感性    |
| D.4 ADC 自适应→高效  | ★★★（Table 5 效率数据） | 无固定深度对照，ADC 单独贡献未知      |
| D.5 HP-MoE 解耦意图 | ★★（消融 + 骨干组合）     | 专家分工无实证，解耦无量化指标         |
| D.6 稀疏提升最显著     | ★★★（Table 2 数值）   | **Yelp 不支持**该主张，统计显著性缺失 |
|                 |                   |                         |

**整体评价**：ADARec 的实验设计覆盖面广（11 个基线 × 4 个数据集 × 4 个指标），消融覆盖主要组件，效率分析提供了对比数据。关键弱点：①机制解释依赖理论假设，缺乏去噪轨迹可视化等**直接中间层证据**；②"稀疏提升最显著"的主张**因 Yelp 数据而无法成立**；③**统计显著性信息不完整**；④专家分工缺乏实证。

---

*本节（§C–§E）依据论文原文 Page 5–7 + 第六次课 PPT Slide 22 / 25 框架整理。所有数值与原文表格严格一致。*

---

### 9. 结论

1. **六步映射闭环成立**：论文 (1)-(13) 的核心运算在代码中均有明确落点——ADC 落在 `AdaptiveDepthController`（普通 softmax，非 Gumbel），HDA 落在 `diffusion_utils`（仅前向加噪），HP-MoE 落在 `models.forward` 的加权求和 + `DualExpertTrainer` 等权融合，总损失落在 `_train_one_expert` 的 `joint_loss`。
2. **最大证据缺口**：① Gumbel-Softmax 未实现；② 无独立反向去噪网络；③ 专家间融合为等权平均而非可学习门控。三者共同意味着本仓库的“HP-MoE + 自适应扩散”实现比论文摘要的措辞更轻量（更像一个“双深度扩散增强 + 平均集成 + 对比正则”系统）。**此结论为对代码+摘要的判断，需以论文发表稿最终确认（Slide 22/25 证据边界）。**
3. **可立即修复的高危项**：`datasets.py` 的 `index` 未定义 bug（#1）与单专家 IntentCL 的 `for cluster in model.cluster_centers`（#3）会在特定路径直接崩溃；双专家默认路径（代码主线）未触发二者。
4. **数值正确性已核验**：padding 三层清零、scaled-dot-product 的缩放/softmax 维度、rank+2 的 DCG 口径、loss 分母用有效位置数（非 B×T）均与标准 SASRec/InfoNCE 实现一致。
