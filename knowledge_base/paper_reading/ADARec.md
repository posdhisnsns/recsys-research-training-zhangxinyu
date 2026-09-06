# ADARec 文献阅读及代码复现

> 文献:ADARec(AAAI 2026,自适应扩散增强 + 混合专家 MoE 的序列推荐框架)
> 源码:https://github.com/Cxx-0/ADARec(最新仓库,原 ZhaoChao52 已重定向)
> 说明:本笔记基于论文原文、官方代码库及 README.md 整理,供学习参考。


## 文献阅读

### 0 符号约定

| 符号         | 含义                    | CLI 默认值                                   |     |
| ---------- | --------------------- | ----------------------------------------- | --- |
| $B$        | batch_size            | 256                                       |     |
| $T$        | max_seq_length        | 50(若 shorten_seq_to 则 = shorten+3)        |     |
| $D$        | hidden_size           | 128                                       |     |
| $H$        | num_attention_heads   | 2                                         |     |
| $L$        | head_size = $D/H$     | 64                                        |     |
| $K$        | num_intent_clusters   | 512(Toys/Sports) / 256(Yelp/Beauty/ml-1m) |     |
| $M$        | controller_num_levels | 2(硬编码)                                    |     |
| $T_{\max}$ | diffusion_t_max       | 50                                        |     |


### A. 研究问题

#### A.1 研究对象、任务、输入和预测目标是什么?

- **研究对象**:**序列推荐**(Sequential Recommendation, SR)领域,聚焦于"**数据稀疏**(data sparsity)场景下**用户意图层次建模**"这一子问题。
- **任务**:基于用户的历史行为序列,预测用户的下一个交互动作(predict users' next action)。
- **输入**:用户的历史交互记录(historical behavior / historical interactions),在 ADARec 中具体表现为**单条稀疏序列(single sparse sequence)**--即用户仅有少数历史交互。
- **预测目标**:用户下一个可能交互的物品(next item),即给定历史序列预测未来动作。

【原文依据:P1 摘要 "Sequential recommendation (SR) aims to predict users' next action based on their historical behavior";P1-2 引言 "用户活动遵循长尾分布,导致数据稀疏问题";P1 摘要 "reconstruct a user's intent hierarchy from a single sparse sequence"】

#### A.2 为什么这个问题值得研究?

- SR 模型的性能依赖丰富的交互数据("The performance of SR models relies on rich interaction data")。
- 真实场景中大量用户仅有少量历史交互 → 数据稀疏("many users only have a few historical interactions, leading to the problem of data sparsity")。
- 数据稀疏带来两个直接后果:
  1. 模型在稀疏序列上**过拟合**("model overfitting on sparse sequences");
  2. 阻碍模型捕获用户意图的**底层层次结构**("hinders the model's ability to capture the underlying hierarchy of user intents")。
- 最终结果:误解用户真实意图、推荐无关物品("misinterpreting the user's true intents and recommending irrelevant items")。
- 用户活动服从**长尾分布**(long-tail distribution),进一步加剧稀疏,严重削弱模型学习过程。

【原文依据:P1 摘要;P1-2 引言 "用户活动遵循长尾分布,导致数据稀疏问题,严重削弱模型学习过程"】

#### A.3 已有方法采用什么路线,具体不足是什么?

**已有路线(现有数据增强方法)**:

- 核心策略:通过生成**相关且多样**(relevant and varied)的数据来缓解过拟合。
- 两种增强方向:生成**相似物品**(保持相关性 relevance)与生成**多样物品**(增加多样性 diversity)。
- 实例:ELCRec 会过拟合稀疏物品--将"能量棒"简单归类为零食、将"登山靴"归为鞋类,错误地推断用户兴趣局限于这些宽泛类别。

**具体不足**:

- 忽视了**重建用户意图层次**(reconstructing the user's intent hierarchy)这一关键问题,而该层次在稀疏数据中已经**丢失**("lost in sparse data")。
- 增强的多样性**缺乏用户高层意图的引导**("lacks the guidance of user's high-level intent"),导致增强数据往往**偏离用户真实意图**("often fails to align with the user's true intents")。
- 相关工作综述指出:现有数据增强的核心是"通过生成更多物品来丰富训练集",而非建模意图结构,最终未能解决"数据稀疏导致的意图层次表征不完整"这一根本问题。

【原文依据:P1 摘要 "Existing data augmentation methods attempt to mitigate overfitting by generating relevant and varied data. However, they overlook the problem of reconstructing the user's intent hierarchy...";P1-2 引言 ELCRec 例子与 "多样性缺乏用户高层意图的引导";P3 相关工作 "现有数据增强核心策略:通过生成更多物品来丰富训练集,而非建模意图结构,最终未能解决数据稀疏导致的意图层次表征不完整这一根本问题"】

#### A.4 作者提出什么核心假设?

- **核心论点(核心假设)**:数据增强的关键**不在于盲目权衡相似性与多样性**("blindly trade-off between relevance and diversity"),**而在于构建用户意图的层次结构**("constructing the hierarchy of user intents")。
- **方法学假设**:扩散模型的**逐步去噪轨迹**(step-wise denoising trajectory)会迫使模型在更高抽象层次上推断用户意图;轨迹中每一步都产生**从细粒度到粗粒度**(fine- to coarse-grained)的丰富意图层次。

【原文依据:P1-2 引言 "核心论点:数据增强的关键不在于盲目权衡相似性与多样性,而在于构建用户意图的层次结构";P1-2 "扩散模型的去噪过程迫使模型在更高抽象层次上推断用户意图,每一步去噪轨迹产生从细粒度到粗粒度的丰富意图层次";P1 摘要 "we use its entire step-wise denoising trajectory to reconstruct a user's intent hierarchy from a single sparse sequence"】

#### A.5 问题成立需要哪些场景、数据或前提?

- **场景前提**:真实世界场景(real-world scenarios),用户交互呈**长尾分布**,大量用户**仅有少数历史交互**("only have a few historical interactions")。
- **数据前提**:输入为**单一稀疏序列**(single sparse sequence)--用户历史交互记录稀少,无法从中直接学到完整的意图层次,需被**重建**("reconstruct")。
- **前提假设**:意图层次在稀疏数据中会**丢失**("lost in sparse data"),且稀疏数据下聚类/层次表征效果差(Fig.2 的 t-SNE 可视化)。
- **评价前提**:方法须同时在**标准基准**(standard benchmarks)与**稀疏序列**(sparse sequences)上优于已有方法,方能证明问题成立与方案有效。

【原文依据:P1 摘要;P1-2 引言;P2 Fig.1、Fig.2;P1 摘要 "Experiments show ADARec outperforms state-of-the-art methods on standard benchmarks and on sparse sequences"】


### B. 解决方法

#### B.1 输入与输出:模型接收什么,最终产生什么?

**输入**:

- 用户历史交互序列 → 经骨干编码器(ELCRec)得到**序列嵌入** $E^{0}_{u} \in \mathbb{R}^{L \times D}$(数据形状为序列长度 $L$ × 隐维度 $D$)。
- 序列长度信息:归一化的**序列长度嵌入向量** $V_{l_u}$(length embedding)。
- 预定义的**最大扩散深度** $T_{\max}$(控制器可选深度的上限)。

**输出**:

- 经 HP-MoE 融合得到的**最终用户表征** $h_u$(对用户意图进行重构后得到的序列)
- 将 $h_u$ 送入预测层,得到**所有候选物品的概率分数** $y_u$(用于排序并推荐下一个物品)。

【原文依据:P3-4 "使用ELCRec作为骨干编码器获得序列嵌入 $E_u$";P3-4 ADC 输入 "$E^{0}_{u}$(用户历史序列嵌入,$\mathbb{R}^{L \times D}$)和 $v_{l_u}$(归一化序列长度嵌入向量)";P5 "最终融合的用户表征 $h_u$ 送入预测层计算所有候选物品的概率分数 $y_u$"】

#### B.2 方法数据流:原始数据如何经过模块形成预测?

**(a) 序列嵌入(前置)**:用户历史序列输入 ELCRec 骨干编码器 → 得到序列嵌入 $E^{0}_{u}$($\mathbb{R}^{L \times D}$);同时编码序列长度得到归一化长度嵌入 $V_{l_u}$。

**(b) ADC(自适应深度控制器)**:以 $E^{0}_{u}$ 与 $v_{l_u}$ 为输入,拼接后送入 GRU 捕获时序依赖,池化后经线性层得到跨深度($0,\dots,T_{\max}$)的 logits,再用 Gumbel-Softmax 估计器在 $T_{\max}+1$ 个离散选择中选出该序列的离散深度 $T_u$。辅助探索损失 $\mathcal{L}_{\text{expl}}$ 防止 ADC 坍缩到只选浅深度。ADC的作用是为每条序列**自适应确定增强深度**。

**(c) HDA(分层扩散增强)**:给定 $E^{0}_{u}$ 与 $T_u$,前向扩散逐步加高斯噪声生成噪声表征 $\{E^{0}_{u},\dots,E^{T_u}_{u}\}$(Eq.(4));再由可学习去噪网络 $D_{\theta}$ 反向去噪,重建各步去噪表征 $\hat{E}^{k}_{u}$(Eq.(5));迭代得到意图层次集合 $A_u = \{\hat{E}^{T_u}_{u},\dots,\hat{E}^{0}_{u}\}$(高 $k$ 元素=粗粒度意图,低 $k$ 元素=细粒度意图)。HDA 输出是整个结构化集合 $A_u$,而非单一向量。

**(d) HP-MoE(分层解析混合专家)**:细粒度专家 $E_{\text{fin}}$(Transformer,受 $\mathcal{L}_{\text{rec}}$ 驱动)捕获细粒度意图;粗粒度专家 $E_{\text{coar}}$(受排斥性对比损失 $\mathcal{L}_{\text{cont}}$ 驱动)学习用户不变本质。两级门控:第一级对每个层次 $k$ 用内容感知门控 $g_k$ 动态加权(Eq.(7));第二级按 $g_k$ 聚合全层次专家输出,得聚合细粒度表征 $z_{\text{fin}}$ 与聚合粗粒度表征 $z_{\text{coar}}$(Eq.(8)、(9));最后由 $z_{\text{coar}}$ 算出最终门控 $\text{gate}_u$(Eq.(10)),加权融合得最终用户表征 $h_u$(Eq.(11))。

**(e) Prediction(预测层)**:将 $h_u$ 送入预测层,计算所有候选物品概率分数 $y_u$,按分数排序产出"下一个物品"推荐。

【原文依据:P3-4 ADC;P4 HDA;P4 HP-MoE;P5 Prediction;P1-2 "三个关键创新:ADC、HDA、HP-MoE"】

#### B.3 新增与继承:哪些是作者提出的,哪些来自已有工作?

**继承(来自已有工作)**:

- **骨干编码器 ELCRec**:作为现成骨干(backbone)获得序列嵌入 $E^{0}_{u}$。
- **扩散基础 DDPM**:前向加噪采用 DDPM 的线性方差调度(linear variance schedule),去噪网络 $D_{\theta}$ 为已有扩散框架组件。
- **意图学习范式**:层次化意图建模思路继承自现有意图学习方法(intent learning)。

**作者新提出(ADARec 三个关键创新)**:

- **ADC(自适应深度控制器)**:自适应确定每条序列所需的增强深度,含 Gumbel-Softmax 选择 + 探索损失 $\mathcal{L}_{\text{expl}}$。
- **HDA(分层扩散增强)**:利用**整条逐步去噪轨迹**从单条稀疏序列重建粗/细粒度意图层次(核心区别于"仅生成相似/多样物品"的增强)。
- **HP-MoE(分层解析混合专家)**:解耦粗/细粒度意图的专家 + 内容感知两级门控融合。
- **总体框架论点**:用扩散去噪轨迹重建意图层次,而非盲目权衡相似性与多样性。

【原文依据:P3-4 "使用ELCRec作为骨干编码器";P4 "采用DDPM的线性方差调度";P1-2 "三个关键创新:ADC(自适应深度控制器)、HDA(分层扩散增强)、HP-MoE(分层解析混合专家)";P3 相关工作 "意图学习"】

#### B.4 关键公式:符号、形状(张量维度)、运算、输出和作用

> 格式:符号 / 形状 / 运算 / 输出 / 作用 / 原文依据(Equations)。凡原文未明确给出的形状,均标注"原文未明确"或"推断",不臆造维度。

**1 ADC - Eq.(1) 上下文编码**

- 符号:$E^{0}_{u}$(用户输入序列嵌入,$\mathbb{R}^{L \times D}$);$V_{l_u}$(归一化序列长度嵌入,**原文未给明确形状**,仅称向量);GRU 捕获时序依赖;Pool 池化。
- 形状:$h_c$ 为 GRU 池化后的上下文向量(原文未给维度,推断为 $\mathbb{R}^{D}$ 或 controller 隐藏维)。
- 运算:拼接 $[E^{0}_{u}; V_{l_u}]$ → GRU → Pool。
- 输出:$h_c$(序列级上下文表征)。
- 作用:为自适应深度选择提供序列级上下文。
- 【原文依据:P3-4 Eq.(1) $h_c = \text{Pool}(\text{GRU}([E^{0}_{u}; V_{l_u}]))$】

**2 ADC - Eq.(2) 深度 logits**

- 符号:$W_{\sigma}$、$b_{\sigma}$ 为可学习参数;$h_c$ 为1输出。
- 运算:线性变换 $\text{logits}_{u} = W_{\sigma} \cdot h_c + b_{\sigma}$。
- 输出:$\text{logits}_{u}$(跨 $0,\dots,T_{\max}$ 共 $T_{\max}+1$ 个可能深度的 logits)。
- 作用:为深度选择提供各深度打分。
- 【原文依据:P3-4 Eq.(2)】

**3 ADC - Eq.(3) Gumbel-Softmax 深度选择**

- 符号:$g$ 为来自 $\text{Gumbel}(0,1)$ 的 i.i.d. 样本;$\tau_{gs}$ 为 softmax 温度。
- 运算:$p_u = \text{Softmax}((\text{logits}_u + g) / \tau_{gs})$;前向取 argmax 得离散深度 $T_u$,反向用软概率 $p_u$(Gumbel-Softmax 估计器)。
- 输出:$p_u$($T_{\max}+1$ 维深度选择概率分布);$T_u$(选定的离散深度)。
- 作用:在 $T_{\max}+1$ 个离散深度中软/硬选择增强深度,实现自适应深度控制且前向可微。
- 【原文依据:P3-4 Eq.(3)】

**4 HDA - Eq.(4) 前向加噪**

- 符号:$\bar{\alpha}_{k}$ 为 DDPM 累积保留系数(线性方差调度);$E^{0}_{u}$(干净序列嵌入 $\mathbb{R}^{L \times D}$);$\epsilon_{k} \sim \mathcal{N}(0, I)$(与 $E^{0}_{u}$ 同形状的标准高斯噪声)。
- 形状:$E^{k}_{u}$ 与 $E^{0}_{u}$ 同形,$\mathbb{R}^{L \times D}$。
- 运算:闭式前向加噪 $E^{k}_{u} = \sqrt{\bar{\alpha}_{k}} \cdot E^{0}_{u} + \sqrt{1 - \bar{\alpha}_{k}} \cdot \epsilon_{k}$。
- 输出:$E^{k}_{u}$(第 $k$ 步噪声表征)。
- 作用:构造从干净到高度噪声的扩散视图,为去噪/意图层次提供多噪声水平输入。
- 【原文依据:P4 Eq.(4)】

**5 HDA - Eq.(5) 反向去噪**

- 符号:$\epsilon_{\theta}$ 为可学习去噪网络(原文指明实现为 MLP);$k$ 为噪声水平(同时作条件输入)。
- 形状:$\hat{E}^{k}_{u}$ 与 $E^{k}_{u}$ 同形,$\mathbb{R}^{L \times D}$。
- 运算:一步反向去噪 $\hat{E}^{k}_{u} = \sqrt{1 - \bar{\alpha}_{k}} \cdot \epsilon_{\theta}(E^{k}_{u}, k) / \sqrt{\bar{\alpha}_{k}}$。
- 输出:$\hat{E}^{k}_{u}$(第 $k$ 步去噪表征)。
- 作用:从各噪声步重建意图表征--高 $k$→粗粒度意图(信息被压缩,$I(\hat{E}_k; E_u) \ll H(E_u)$);低 $k$→细粒度意图(信息近全保留,$I(\hat{E}_k; E_u) \approx H(E_u)$);整条轨迹构成意图层次 $A_u$。
- 【原文依据:P4 Eq.(5);P4 高/低噪声水平段落】

**6 HP-MoE - Eq.(6) 粗粒度对比损失**

- 符号:$z_{u,k}$(粗粒度表征);$z_{u,k+1}$ 为正样本;$\mathcal{B}_k$ 含 $z_{u,k+1}$ 与同批其他用户的粗粒度表征(负样本);$\tau$ 温度;$k$ 范围 $\lfloor T_u/2 \rfloor,\dots,T_u-1$。
- 运算:基于 InfoNCE 的排斥性对比损失(repulsive contrastive loss)
  $$
  \mathcal{L}_{\text{cont}} = \sum_{k = \lfloor T_u/2 \rfloor}^{T_u - 1} - \log \left[ \frac{ \exp(\text{sim}(z_{u,k}, z_{u,k+1}) / \tau) }{ \sum_{z_j \in \mathcal{B}_k} \exp(\text{sim}(z_{u,k}, z_j) / \tau) } \right]
  $$
- 输出:标量损失。
- 作用:强制粗粒度专家学习"用户不变本质"(user-invariant essence),使不同粗粒度层次一致、并与同批其他用户区分。
- 【原文依据:P4 Eq.(6)】

**7 HP-MoE - Eq.(7) 内容感知门控**

- 符号:$\text{Pool}(\hat{E}^{k}_{u})$(第 $k$ 层去噪表征的池化,即内容表征);$\text{emb}(k)$(噪声水平嵌入);MLP 多层感知机。
- 形状:$g_k$ **原文未明确给出**(推断为逐层次标量或向量,取值 $[0,1]$)。
- 运算:$g_k = \text{Sigmoid}(\text{MLP}([\text{Pool}(\hat{E}^{k}_{u}); \text{emb}(k)]))$。
- 输出:$g_k$(第 $k$ 层门控概率)。
- 作用:内容感知路由--动态决定每个层次 $k$ 偏向细粒度专家($g_k$ 大)还是粗粒度专家($g_k$ 小)。
- 【原文依据:P4 Eq.(7)】
- 【职责归属】:这是 **C-A Router(Content-Aware Router)**,属 **HP-MoE 的融合端**;与 ADC(生成端控深度)是职责不同的两个模块。注意:代码中此模块被简化合并进了 ADC 的 `controller + probs`(详见 §2.1.5),故消融 ADC 时 C-A Router 也被一并废掉,这是本地消融 ADC 未复现论文 -3.9% 下降的根因之一。

**8 HP-MoE - Eq.(8) 聚合细粒度表征**

- 符号:$g_k$(7输出);$E_{\text{fin}}$(细粒度专家,Transformer);$\hat{E}^{k}_{u}$(HDA 输出);$\varepsilon$ 数值稳定项。
- 运算:
  $$
  z_{\text{fin}} = \frac{ \sum_{k=0}^{T_u} g_k \cdot E_{\text{fin}}(\hat{E}^{k}_{u}) }{ \sum_{k=0}^{T_u} g_k + \varepsilon }
  $$
- 输出:$z_{\text{fin}}$(聚合细粒度表征)。
- 作用:按门控对各层细粒度专家输出加权聚合,强调内容相关层次。
- 【原文依据:P4 Eq.(8)】

**9 HP-MoE - Eq.(9) 聚合粗粒度表征**

- 符号:同8,权重用 $(1 - g_k)$;$E_{\text{coar}}$(粗粒度专家)。
- 运算:
  $$
  z_{\text{coar}} = \frac{ \sum_{k=0}^{T_u} (1 - g_k) \cdot E_{\text{coar}}(\hat{E}^{k}_{u}) }{ \sum_{k=0}^{T_u} (1 - g_k) + \varepsilon }
  $$
- 输出:$z_{\text{coar}}$(聚合粗粒度表征)。
- 作用:按反向门控聚合各层粗粒度专家输出。
- 【原文依据:P4 Eq.(9)】

**10 HP-MoE - Eq.(10)(11) 最终融合**

- 符号:$\text{gate}_u = \text{Sigmoid}(W_g \cdot z_{\text{coar}} + b_g)$(Eq.(10),由粗粒度聚合表征算最终门控);$z_{\text{fin}}$、$z_{\text{coar}}$(89输出);$\odot$ 逐元素乘。
- 运算:$h_u = \text{gate}_u \odot z_{\text{fin}} + (1 - \text{gate}_u) \odot z_{\text{coar}}$(Eq.(11))。
- 输出:$h_u$(最终融合用户表征,送预测层)。
- 作用:平衡稳定的长期主题(粗粒度,$z_{\text{coar}}$)与瞬时细节(细粒度,$z_{\text{fin}}$),得到综合用户表征。
- 【原文依据:P4 Eq.(10)、Eq.(11)】

**11 总损失 - Eq.(12)(13)**

- 符号:$\mathcal{L}_{\text{rec}}$(推荐损失,Eq.(13));$\mathcal{L}_{\text{deno}}$(去噪 MSE);$\mathcal{L}_{\text{cont}}$(Eq.(6) 对比);$\mathcal{L}_{\text{expl}}$(探索负熵);$\lambda$ 加权系数。
- 运算:
  $$
  \mathcal{L}_{\text{tot}} = \mathcal{L}_{\text{rec}} + \lambda_{\text{deno}} \cdot \mathcal{L}_{\text{deno}} + \lambda_{\text{cont}} \cdot \mathcal{L}_{\text{cont}} + \lambda_{\text{expl}} \cdot \mathcal{L}_{\text{expl}} \quad \text{(Eq.(12))}
  $$
  $$
  \mathcal{L}_{\text{rec}} = - \sum_{i \in \mathcal{I}} p(i) \log(y_{u,i}) \quad \text{(Eq.(13),标准交叉熵)}
  $$
- 输出:标量总损失。
- 作用:联合优化推荐精度、意图层次重建质量、粗粒度不变性、深度探索多样性。
- 【原文依据:P5 Eq.(12)、Eq.(13)】

#### B.5 训练目标:模型实际优化什么,是否对应最终任务?

**实际优化的目标**是联合损失(Eq.(12)):

$$
\mathcal{L}_{\text{tot}} = \mathcal{L}_{\text{rec}} + \lambda_{\text{deno}} \cdot \mathcal{L}_{\text{deno}} + \lambda_{\text{cont}} \cdot \mathcal{L}_{\text{cont}} + \lambda_{\text{expl}} \cdot \mathcal{L}_{\text{expl}}
$$

各分项及其与最终任务的关系:

- **$\mathcal{L}_{\text{rec}}$(Eq.(13),标准交叉熵)**:直接对应**最终推荐任务**--预测用户下一个交互物品的概率分布 $y_u$,是任务主监督信号。✓ **直接对应最终任务**。
- **$\mathcal{L}_{\text{deno}}$(去噪 MSE:$\|\epsilon_{\theta}(E^{k}_u,k) - \epsilon_k\|^2$)**:优化可学习去噪网络 $D_{\theta}$ 准确估计噪声,支撑 HDA 意图层次重建质量(间接服务推荐)。
- **$\mathcal{L}_{\text{cont}}$(Eq.(6) 排斥性对比损失)**:强制粗粒度专家学习用户不变本质,提升粗粒度意图判别性。
- **$\mathcal{L}_{\text{expl}}$(负熵探索损失:$- \sum p_{u,i} \log p_{u,i}$)**:防止 ADC 坍缩到只选浅深度,鼓励深度选择多样性。

**是否对应最终任务**:主损失 $\mathcal{L}_{\text{rec}}$ 直接对应"下一个物品预测"这一最终推荐任务;其余三项为辅助正则/重构损失,服务于"从稀疏序列重建完整意图层次"这一研究问题的解法质量,间接提升最终推荐效果。论文在**标准基准与稀疏序列上均优于 SOTA**(P1 摘要),验证了训练目标与最终任务的一致性。

【原文依据:P4-5 Eq.(12)、Eq.(13);P3-4 $\mathcal{L}_{\text{expl}}$("引入辅助探索损失 $\mathcal{L}_{\text{expl}}$ 防止 ADC 坍缩");P4 $\mathcal{L}_{\text{deno}}$("去噪网络 $D_{\theta}$ 使用标准均方误差损失优化")、$\mathcal{L}_{\text{cont}}$(Eq.(6));P1 摘要 "outperforms state-of-the-art methods on standard benchmarks and on sparse sequences"】


## 代码复现

---

### §1 代码体系总览

#### 1.1 文件职责速查表

| 文件 | 职责 | 核心类/函数 |
|------|-----------|------------|
| `modules.py` | 基础神经网络组件(注意力、LayerNorm、ADC控制器等) | `LayerNorm`, `Embeddings`, `SelfAttention`, `Intermediate`, `Layer`, `Encoder`, `NCELoss`, `PCLoss`, `AdaptiveDepthController` |
| `models.py` | SASRec 主干 + 自适应扩散分支 + KMeans 聚类器 | `KMeans`, `SASRecModel` |
| `datasets.py` | 数据预处理:把原始序列加工成 Batch(含增强、Padding、标签切分) | `RecWithContrastiveLearningDataset` |
| `diffusion_utils.py` | 扩散工具:噪声调度表生成 + 前向加噪(无去噪网络) | `get_noise_schedule`, `forward_diffusion` |
| `trainers.py` | 训练引擎:损失函数、训练循环、评测逻辑、双专家调度 | `Trainer`, `ELCRecTrainer`, `DualExpertTrainer` |
| `main.py` | 入口:命令行参数解析 → 模型/数据组装 → 启动训练或评测 | `main()` |

#### 1.2 各文件说明

- **`modules.py`**:基础组件。LayerNorm 负责标准化、SelfAttention 让序列中每个物品看到上下文、AdaptiveDepthController 决定用多强的噪声增强。

- **`models.py`**:模型主体。把 item ID 变成向量(Embedding)、在 Transformer 中流转、输出序列表示。开启自适应扩散时分为两路:干净分支和加噪后编码分支。

- **`datasets.py`**:数据预处理。把交互序列切分成历史记录(input_ids)和预测目标(target_pos/target_neg),同时生成两份增强副本用于对比学习。

- **`diffusion_utils.py`**:扩散函数。`get_noise_schedule` 生成从 1 降到 0 的调度曲线、`forward_diffusion` 用闭式公式把干净嵌入变成带噪嵌入。

- **`trainers.py`**:训练流程。定义推荐损失 + 重构损失 + 探索损失 + 对比损失的加权组合,执行前向算 loss → 反向算梯度 → 更新参数。

- **`main.py`**:入口。组装所有组件,根据命令行参数(如 `--dual_expert`)选择单专家或双专家模式,训练至早停后加载最优权重测试。

---

### §2 六步映射体系

每个模块先说明功能,再看代码实现和论文对应关系。

---

#### 2.1 ADC(Adaptive Depth Controller,自适应深度控制器)

**功能**:输入用户序列 embedding 和长度,输出概率分布表示浅层/深层扩散的偏好比例。

##### 2.1.1 在模型中如何被调用

**代码位置**:`models.py:SASRecModel.forward`(L131~194)

```python
# L162: sequence_emb.detach() 阻断梯度,controller 只能看到 embedding 不能反传
probs = self.controller(sequence_emb.detach(), sequence_lengths)
```

`probs` 的 shape = `(batch_size, num_levels)`,即每个样本在每个扩散深度上的 softmax 概率。

##### 2.1.2 ADC 内部计算流程(modules.py L276~312)

ADC 的 forward 分三步:

**第一步:GRU 编码序列**(L295~296)
```
sequence_emb (B, T, D) → GRU → last_hidden.squeeze(0) → fused_representation (B, D)
```
把 T 个时间步的 embedding 序列压缩成 GRU 的最终隐状态,得到一个"序列级表示"。这里用的是 `squeeze(0)`,即只取最后一层最后一个时间步的输出,而不是所有时间步。

**第二步:长度 embedding**(L300~301)
```python
lengths_clipped = clamp(sequence_lengths, max=512)  # 防止越界
length_emb = self.length_embedding(lengths_clipped)  # (B, D)
```
把序列长度(被截断到 [0,512])查表变成一个 D 维向量。逻辑是:长序列和短序列可能有不同的最优扩散深度。

**第三步:拼接 + 线性输出**(L304~308)
```python
concat = [fused_representation, length_emb]    # (B, 2D)
logits = self.output_layer(concat)            # (B, 2)  ← 这里 num_levels=2
probs = F.softmax(logits, dim=-1)             # (B, 2)
```
用两个向量拼成一个 2D 向量,过一个 Linear 变成 2 维 logits,再 softmax 得到概率分布。

> 代码使用普通 softmax,而非 Gumbel-Softmax 或 argmax 采样。论文描述为"前向用 argmax 选深度,后向用概率回传",实际代码前向和后向都用 softmax 概率向量。

##### 2.1.3 probs 的三条去向

ADC 输出的 `probs` 在三个地方被使用:

**去向一:前向加权求和(参与数值计算)** - `models.py:forward` L176~194

```python
# L176: 提取所有深度 > 0 的索引
non_zero_depth_indices = torch.where(self.diffusion_levels > 0)[0]

# L180: 归一化使每行和为1
probs_for_diffusion = probs[:, non_zero_depth_indices]
probs_for_diffusion_normalized = probs_for_diffusion / (torch.sum(...)+1e-8)

# L182~189: 对每个深度分别加噪,再按概率加权求和
weighted_embs = []
for i, idx in enumerate(non_zero_depth_indices):
    depth = self.diffusion_levels[idx]
    t = torch.full((B,), depth - 1, ...)          # 扩散步数
    diffused_emb = forward_diffusion(sequence_emb, t, alphas_cumprod)  # 加噪
    weight = probs_for_diffusion_normalized[:, i].unsqueeze(1).unsqueeze(2)  # (B,1,1)
    weighted_embs.append(diffused_emb * weight)

final_sequence_emb = sum(weighted_embs)  # 加权求和
```

**解读**:probs 作为**加权系数**,对不同深度的加噪嵌入做软混合。权重越大,该深度的加噪嵌入对最终结果贡献越多。这是纯数值计算,probs 参与了前向传播的每一步。

**去向二:反向探索损失(梯度回传)** - `trainers.py:_train_one_expert` L660~662

```python
if probs is not None:
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    exploration_loss = -torch.mean(entropy)          # L662
```

**解读**:这里计算的是 `probs` 的负熵(negative entropy)。`entropy = -sum(p*log(p))`,取负号后 `exploration_loss = -entropy`。

- 当 `probs` 分布平坦(各深度概率相近)→ 熵大 → `exploration_loss` 小 → 梯度鼓励 controller 保持多样化
- 当 `probs` 分布尖锐(只选一个深度)→ 熵小 → `exploration_loss` 大 → 梯度惩罚这种"偏科"行为

这鼓励 ADC 不要总选同一个深度,而是探索不同的扩散强度。梯度通过这个损失回传到 controller 的参数。

**去向三:被 InstanceCL 分支丢弃** - `trainers.py:_train_one_expert` L675

```python
_, cl_sequence_output_inst, _ = model(cl_inputs_inst, sequence_lengths=cl_inputs_inst_lengths)
#                     ↑ 忽略第三个返回值 probs
```

**解读**:在 InstanceCL(实例级对比学习)的分支里,代码只用了 `raw_sequence_output`(第二个返回值),完全丢弃了 `probs`。这意味着 InstanceCL 的前向传播不会对 ADC 产生梯度--只有 exploration_loss 和 reconstruction_loss 能驱动 ADC 学习。

##### 2.1.4 实现补充:与论文描述的关键差异

> **结论**:代码中 `probs` **同时参与前向(作为加权系数)和反向(作为探索损失输入)**,而论文描述的是"前向用 argmax 选一个、后向用概率"。实际行为更接近"所有深度软混合"--每个扩散深度的加噪表示都参与最终输出,只是权重由 ADC 的 softmax 决定。

- **不是 argmax**:代码没有 `torch.argmax(probs, dim=-1)`,也没有 `torch.multinomial` 采样
- **不是 Gumbel-Softmax**:没有 `F.gumbel_softmax` 或添加 Gumbel 噪声
- **是纯 softmax 软混合**:所有深度按概率加权求和

##### 2.1.5 关键辨析:代码把 ADC 与 C-A Router 两个职责合并进了一个 `probs`

论文里 **ADC(生成端控深度)** 与 **C-A Router(解析端控融合)** 是两个职责不同、分属不同组件的模块,代码却把它们坍缩进同一个 `controller + probs`:

| 维度 | 论文 ADC | 论文 C-A Router | 代码实际(合并后) |
| --- | --- | --- | --- |
| **回答的问题** | 加噪到第几层?(决定层次深度) | 各层次各占多大权重?(层级加权) | 两件事都靠一个 probs |
| **输出** | 离散深度 $T_u$(整数) | 每层门控 $g_k$(向量) | `(B,2)` 的 softmax 概率 |
| **输入** | 序列 embedding + 长度 | 每层内容 $\text{Pool}(\hat{E}_u^k)$ + 噪声层级嵌入 $\text{emb}(k)$ | 序列 embedding + 长度(GRU) |
| **控制对象** | 扩散深度(生成端) | 层级加权融合(解析端) | 相邻两深度 `[4,5]`/`[9,10]` |
| **归属组件** | HDA(扩散增强) | HP-MoE(融合) | 无归属,合并 |

**职责串起来的先后顺序**(论文):
1. ADC 决定意图层次"有多少层、深到哪"($T_u$)
2. C-A Router 决定这 $T_u$ 层"各自占多少权重"($g_k$)来融合

**为什么代码里看起来像同一个东西**:三重简化把它们坍缩了——
1. ADC 的 0~k 连续深度 → 硬编码 `num_levels=2`(两档);
2. C-A Router 的逐层门控 $g_k$(MLP + emb(k)) → 同一个 softmax `probs`;
3. 两个独立模块两套参数 → 合并成一个 controller。

于是 `probs` 同时承担两份职责:
- **前向**:对两个深度 `[4,5]` 的加噪表示加权求和 = **C-A Router 的活**(Eq.7/8 层级加权);
- **反向**:负熵探索损失 $-\text{mean}(\text{entropy}(probs))$ = **ADC 的活**(保持"选深度"的多样性)。

**这直接解释了消融 ADC 做不出效果的原因**:无论固定 probs 还是去掉 controller,都同时"废掉"了 ADC 和 C-A Router;但深度只有相邻两档、探索损失权重仅 1e-4,两者的"自适应空间"都被 num_levels=2 压没了,所以指标几乎不动。论文 Table 4 能分别看到 "w/o ADC"(-3.9%)与 "w/o C-A Router"(-1.3%)的独立效果,正是因为它们是两个独立模块、各有独立参数和输入,可单独拆掉其中一个。

**第二重耦合:ADC 与 HDA 也串在一起(代码里 ADC 是枢纽)**。上面只讲了 `probs` 的"前向/反向"双重身份,但还有一个更根本的问题——`probs` 的**唯一消费方就是 HDA 的加噪加权**。看 forward 这条链:

```
controller(sequence_emb, lengths) → probs
        ↓
对每个 diffusion_level 加噪 → weighted_embs（HDA 的加噪,权重就是 probs）
        ↓
sum → item_encoder → adaptive_sequence_output
```

即:controller 输出的 `probs` 直接就是"每个深度加噪结果的加权系数",ADC 与 HDA 在代码里是**同一个 forward 里的先后两步**,不是两个可独立拆解模块。因此:

- **去 HDA 会连带废掉 ADC**:若 adaptive 分支直接等于 raw 且不调用 controller,则 `probs=None` → 探索损失归零 → ADC 不再被训练(早期 `ablate_hda` 实现的 bug);
- **去 ADC 会连带废掉 HDA 的"分层"语义**:固定单一深度后,HDA 退化为"固定单步加噪",不再产生意图层次。

论文里 ADC 输出离散深度 $T_u$,HDA 用 $T_u$ 做加噪+去噪,两者是独立模块,故 Table 4 里 `w/o HDA`(✓✗✓✓✓)能保留 ADC。代码因把两者揉进同一个 forward,无法像论文那样干净地"只拆 HDA、保留 ADC"。

**修正后的 `ablate_hda`(保留 ADC,只去加噪)**:
```python
if getattr(self.args, 'ablate_hda', False):
    probs = self.controller(sequence_emb.detach(), sequence_lengths)  # ADC 照常跑
    return raw_sequence_output, raw_sequence_output, probs           # adaptive=raw,探索损失保留
```
这样 ADC 仍在(controller 有梯度、探索损失照常训练),仅去掉 HDA 的"加噪+再编码";副作用是 `reconstruction_loss = MSE(raw, raw.detach()) = 0`(去 HDA 本就没有重构,合理)。

**结论(三层耦合,消融均难复现)**:代码里 **ADC、HDA、C-A Router 三个组件实际全被简化成"几乎不影响前向"的装饰**——
1. `num_levels=2` 使深度只有相邻两档,HDA 的"分层"名存实亡;
2. `expl_weight=1e-4` 使 ADC 的探索信号趋近于零;
3. 融合固定 0.5/0.5,无 C-A Router 门控;
4. 三者都挂在同一个 `probs`/`controller` 上,无法独立消融。

真正在起作用的是 **HP-MoE 的双专家集成(depth=4,5 vs 9,10 两套独立参数)+ SASRec 骨干**。这解释了为什么完整模型指标能对上论文,但每个组件消融都做不出论文的下降幅度(ADC -3.9%、HDA -6.9%、HP-MoE -9.5%)。

---

#### 2.2 HDA(Hybrid Diffusion Augmentation,混合扩散增强)

**功能**:对干净嵌入加噪声(非去噪),再过 Transformer Encoder,用加噪后的表示辅助推荐训练。

##### 2.2.1 扩散过程原理

DDPM 的前向扩散(Forward Process)有一个闭式解:

$$E_t = \sqrt{\bar{\alpha}_t} \cdot E_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

翻译成人话就是:

- $E_0$ = 原始干净嵌入(物品序列向量)
- $E_t$ = 加了 t 步噪声后的嵌入
- $\bar{\alpha}_t$ = 从第 1 步到第 t 步的信号保留系数之积(越往后乘得越小 → 噪声越多)
- $\epsilon$ = 每次随机采样的高斯噪声

`t` 越大 → $\bar{\alpha}_t$ 越小 → 噪声占比越高 → 嵌入越"模糊"。

##### 2.2.2 forward_diffusion 完整执行流程

**代码位置**:`diffusion_utils.py:forward_diffusion`(L15~23)

```python
def forward_diffusion(E_0, t, alphas_cumprod):
    # E_0: (B, T, D) 干净嵌入
    # t:   (B,) 每条样本对应的扩散步数(可以不同)
    # alphas_cumprod: (Tmax,) 预计算的累积系数

    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1)   # (B,1,1)
    sqrt_one_minus_alpha_t = torch.sqrt(1. - alphas_cumprod[t]).view(-1, 1, 1)  # (B,1,1)
    noise = torch.randn_like(E_0)              # (B, T, D) 标准正态噪声
    E_t = sqrt_alphas_cumprod_t * E_0 + sqrt_one_minus_alpha_t * noise  # 闭式混合
    return E_t
```

**逐步拆解**:

1. `alphas_cumprod[t]`:根据每条样本的扩散步数 t,取出对应的累积系数。形状 (B,),广播到 (B,1,1) 以便与 (B,T,D) 的嵌入相乘。
2. `sqrt(alphas_cumprod[t])`:取平方根,得到信号项的缩放系数。
3. `sqrt(1 - alphas_cumprod[t])`:取平方根,得到噪声项的缩放系数。
4. `torch.randn_like(E_0)`:生成与嵌入同形的高斯噪声(均值0,方差1)。
5. 加权求和:输出加噪嵌入 `E_t`。

**注意**:`t` 是 per-sample 的(`torch.full((B,), depth-1)`),所以每条样本加的噪声步数可以不同。但代码中同一 batch 里的所有样本共享同一个 `depth`,只是 `noise` 各自独立采样。

##### 2.2.3 adaptive_sequence_output 的生成路径

**代码位置**:`models.py:forward` L181~194

整个自适应扩散分支的流程如下:

```
input_ids (B, T)
    ↓ add_position_embedding (L154)
sequence_emb (B, T, D)
    ↓ controller (detach) → probs (B, 2)  [ADC 不参与此分支]
    ↓
for each diffusion_level:
    forward_diffusion(sequence_emb, t, alphas_cumprod)
        → diffused_emb (B, T, D)
    diffused_emb * weight(probs)
        → weighted_emb (B, T, D)
    ↓ sum all weighted_embs
final_sequence_emb (B, T, D)
    ↓ item_encoder (Transformer) (L192)
adaptive_item_encoded_layers (list of L layers)
    ↓ [-1] 取最后一层
adaptive_sequence_output (B, T, D)
```

**关键发现**:这里**没有独立的去噪网络**。加噪后的嵌入直接作为 Transformer Encoder 的输入,重新编码一遍。不是"先去噪再编码",而是"加噪后直接再编码"。

##### 2.2.4 自适应分支与原始分支的关系

- **原始分支(raw)**:直接用 `sequence_emb` → Encoder → `raw_sequence_output`
- **自适应分支(adaptive)**:`sequence_emb` → 加噪 → Encoder → `adaptive_sequence_output`

两路并行,共享同一个 `sequence_emb` 的输入(但加噪前会 detach 阻断梯度回传到 embedding 层)。

---

#### 2.3 HP-MoE(Hierarchical Prototype Mixture of Experts,分层原型双专家)

**功能**:用两个独立的 SASRec 专家分别处理浅层扩散(depth=4,5)和深层扩散(depth=9,10),推理时把两者的预测分数等权平均。

##### 2.3.1 双专家的模型构造

**代码位置**:`main.py:main` L178~184

```python
# 启用 --dual_expert 后:
levels_e1 = [4, 5]   # 专家1负责浅层扩散
levels_e2 = [9, 10]  # 专家2负责深层扩散

model_e5  = SASRecModel(args=args, diffusion_levels=levels_e1)  # 专家1
model_e10 = SASRecModel(args=args, diffusion_levels=levels_e2)  # 专家2
```

每个专家都是完整的 `SASRecModel`,拥有自己的:
- 物品嵌入表(`item_embeddings`)
- 位置编码(`position_embeddings`)
- Transformer Encoder(`item_encoder`)
- 可学习的簇中心(`cluster_centers`)
- 如果 `enable_adaptive_diffusion=True`,还有各自的 `controller`

##### 2.3.2 双专家训练流程(文字版流程图)

```
训练循环 (DualExpertTrainer.train_epoch)
│
├── 遍历每个 batch
│   │
│   ├── 【训练专家1 model_e5】
│   │   └─ _train_one_expert(model_e5, optim_e5, batch)
│   │       ├─ raw_loss  = cross_entropy(raw_output, targets)
│   │       ├─ adaptive_loss = cross_entropy(adaptive_output, targets)  [仅 adaptive 专家]
│   │       ├─ total_rec_loss = raw_loss + diff_rate * adaptive_loss
│   │       ├─ reconstruction_loss = MSE(adaptive, raw.detach())  [仅 adaptive 专家]
│   │       ├─ exploration_loss = -entropy(probs)  [仅 adaptive 专家]
│   │       ├─ cl_loss = InstanceCL + IntentCL + distance_loss
│   │       └─ joint_loss = rec*1.0 + deno*0.0001 + expl*0.0001 + cl
│   │
│   └── 【训练专家2 model_e10】
│       └─ _train_one_expert(model_e10, optim_e10, batch)
│           └─ (同上,两个专家独立 forward/backward,无参数共享)
│
└── valid_epoch:评测两个专家 + 融合预测
```

两个专家**完全独立训练**,各自有独立的 optimizer 和 early stopper。

##### 2.3.3 双专家推理流程(文字版流程图)

```
评测循环 (DualExpertTrainer.iteration, expert_to_eval='fused')
│
├── batch = (user_ids, input_ids, target_pos, target_neg, answers)
│
├── 【专家1前向】  raw_e5, _, _ = model_e5(input_ids)
│                  output_e5 = raw_e5[:, -1, :]              # 取末位
│                  pred_e5 = predict_full(output_e5, model_e5)  # 全量物品打分
│
├── 【专家2前向】  raw_e10, _, _ = model_e10(input_ids)
│                  output_e10 = raw_e10[:, -1, :]
│                  pred_e10 = predict_full(output_e10, model_e10)
│
└── 【等权融合】  fused = (pred_e5 + pred_e10) / 2.0
                   ↓
                   去训练集干扰(已有交互物品置零)
                   ↓
                   取 Top-20 推荐
```

##### 2.3.4 实现补充:与论文描述的关键差异

> **结论**:代码中**没有跨专家门控网络**。两个专家完全独立训练、独立推理,融合仅是推理时对两个专家的原始预测做简单平均。

- **没有门控网络**:论文可能描述了某种"根据输入动态分配专家权重"的网络,但代码中不存在这样的模块
- **不是 MoE 路由**:不是"每个 token/每个样本动态选择专家",而是"两个专家各自处理全量数据"
- **是双模型集成**:更接近两个独立模型的预测平均(类似于模型蒸馏 ensemble),而非"混合专家"

---

#### 2.4 损失函数全景图

##### 2.4.1 单专家(ELCRecTrainer)的损失构成

```
joint_loss
├── rec_weight * total_rec_loss          [推荐损失]
│   ├── raw_rec_loss                     [干净序列的 BPR 损失]
│   └── diffusion_aug_rate * diffusion_rec_loss  [加噪序列的 BPR 损失]
├── cf_weight * instance_cl_loss         [实例级对比损失,NCE]
├── intent_cf_weight * intent_cl_loss    [意图级对比损失,PCL]  [warm_up后]
├── trade_off * (sample_dist_loss + center_dist_loss)  [簇内聚+簇间散损失]  [Hybrid模式]
└── deno_weight * reconstruction_loss     [adaptive 重构损失]  [adaptive专家]
    expl_weight * exploration_loss       [负熵探索损失]       [adaptive专家]
```

##### 2.4.2 双专家(DualExpertTrainer)的损失构成

每个专家各自计算上述损失(通过 `_train_one_expert`),两个专家的损失**不共享梯度**,各自独立反向传播。

##### 2.4.3 各损失项的物理意义

| 损失项 | 作用 | 梯度回传给谁 |
|--------|------|-------------|
| `raw_rec_loss` | 学会从干净序列预测下一个物品 | `item_encoder`, `item_embeddings`, `position_embeddings` |
| `diffusion_rec_loss` | 学会从带噪序列预测下一个物品(鲁棒性) | 同上 + `item_encoder`(加噪路径) |
| `reconstruction_loss` | 让 adaptive 表示尽量接近 raw 表示 | `controller`, `item_encoder`(adaptive分支) |
| `exploration_loss` | 鼓励 controller 不偏科,探索多样深度 | `controller`(GRU + output_layer) |
| `instance_cl_loss` | 同一样本的两个增强视图应该相似 | `item_encoder`, `item_embeddings` |
| `intent_cl_loss` | 同意图的样本应该相似(跨样本) | 同上 |
| `cluster_distance_loss` | 簇内紧凑、簇间分散 | `cluster_centers` |

---

### §3 Debug 观察清单

#### 3.1 断点调试清单(按执行顺序)

| 顺序 | 观察点 | 文件:行号 | 看什么变量 | shape | 正常范围 |
|------|--------|----------|-----------|-------|---------|
| 1 | `input_ids` | `datasets.py:__getitem__` L161 返回 | 物品ID序列 | `(batch, max_seq_len)` | int, 0=padding |
| 2 | `sequence_emb` | `models.py:add_position_embedding` L103 | 物品+位置向量 | `(batch, T, 128)` | float32, ~N(0,0.05) |
| 3 | `raw_sequence_output` | `models.py:forward` L158 | 编码后序列表示 | `(batch, T, 128)` | float32 |
| 4 | `probs` | `models.py:forward` L164 | 各深度概率分布 | `(batch, 2)` | float32, 和=1, 0~1 |
| 5 | `diffused_emb` | `diffusion_utils.py:forward_diffusion` L21 | 加噪后的嵌入 | `(batch, T, 128)` | float32, 方差在变大 |
| 6 | `adaptive_sequence_output` | `models.py:forward` L194 | 自适应分支编码结果 | `(batch, T, 128)` | float32 |
| 7 | `reconstruction_loss` | `trainers.py:_train_one_expert` L630 附近 | 自适应 vs 原始表示的MSE | scalar | 0.0 ~ 1.0 |
| 8 | `exploration_loss` | `trainers.py:_train_one_expert` L662 | probs 的负熵 | scalar | 负数,绝对值 ≤ log(2) |
| 9 | `joint_loss` | `trainers.py:_train_one_expert` L732 | 总损失 | scalar | >0 |
| 10 | `rating_pred` | `trainers.py:iteration` L838 | 融合预测得分 | `(batch, num_items)` | float32 |
| 11 | `HIT@5/NDCG@5` | `trainers.py:get_full_sort_score` L146 | 评测指标 | scalar | 0~1 |

#### 3.2 梯度流动检查(判断模块是否被正确训练)

**检查 ADC 是否有梯度**:
```python
# 在 trainer._train_one_expert 中:
print(model.controller.output_layer.weight.grad)  # 非 None 表示 ADC 在学
print(model.controller.gru.weight_hh_l0.grad)      # 同上
```
如果全是 None,检查 exploration_loss 是否真的被加到了 joint_loss 里。

**检查加噪分支是否有梯度**:
```python
# 在 _train_one_expert 中:
print(model.item_encoder.layer[0].attention.query.weight.grad)  # 非 None 表示 Encoder 在学加噪分支
```

**检查 probs 是否在探索**:
```python
# 每隔几步打印:
print("probs mean:", probs.mean().item())   # 接近 0.5 表示不偏科
print("probs max:",  probs.max().item())    # 接近 1.0 表示很偏科
```

#### 3.3 常见报错速查

| 报错信息 | 可能原因 | 修复位置 |
|---------|---------|---------|
| `IndexError: list index out of range` | `datasets.py L121` 用了未定义的 `index` | datasets.py `_data_sample_rec_task` |
| `RuntimeError: dimension too short` | `forward_diffusion` 的 `t` 超出了 `alphas_cumprod` 长度 | main.py `diffusion_t_max` 参数 |
| `AttributeError: 'NoneType' object has no attribute 'controller'` | 推理时调用了有 controller 的接口但传了普通模型 | trainers.py `iteration` |
| `CUDA out of memory` | batch_size 过大或 sequence 太长 | main.py `batch_size` 或 `max_seq_length` |
| `NaN loss` | 学习率太高或 `alphas_cumprod` 有 0 | 降低 lr 或检查 `diffusion_t_max` |

---

### §4 代码阅读三问

#### 4.1 "这段代码在算什么?"

分析陌生代码段的三个维度:

**输入**:函数参数类型与形状。例如 `forward_diffusion(E_0, t, alphas_cumprod)` 三个张量形状分别为 `(B,T,D)`, `(B,)`, `(Tmax,)`。

**输出**:函数返回值。`forward_diffusion` 返回 `E_t` 即加噪后的嵌入,形状与输入 `E_0` 相同。

**数学操作**:拆解为"取索引 → 算 sqrt → 采样噪声 → 加权求和"四步。

#### 4.2 "数据在模块间的流转"

文字版简要流程(详细数据流图见 **§5 数据流链路图**):

1. `datasets.py.__getitem__` 把原始序列切分成 input_ids / target_pos / target_neg / cf_views / sequence_len
2. `trainers.py.iteration` 根据专家类型调用 `models.py.forward`
3. 模型分叉:raw 分支(无加噪) + adaptive 分支(经 ADC 控制器 + HDA 加噪)
4. 两分支输出分别算推荐损失,adaptive 分支额外算重构损失 + 探索损失
5. 对比学习分支独立运行 InstanceCL + IntentCL
6. 最终 `joint_loss = rec + deno + expl + cl` 反向传播

#### 4.3 "这段代码和论文公式对应吗?"

论文公式与代码实现的系统对照见 **§D 论文主张 vs 代码实现**，逐模块列出了 6 个核心主张的对应关系与一致性判断。

---

### §5 数据流链路图

#### 5.1 单 forward pass 完整数据流

```
推理/训练时的前向路径(以 DualExpertTrainer iteration 为例)

batch = (user_ids, input_ids, target_pos, target_neg, answers, sequence_len)
    │
    │  【===== 专家1 model_e5 前向 =====】
    ▼
input_ids (B, T)  ──►  add_position_embedding
                            │
                            ▼
                       sequence_emb (B, T, D) = item_emb + pos_emb + LayerNorm + Dropout
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
        [原始分支]                  [自适应分支](仅 enable_adaptive_diffusion + training)
                │                       │
                ▼                       ▼
        item_encoder              controller(sequence_emb.detach(), seq_len) → probs (B, 2)
        (Transformer)                  │
                │               ┌─────┴─────┐
                ▼               ▼           ▼
        raw_output (B, T, D)   depth=4     depth=5
                                       │           │
                                       ▼           ▼
                                  forward_diffusion  forward_diffusion
                                       │           │
                                       ▼           ▼
                                  emb*w4        emb*w5
                                       │           │
                                       └─────┬─────┘
                                             ▼
                                    final_sequence_emb (B, T, D)
                                             │
                                             ▼
                                    item_encoder → adaptive_output (B, T, D)
                                             │
                                             ▼
    ◄──────────────────────┬──────────────────┘
                           │                 │
                           ▼                 ▼
                    raw_output[:,-1,:]   adaptive_output[:,-1,:]
                           │                 │
                           ▼                 ▼
                   BPR交叉熵(raw)      BPR交叉熵(adaptive)
                           │                 │
                           └────────┬────────┘
                                    ▼
                            total_rec_loss
                                    │
                ┌───────────────────┼───────────────────┐
                ▼                   ▼                   ▼
        reconstruction_loss   exploration_loss   cl_losses
        (MSE adaptive vs raw)  (-entropy probs)  (NCE+PCL)
                │                   │                 │
                └───────────────────┴─────────────────┘
                                │
                                ▼
                           joint_loss
                                │
                                ▼
                          backward()
                          optimizer.step()
                                │
    │  【===== 专家2 model_e10 同上,独立参数 =====】


推理时的融合路径(DualExpertTrainer.iteration expert_to_eval='fused')

model_e5(input_ids)  ──► raw_e5[:, -1, :] ──► predict_full ──► pred_e5 (B, num_items)
                                                                              │
model_e10(input_ids) ──► raw_e10[:, -1, :] ─► predict_full ──► pred_e10 ──┤
                                                                              │
                                                                  fused = (pred_e5 + pred_e10) / 2
                                                                              │
                                                                  去训练集干扰(已有交互位置零)
                                                                              │
                                                                  取 Top-20 推荐
```

#### 5.2 关键变量速查表

| 变量名 | 定义位置 | shape | 含义 |
|--------|---------|-------|------|
| `input_ids` | `datasets.py:__getitem__` | `(B, max_seq_len)` | 交互序列的 item ID,含左侧 padding |
| `sequence_emb` | `models.py:add_position_embedding` L154 | `(B, T, 128)` | 物品向量 + 位置向量的融合结果 |
| `raw_sequence_output` | `models.py:forward` L158 | `(B, T, 128)` | 原始序列的 Transformer 编码结果(末层) |
| `probs` | `modules.py:AdaptiveDepthController.forward` L308 | `(B, 2)` | ADC 输出的深度选择概率分布 |
| `diffused_emb` | `diffusion_utils.py:forward_diffusion` L21 | `(B, T, 128)` | 加噪后的序列嵌入 |
| `final_sequence_emb` | `models.py:forward` L191 | `(B, T, 128)` | 各深度加噪嵌入按概率加权求和后的结果 |
| `adaptive_sequence_output` | `models.py:forward` L194 | `(B, T, 128)` | 自适应分支的编码结果 |
| `reconstruction_loss` | `trainers.py:_train_one_expert` L630 附近 | scalar | MSE(raw, adaptive.detach()) |
| `exploration_loss` | `trainers.py:_train_one_expert` L662 | scalar | -mean(entropy(probs)) |
| `joint_loss` | `trainers.py:_train_one_expert` L732 | scalar | 所有损失项的加权和 |
| `pred_e5`, `pred_e10` | `trainers.py:iteration` L838 | `(B, num_items)` | 两个专家的全量物品预测得分 |
| `fused` | `trainers.py:iteration` L838 | `(B, num_items)` | 两专家预测的等权平均 |

---

### §6 关键 Bug / 风险清单

#### Bug #1:datasets.py 中 test 分支使用了未定义的 `index` 变量

**文件**:`datasets.py`
**位置**:`_data_sample_rec_task` 函数内(原约 L121,**修复后位于 L124**)
**问题代码(已修复)**:
```python
test_samples = self.test_neg_items[index]  # ← index 未定义!(旧代码)
```

**原因**:`index` 是 `__getitem__` 的参数(数据集索引),但在 `_data_sample_rec_task` 这个函数里,`user_id` 是单独传入的变量,用 `user_id` 而不是 `index` 才对。

**影响**:训练集和验证集不受影响(走 else 分支)。**补充**:当前 `main.py` 构造测试集时未传入 `test_neg_items`(`test_neg_items=None`),该分支实际不会触发,属**潜伏 bug**;一旦启用 sample/test_neg 模式即崩溃(NameError)。

**修复(已实施)**:datasets.py L124 已将 `self.test_neg_items[index]` 改为 `self.test_neg_items[user_id]`。

---

#### Bug #2:trainers.py iteration 中 Hybrid 模式的 dead code

**文件**:`trainers.py`
**位置**:`ELCRecTrainer.iteration` Hybrid/IntentCL 分支(修复前约 L442,现已删除该行)
**问题**:Hybrid 分支在计算簇间距离时有一行纯 dead code--`center_center_distance.flatten()[:-1].view(center.shape[0]-1, center.shape[0]+1)[:,1:].flatten()`,其计算结果**未赋给任何变量**,立即被丢弃;上方注释声称"提取上三角去对角",但实际 `center_distance_loss` 用的是全矩阵均值 `center_center_distance.mean()`(含对角线,对角线距离恒为 0,不影响推远方向)。原笔记描述"条件判断 else 永不执行"与该处真实代码不符,此处按实际代码修正。

**影响**:代码冗余,不影响训练结果(dead code 无副作用),但误导维护者。

**修复(已实施)**:trainers.py 已删除该未赋值语句及其误导注释,保留 `center_distance_loss = -center_center_distance.mean()`。

---

#### Bug #3:DualExpertTrainer iteration 的评测逻辑

**文件**:`trainers.py`
**位置**:`DualExpertTrainer.iteration` L838
**问题**:`fused=(pred_e5 + pred_e10)/2` 的融合发生在原始分支的输出上(`raw_output`),而非自适应分支的输出。论文中 HP-MoE 的融合语义是"不同深度专家的加权",但代码里两个专家都只取了 raw 输出。

**影响**:评测时忽略了 diffusion augmentation 的效果,只利用了两套独立 Transformer 参数的集成。若希望同时利用自适应扩散增强,需要在评测时也调用含 controller 的完整 forward 并取 `adaptive_output`。

> **状态说明**:该条属"论文-代码语义偏差"的设计观察,非崩溃性 bug。评测时 `model.forward` 因 `self.training=False` 不执行扩散分支(只返回 raw),故即便调用完整 forward 也拿不到 `adaptive_output`;若强行令 `training=True` 取 adaptive,会引入 dropout 与随机加噪,反而不当。因此**保持代码现状**,仅作记录。

---

#### Bug #4:trainers.py 评测阶段 `.toarray()` 引发 host RAM OOM(已修复)

**文件**:`trainers.py`
**位置**:`ELCRecTrainer.iteration`(原 L517)与 `DualExpertTrainer.iteration`(原 L891)--评测去训练集干扰的掩码构造
**问题代码(已修复)**:
```python
rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0  # ← 每批把 (B, n_items) 稀疏矩阵转稠密 int32
```
**原因**:`self.args.train_matrix` 是 scipy 稀疏矩阵。`train_matrix[batch_user_index].toarray()` 每批验证都把 `(256, 18359)` 的子矩阵转成**稠密 int32 数组(约 17.9 MiB)**。训练 189 epoch(约 4.5 小时)后,host RAM 余量被压到临界,连这 17.9 MiB 都分配不出 → `numpy.core._exceptions._ArrayMemoryError`。`rating_pred` 本身(18.8 MiB float32)尚能分配,叠加此 17.9 MiB 才超限。

**影响**:双专家训练跑到约 189/400 epoch 的验证阶段崩溃,损失全部进度(代码无续训机制)。属**崩溃性 bug**(本仓库首次实跑即触发)。

**修复(已实施)**:改用稀疏索引,避免稠密转换(trainers.py L519 / L895):
```python
batch_train = self.args.train_matrix[batch_user_index]  # CSR 子矩阵,仅保留非零项
rating_pred[batch_train.nonzero()] = 0                  # nonzero() 返回 (行,列) 索引,约 200KB
```
已用 scipy 验证 `nonzero()` 结果与原 `.toarray() > 0` 掩码**逐元素一致**(行索引对齐 batch 位置)。修复后验证阶段瞬时峰值内存降低约 22.6 MiB(int32 17.9 MiB + bool 掩码 4.7 MiB),足以避免 OOM。

**兜底**:若重跑仍 OOM(机器 RAM 偏紧),将 `--batch_size 256` 降到 `--batch_size 128`(验证时 `rating_pred` 大小减半)。

---

#### Bug #5:`--do_eval` 双专家路径加载权重缺少 `_e5.pt` / `_e10.pt` 后缀(已修复)

**文件**:`main.py`
**位置**:`main()` 的 `if args.do_eval:` 分支(**修复后位于 L188-189**;对照训练结束分支 L200-201 已正确带后缀)
**问题代码(已修复)**:
```python
trainer.model_e5.load_state_dict(torch.load(checkpoint_prefix))   # ← 缺 "_e5.pt" 后缀!
trainer.model_e10.load_state_dict(torch.load(checkpoint_prefix))   # ← 缺 "_e10.pt" 后缀!
```

**原因**:早停保存的权重文件名是 `checkpoint_prefix + "_e5.pt"` / `"_e10.pt"`(由 `EarlyStopping` 的 `checkpoint_path` 决定,见 main.py L159/L163),而 `checkpoint_prefix` 本身是不带后缀的前缀(如 `output/Beauty/ELCRec-Beauty-1-jianduan555`)。`--do_eval` 分支直接 `torch.load(checkpoint_prefix)`,加载一个**不存在的裸路径** → `FileNotFoundError: [Errno 2] No such file or directory: 'output/Beauty/ELCRec-Beauty-1-jianduan555'`。训练结束分支(L200-201)用的是带后缀的 `checkpoint_prefix + "_e5.pt"`,故训练→评测链路本身正常,唯独 `--do_eval` 单独入口漏了后缀。

**影响**:`--do_eval` 模式(加载已训权重做测试评测)**无法启动**,对双专家配置必然崩溃。属**崩溃性 bug**,但仅在「跳过训练、直接评测已有权重」时触发。

> **额外坑**:初次修复时曾误加 `map_location=args.device`,但 `args.device` 在 `main.py` 中从未定义(仅有 `args.cuda_condition`),会触发 `AttributeError`。已去掉 `map_location`,与能正常工作的训练结束分支(L200-201)保持一致。

**修复(已实施)**:main.py L188-189 改为带后缀加载(与 L200-201 一致):
```python
trainer.model_e5.load_state_dict(torch.load(checkpoint_prefix + "_e5.pt"))  # 加载专家1权重
trainer.model_e10.load_state_dict(torch.load(checkpoint_prefix + "_e10.pt"))  # 加载专家2权重
```
`py_compile` 通过;`output/Beauty/` 下 `_e5.pt` / `_e10.pt` 权重文件确认存在。

---

#### 风险 #1:ADC 的 detach 操作

**位置**:`models.py:forward` L162
```python
probs = self.controller(sequence_emb.detach(), sequence_lengths)
```

`sequence_emb.detach()` 阻断了梯度从 controller 回传到 embedding 层。这意味着:

- embedding 层**不会**因为深度选择偏好而调整(ADC 只能通过 exploration_loss 间接学习)
- 如果 exploration_loss 权重(`expl_weight`)太小,ADC 的梯度信号会很弱

**建议**:如果希望 embedding 和 controller 联合优化,考虑去掉 `.detach()` 或增大 `expl_weight`。

---

#### 风险 #2:双专家完全独立,无参数共享

**位置**:`main.py:dual_expert` 分支

两个专家(`model_e5` 和 `model_e10`)各自维护独立的:
- 物品嵌入表
- 位置编码
- Transformer 参数
- 簇中心

这导致:
- 显存占用翻倍(2× 所有参数)
- 没有跨专家的知识迁移
- 如果数据集较小,容易各自过拟合

如果想实现论文中"分层原型"的效果(不同深度的专家共享底层表示),需要在模型初始化或损失函数层面引入共享机制。

---

#### 风险 #3:diffusion_levels 的魔法数字

**位置**:`main.py` L179~180
```python
levels_e1 = [4, 5]   # 硬编码
levels_e2 = [9, 10]  # 硬编码
```

扩散深度级别是直接写死的两个区间,没有超参数化。如果想调优深度范围,需要改代码而非调参。

---

#### 训练改进 #1:双专家断点续训机制(已实施)

**位置**:`main.py`(`--resume` 参数 + 训练循环)、`trainers.py`(`DualExpertTrainer.save_resume_ckpt` / `load_resume_ckpt`)

**动机**:原代码无续训机制,进程中断(OOM / 手动停止 / 崩溃)后只能从头重训。Beauty 跑至 epoch 341 中断即丢失全部进度,重跑代价极高。

**实现**:
- 新增显式开关 `--resume`(默认关闭,绝不破坏"从头训练"的既有行为)。
- 双专家每个 epoch 训练+验证后,把完整训练状态打包存为 `{output_dir}/{model_name}-{data_name}-{model_idx}-jianduan555_resume.pt`:
  - 两个专家模型权重:`model_e5` / `model_e10`
  - 两个 Adam 优化器状态:`optim_e5` / `optim_e10`
  - 两个早停器状态:`best_score`、`score_min`、`counter`、`early_stop`、`patience`、`delta`
  - 两专家已停标志:`e5_stopped` / `e10_stopped`
  - 当前已完成 `epoch`
- 启用 `--resume` 且快照存在时:`load_resume_ckpt` 恢复上述全部状态,训练循环从 `epoch+1` 继续(**优化器状态也恢复,学习率/动量不丢失**)。
- 加载异常(快照损坏等)时 `try/except` 回退为从头训练并打印提示。
- 仅双专家模式支持;单专家传 `--resume` 会打印"暂不支持"并从头训练。

**关键坑(已踩已过)**:PyTorch ≥ 2.6 将 `torch.load` 默认 `weights_only` 改为 `True`,而续训快照含 numpy 数组(`best_score`/`score_min`),直接加载会抛 `UnpicklingError`。已在 `load_resume_ckpt` 显式传 `weights_only=False`(快照为本地可信产物)。

**自测**:
- `py_compile` 通过;
- 含 numpy 的续训快照 dict 序列化 round-trip 测试通过;
- 真实 `save/load_resume_ckpt` 方法集成测试通过:模型权重、双优化器状态、早停器 `best_score`(numpy)、`e5/e10_stopped` 标志、返回 `start_epoch=epoch+1` 全部正确恢复。

**用法**:用与首次完全相同的命令重跑,加 `--resume` 即从最近完成的 epoch 继续;同一 run 前缀下的 `_resume.pt` 随每个 epoch 覆盖更新。

#### 训练改进 #2:评测日志格式调整(e5/e10/fused 字段命名,已实施)

**背景**:原版代码的 `valid_epoch` 调用三次 `iteration(expert_to_eval='e5'/'e10'/'fused')`,每次都会通过 `get_full_sort_score` 打印一行指标,因此日志有三行输出(e5/e10/fused)。本地修改调整了日志格式(增加字段命名如 HIT@5_e5),方便解析。

**改动**(2026-08-24):
- `trainers.py` `valid_epoch`:每个 epoch 打印一行规整汇总,含 **e5 / e10 / fused 三组各 HIT@5、NDCG@5、HIT@20、NDCG@20**(共 12 个值)。
- `main.py` `--do_eval` 分支与训练结束 test 评测分支:同样分别评测 e5/e10/fused 并打印三行(含 NDCG@20)。
- 新日志格式示例:`{'Epoch': 320, 'HIT@5_e5': '0.0693', 'NDCG@5_e5': '0.0471', 'HIT@20_e5': '0.1397', 'NDCG@20_e5': '0.0671', 'HIT@5_e10': '0.0715', 'NDCG@5_e10': '0.0489', 'HIT@20_e10': '0.1411', 'NDCG@20_e10': '0.0686', 'HIT@5_fused': '0.0704', 'NDCG@5_fused': '0.0480', 'HIT@20_fused': '0.1404', 'NDCG@20_fused': '0.0678'}`。

**顺带修复的隐患**:原 `valid_epoch` 日志行用 `early_stopper_e5.best_score`(仅含 NDCG@20 一个元素)索引 `[1]`,专家一旦停止或首轮即会 `IndexError`。改为在每次评测时缓存完整 `self.last_scores_e5 / self.last_scores_e10`(4 元素 `[HIT@5, NDCG@5, HIT@20, NDCG@20]`),已停专家沿用缓存值打印。

**说明**:`py_compile` 通过;日志含 12 字段(e5/e10/fused 各 HIT@5、NDCG@5、HIT@20、NDCG@20)。

#### Bug #6:`--do_eval` 测试评测泄漏验证集物品(train_matrix 未切到 test_rating_matrix,已修复)
**现象**:`--do_eval` 的 Beauty fused NDCG@5≈0.0266,明显低于训练期验证(~0.049)与论文 Table 1 的 ADARec Beauty NDCG@5(0.0403)。
**根因**:
- `main.py` L127 默认 `args.train_matrix = valid_rating_matrix`;该矩阵由 `generate_rating_matrix_valid`(`utils.py` L181)生成,只含 `item_list[:-2]`,即**仅训练物品**(排除最后 2 个:验证标签 + 测试标签)。
- 训练结束评测分支(L230)会显式 `trainer.args.train_matrix = test_rating_matrix`(`generate_rating_matrix_test` 含 `item_list[:-1]` = 训练+验证物品);`--do_eval` 分支(L187-203)**没有**这步。
- `predict_full` 用 `rating_pred[batch_train.nonzero()] = 0`(`trainers.py` L518/966)把矩阵中物品清零、踢出排序池。do_eval 用 valid_rating_matrix → 测试目标 `i_n` 在「所有-训练」里排序,该序列的**验证物品 `i_{n-1}` 仍留在池内**与之竞争 → 测试 NDCG 被人为压低。
**修复**:`--do_eval` 分支加载权重后加 `trainer.args.train_matrix = test_rating_matrix`,使测试目标只在「未见物品」(所有-训练-验证)中排序,与训练结束评测、SASRec 标准协议一致。`py_compile` 通过。
**含义**:do_eval 的 0.0266 是「验证集泄漏 + 协议错误」导致的低估,不能直接与论文对比。修复后重跑 `--do_eval`(切 test_rating_matrix)得到 Beauty fused NDCG@5=0.0392,与论文 Table 1 的 ADARec Beauty NDCG@5=0.0403 基本吻合(差 ~0.001),说明本代码复现与论文一致(详见 §C.2)。

---

## 文献检查

### C. 实验证据(主张-实验-边界)

#### C.1 研究问题的实验映射

> 论文 Experiments 章节开篇明确提出 5 个研究问题（Research Questions），实验设计全部围绕回答这 5 问展开。完整问句（英文原文 + 中文翻译）如下：

**论文原文（Experiments, P5）**：

> *We conduct extensive experiments to evaluate our proposed ADARec framework, aiming to answer the following research questions:*

| 编号 | 英文原文（完整问句） | 中文翻译 |
| --- | --- | --- |
| **RQ1** | Does ADARec outperform state-of-the-art SR models and can it effectively enhance the performance of various intent-based backbone models? | ADARec 是否优于 SOTA 序列推荐模型，且能否有效提升各类「基于意图」的骨干模型的性能？ |
| **RQ2** | How effective are the key components of ADARec: the ADC, HDA, and HP-MoE? | ADARec 的三个关键组件（ADC、HDA、HP-MoE）各自有多有效？ |
| **RQ3** | How computationally efficient is ADARec? | ADARec 的计算效率如何？ |
| **RQ4** | How sensitive is ADARec's performance to its main hyperparameters? | ADARec 的性能对其主要超参数有多敏感？ |
| **RQ5** | Can ADARec effectively construct hierarchical user intents from sparse data? | ADARec 能否从稀疏数据中有效构建层次化的用户意图？ |

> 每个 RQ 对应一组实验/图表：RQ1→Table 1（主结果）+ Table 3（骨干组合）；RQ2→Table 4（消融）；RQ3→Table 5（效率）；RQ4→Figure 4（λ 敏感性）+ Figure 5（专家数）；RQ5→Table 2/3（稀疏序列）+ Figure 6（t-SNE）。

| 研究问题                  | 对应实验                                   | 数据来源 |
| --------------------- | -------------------------------------- | ---- |
| RQ1:ADARec 是否优于 SOTA? | Table 1(全序列)+ Table 3(骨干组合)            | P5-6 |
| RQ2:关键组件的贡献?          | Table 4(消融实验)                          | P6-7 |
| RQ3:计算效率如何?           | Table 5(训练/推理时间)                       | P7   |
| RQ4:超参数敏感性?           | Figure 4($\lambda$ 分析)+ Figure 5(专家数量) | P7   |
| RQ5:能否从稀疏数据构建层次化意图?   | Table 2(稀疏序列 ≤5)+ Figure 6(t-SNE)      | P6-7 |

#### C.2 RQ1 - 主结果(Table 1,Page 5-6)

**实验条件**:Beauty / Sports / Toys / Yelp(遵循 ELCRec pipeline);指标:Hit Rate @5/20(HR@K)、NDCG @5/20(N@K);候选集 = 100 负样本 + 1 正样本;11 个基线。

| 方法         | HR@5          | N@5           | HR@20         | N@20          |
| ----------- | ------------- | ------------- | ------------- | ------------- |
| **ADARec** | **最优**(全套数据集) | **最优**(全套数据集) | **最优**(全套数据集) | **最优**(全套数据集) |

> 下表 Beauty 数据来自论文 Table 1(ADARec 一行即 Ours)。「模型1」为本地完整模型复现,使用**代码默认参数**(hidden_size=256、num_hidden_layers=1、num_intent_clusters=256、deno_weight=1e-4、expl_weight=1e-4、temperature=1.0、weight_decay=0),尚未按论文参数重训。

**Beauty 主结果对照(论文 Table 1)**:

| 方法             | HR@5       | N@5        | HR@20      | N@20       |
| -------------- | ---------- | ---------- | ---------- | ---------- |
| Caser          | 0.0251     | 0.0145     | 0.0643     | 0.0298     |
| SASRec         | 0.0374     | 0.0241     | 0.0901     | 0.0387     |
| ELCRec         | 0.0529     | 0.0355     | 0.1079     | 0.0509     |
| **ADARec(论文)** | **0.0600** | **0.0403** | **0.1187** | **0.0568** |
| 模型1(代码默认参数)    | 0.0587     | 0.0392     | 0.1169     | 0.0557     |

> 模型1(代码默认参数)与论文 ADARec 在四个指标上均仅差 ~0.001-0.002,属正常随机种子 / 早停时机波动,可认为基本一致。**注意:模型1 尚未按论文参数重训**(论文值见 §C.5:deno_weight=0.1、expl_weight=0.01、temperature=0.07、weight_decay=1e-4、diffusion_t_max=10),需用模型6 的论文参数命令重新训练后再作最终对齐。

**Sports 主结果对照(论文 Table 1)**:

| 方法             | HR@5       | N@5        | HR@20      | N@20       |
| -------------- | ---------- | ---------- | ---------- | ---------- |
| Caser          | 0.0218     | 0.0134     | 0.0525     | 0.0243     |
| SASRec         | 0.0248     | 0.0156     | 0.0582     | 0.0269     |
| ELCRec         | 0.0289     | 0.0184     | 0.0647     | 0.0304     |
| **ADARec(论文)** | **0.0341** | **0.0228** | **0.0745** | **0.0341** |
| 模型1(代码默认参数)    | 0.0428     | 0.0282     | 0.0924     | 0.0420     |

> 模型1 Sports 复现结果(epoch 180 最佳 fused)反而高于论文 ~0.008,可能原因:数据集预处理差异/随机种子/早停时机。论文 Sports 值来自 Table 1,与 Beauty/Yelp/ml-1m 同表。**注意:此结果同样是代码默认参数下所得,尚未按论文参数重训。**

**本地训练最佳结果汇总(均在代码默认参数下所得,未按论文参数重训)**:

| 数据集    | 最佳 epoch | HIT@5_fused | NDCG@5_fused | HIT@20_fused | NDCG@20_fused | 论文 NDCG@5 |
| ------ | -------- | ----------- | ------------ | ------------ | ------------- | --------- |
| Beauty | 138      | 0.0734      | 0.0507       | 0.1403       | 0.0696        | 0.0403    |
| Sports | 180      | 0.0428      | 0.0282       | 0.0924       | 0.0420        | 0.0228    |

> 表中 Beauty/Sports 均对应**模型1**(完整模型,代码默认参数)。按论文参数需用模型6 命令重新训练。

#### C.3 RQ2 - 消融实验(Table 4)

**结论**(论文观点):HDA(分层扩散增强)被认为是基础组件,移除后性能下降明显;ADC 与 HP-MoE 同样重要。

**Beauty 消融(论文 Table 4,HR@20 / NDCG@20)**:

| 变体             | HR@20  | 下降   | NDCG@20 | 下降    |
| -------------- | ------ | ---- | ------- | ----- |
| ADARec(完整)     | 0.1187 | -    | 0.0568  | -     |
| w/o ADC        | 0.1141 | 3.9% | 0.0545  | 4.1%  |
| w/o HDA        | 0.1105 | 6.9% | 0.0528  | 7.0%  |
| w/o HP-MoE     | 0.1075 | 9.5% | 0.0505  | 11.1% |
| w/o C-A Router | 0.1172 | 1.3% | 0.0561  | 1.2%  |
| w/o Lcont      | 0.1158 | 2.4% | 0.0553  | 2.6%  |

> 注:论文正文称「HDA 移除影响最大」,但 Table 4 中 Beauty 的绝对 HR@20 下降以 w/o HP-MoE(9.5%)最大、w/o HDA(6.9%)次之。两者口径略有出入,复现时可重点关注。

**本地去 ADC 消融复现(2026-08-28,模型4)**

本仓库新增 `--ablate_adc` 开关实现「去 ADC」消融:开启时 `levels_e1=[5]`、`levels_e2=[10]`(单一固定深度,替代自适应区间 [4,5]/[9,10]);`models.py` forward 跳过 `controller()` 调用直接固定深度加噪,返回 `(raw, adaptive, None)`(probs=None 使探索损失 L_expl 归零、controller 完全退出训练)。

> 说明:早期「模型3」用 `probs.data.fill_(0.5)` 是**伪消融**——保留 grad_fn 仍回传梯度给 controller,且等权混合两个相邻深度 ≈ 自适应,故指标与完整模型几乎一致。真正的去 ADC 必须固定单一深度并让 controller 完全退出。

**Beauty 去 ADC 消融结果(fused 最终测试集,均在代码默认参数下所得)**:

| 模型                   | HR@5   | NDCG@5 | HR@20  | NDCG@20 |
| -------------------- | ------ | ------ | ------ | ------- |
| 完整模型(模型1,代码默认参数)     | 0.0587 | 0.0392 | 0.1169 | 0.0557  |
| 去 ADC 消融(模型4,代码默认参数) | 0.0594 | 0.0401 | 0.1179 | 0.0566  |

早停:e5 于 epoch 197(best NDCG@20=0.0676)、e10 于 epoch 233(best NDCG@20=0.0690),训练总时长 3h36m。

**关键发现:本地去 ADC 消融未复现论文的 ADC 贡献。** 论文 Table 4 宣称 w/o ADC 下降 3.9%/4.1%(0.1187→0.1141 / 0.0568→0.0545),但本地消融不但未降,反而微升 ~0.001(四指标均略高于完整模型)。

**归因(实现-论文语义偏差导致的消融不敏感)**:

1. 本地 ADC 为「相邻双深度软混合」(`num_levels=2`、`[4,5]`/`[9,10]`),自适应空间极小——softmax 在两个几乎相同的噪声水平间混合,差异微乎其微;
2. 去 ADC 后 `probs=None` → L_expl 归零、controller 退出,反而少了一个干扰源;
3. 固定深度 [5]/[10] 恰落在原区间合理位置,保留了浅/深双专家结构;
4. 早停更「干净」:消融 233 epoch 即收敛,完整模型 324 epoch(噪声 NDCG@20 反复清零计数器)。

> 结论:本地 ADC 对最终指标的贡献 ≈0(甚至略负),与论文 -3.9%/-4.1% 明显不符。根因是本地实现远弱于论文「0~k 连续深度 Gumbel-Softmax 选择」。这是「实现简化导致消融不敏感」的典型案例,本身值得汇报。

#### C.4 RQ3 - 计算效率(Table 5)

ADARec 的训练/推理时间与 ELCRec 相比,额外开销主要来自双专家的前向传播(约增加 80% 训练时间,但推理时间仅增加约 50%,因为推理时仅用 raw 分支)。

#### C.5 RQ4 - 超参数设置与敏感性(Implementation Details,P5-7)

论文 Implementation Details 给出的完整超参数设置(原文 P5):

| 参数                    | 论文设定                                  |
| --------------------- | ------------------------------------- |
| 优化器                   | Adam,weight decay = 1×10⁻⁴            |
| 最大序列长度                | 50                                    |
| Gumbel-Softmax 温度 τgs | 0.5                                   |
| InfoNCE 温度 τ          | 0.07                                  |
| 最大扩散深度 Tmax           | 在 {4, 6, 8, 10, 12} 中网格搜索             |
| 去噪损失权重 λdeno          | 在 {0.01, 0.03, 0.05, 0.1, 0.15} 中搜索   |
| 对比损失权重 λcont          | 在 {0.01, 0.03, 0.05, 0.1, 0.15} 中搜索   |
| 探索损失权重 λexpl          | 在 {0.001, 0.003, 0.01, 0.03, 0.1} 中搜索 |
| 硬件                    | 20GB NVIDIA RTX 3090 GPU              |

**RQ4 敏感性结论(论文 P6)**:模型在 λexpl = 0.01、λdeno = 0.1 时通常表现最优;这些辅助权重在最优点附近的性能曲线相对平缓,说明模型对权重的小幅偏差鲁棒(稳定易用)。

**本地训练实际使用对照(Beauty 模型6 命令,显式传参覆盖默认值)**:

| 论文参数 | 论文设定 | 本地训练实际使用 |
| ---- | ---- | ---- |
| 优化器 | Adam | Adam(beta1=0.9, beta2=0.999) ✅ |
| 权重衰减 | 1×10⁻⁴ | 0.0001(命令行 `--weight_decay 0.0001`) ✅ |
| 最大序列长度 | 50 | 50 ✅ |
| Gumbel-Softmax 温度 τgs | 0.5 | 未实现(用普通 softmax,无此参数) ✗ |
| InfoNCE 温度 τ | 0.07 | 0.07(命令行 `--temperature 0.07`) ✅ |
| 最大扩散深度 Tmax | {4, 6, 8, 10, 12} | 10(命令行 `--diffusion_t_max 10`,在集内) ✅ |
| λdeno | 0.1(最优) | 0.1(命令行 `--deno_weight 0.1`) ✅ |
| λcont | {0.01~0.15} | cf_weight 0.1 ✅ |
| λexpl | 0.01(最优) | 0.01(命令行 `--expl_weight 0.01`) ✅ |

> 说明:训练时通过命令行显式传参,将 weight_decay、InfoNCE τ、diffusion_t_max、λdeno、λexpl 均覆盖为论文值。唯一无法对齐的是 Gumbel-Softmax τgs=0.5——代码中 ADC 用普通 softmax、未实现 Gumbel-Softmax,属代码语义层面缺失而非超参问题。另注意:双专家模式下实际加噪深度仍为硬编码 levels_e1=[4,5]、levels_e2=[9,10],diffusion_t_max 只控制噪声调度 β 的最大步数。

**本地硬件环境**:NVIDIA GeForce RTX 5060 Laptop GPU(8 GB 显存)。论文为 20GB NVIDIA RTX 3090。

**本地模型命名与参数对照总表**:

| 模型编号 | 数据集 | 用途 | 参数设置 | 最终 NDCG@5 | 状态 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 模型1 | Beauty / Sports | 完整模型主结果 | 代码默认参数 | 0.0392 / 0.0224 | ✅ 与论文一致 |
| 模型2 | Sports / Yelp | 稀疏(`--shorten_seq_to 5`) | 代码默认参数 | 0.0204 / 0.0123 | ✅ 与论文一致 |
| 模型3 | Beauty | 伪消融(probs.fill_(0.5)) | 代码默认参数 | 0.0398 | ⚠️ 已废弃(伪消融) |
| 模型4 | Beauty | 去 ADC(`--ablate_adc`) | 代码默认参数 | 0.0401 | ⚠️ 未复现下降 |
| 模型5 | Beauty | 去 HDA(`--ablate_hda`) | 代码默认参数 | — | ⚠️ 训练至 epoch 37 中断 |
| 模型6 | Beauty | 完整(论文参数) | deno=0.1/expl=0.01/τ=0.07 | **0.0151** | ❌ 崩溃(论文参数) |
| Toys 模型1/2 | Toys | 完整(论文参数 τ=0.07) | deno=0.1/expl=0.01/τ=0.07 | 0.0052/0.0049 | ❌ 崩溃(论文参数) |
| Toys 模型3 | Toys | 完整(论文参数 τ=0.5) | deno=0.1/expl=0.01/τ=0.5 | 0.0182 | ❌ 部分崩溃 |
| Toys 模型4 | Toys | 完整(代码默认参数) | 代码默认参数 | 0.0474 | ✅ 与论文完全一致 |

> **关键结论(2026-08-29 实测,替代此前"模型1~4 需按论文参数重训"的说法)**:
> - **代码默认参数**即可完美复现论文(Beauty 0.0392、Sports 0.0224、Toys 0.0474 vs 论文 0.0403/0.0228/0.0474)。
> - **论文参数直接套到代码上反而崩溃**(Beauty 模型6 0.0392→0.0151、Toys 模型1/2 0.0474→0.0052)。根因是代码对三个辅助损失的实现语义与论文不同,套论文超参会放大错误信号——详见 §C.8。
> - 因此本地结果以**代码默认参数**为有效基准,不应再"按论文参数重训"。

#### C.6 RQ5 - 稀疏数据实验(Table 2,序列 ≤5)

| 数据集      | 稀疏场景提升(对比全序列基线) | Yelp 例外原因(仅 +3%)         |
| -------- | --------------- | ------------------------ |
| Sports   | 显著提升            | -                        |
| Beauty   | 显著提升            | -                        |
| Toys     | 显著提升            | -                        |
| **Yelp** | **仅 +3%**(微弱)   | Yelp 本身意图层次结构不显著,稀疏性收益有限 |

> ⚠️ Yelp 例外说明:论文中 Yelp 数据集的用户行为意图层次不如其他三个电商/生活场景明显,ADARec 的"意图层次重建"机制在 Yelp 上收益有限。

**本地稀疏复现(模型2,`--shorten_seq_to 5`,最终测试 fused,代码默认参数)**:

| 数据集    | HR@5   | NDCG@5 | HR@20  | NDCG@20 | 论文 HR@5 | 论文 NDCG@5 |
| ------ | ------ | ------ | ------ | ------- | ------- | --------- |
| Sports | 0.0308 | 0.0201 | 0.0676 | 0.0306  | 0.0305  | 0.0199    |
| Yelp   | 0.0223 | 0.0137 | 0.0611 | 0.0246  | 0.0200  | 0.0124    |

早停:Sports 模型2 于 epoch 163、Yelp 模型2 于 epoch 117。

> 本地稀疏复现与论文 Table 2 基本一致(Sports 差 ~0.0003,Yelp 略高 ~0.001-0.002)。Yelp 虽绝对指标最低,但本地同样未出现「全序列下 Yelp 指标骤降」的反常,说明稀疏序列实验复现成功。

#### C.7 证据边界总结

| 维度   | 内容                         | 边界                |
| ---- | -------------------------- | ----------------- |
| 实验环境 | 4 个公开数据集,100 负采样,5 次随机种子   | 仅在此设置下有效          |
| 基线覆盖 | 11 个基线(SOTA 比较全面)          | 无协同过滤/图神经网络的最新方法  |
| 消融实验 | ADC / HDA / HP-MoE 三组件独立消融 | 未做两两组合消融          |
| 稀疏实验 | `--shorten_seq_to 5` 截断    | 未测试 3/4/6 等其他稀疏粒度 |
| 效率实验 | Table 5 时间对比               | 未报告 GPU 显存占用      |

#### C.8 本地全量训练结果汇总与归因(2026-08-29)

> 汇总本地 14 个训练 run 的最终测试集 fused 指标(每 run 的第一行日志即训练命令)。全部为双专家模式 + `--resume`,epochs=400。

**最终测试集 fused 指标汇总(HIT@5 / NDCG@5 / HIT@20 / NDCG@20)**:

| 数据集 | 模型 | 用途 / 关键参数 | 早停 epoch | HIT@5 | NDCG@5 | HIT@20 | NDCG@20 | 论文 NDCG@5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Beauty | 1 | 完整模型(代码默认参数) | 324 | 0.0587 | 0.0392 | 0.1169 | 0.0557 | 0.0403 |
| Beauty | 3 | 伪消融(probs.fill_(0.5)) | 262 | 0.0588 | 0.0398 | 0.1177 | 0.0565 | — |
| Beauty | 4 | 去 ADC(`--ablate_adc`) | 233 | 0.0594 | 0.0401 | 0.1179 | 0.0566 | — |
| Beauty | 5 | 去 HDA(`--ablate_hda`) | **中断(37)** | 0.0024 | 0.0012 | 0.0085 | 0.0029 | — |
| Beauty | 6 | 完整模型(**论文参数** deno=0.1/expl=0.01/τ=0.07) | 102 | 0.0235 | 0.0151 | 0.0486 | 0.0221 | 0.0403 |
| Sports | 1 | 完整模型(代码默认参数) | 239 | 0.0338 | 0.0224 | 0.0749 | 0.0339 | 0.0228 |
| Sports | 2 | 稀疏(`--max_seq_length 5`) | 163 | 0.0308 | 0.0201 | 0.0676 | 0.0306 | 0.0199 |
| Sports | 3 | 稀疏(`--shorten_seq_to 5`) | 179 | 0.0306 | 0.0204 | 0.0681 | 0.0309 | 0.0199 |
| Toys | 1 | 完整(**论文参数** deno=0.1/expl=0.01/τ=0.07/Tmax=12) | 179 | 0.0070 | 0.0052 | 0.0188 | 0.0084 | 0.0474 |
| Toys | 2 | 完整(**论文参数** deno=0.1/expl=0.01/τ=0.07) | 184 | 0.0069 | 0.0049 | 0.0177 | 0.0078 | 0.0474 |
| Toys | 3 | 完整(**论文参数** deno=0.1/expl=0.01/τ=0.5) | 280 | 0.0281 | 0.0182 | 0.0634 | 0.0281 | 0.0474 |
| Toys | 4 | 完整模型(代码默认参数) | 358 | 0.0687 | 0.0474 | 0.1303 | 0.0649 | 0.0474 |
| Yelp | 2 | 稀疏(`--max_seq_length 5`) | 117 | 0.0223 | 0.0137 | 0.0611 | 0.0246 | 0.0124 |
| Yelp | 3 | 稀疏(`--shorten_seq_to 5`) | 103 | 0.0195 | 0.0123 | 0.0552 | 0.0222 | 0.0124 |

> 注:Yelp 缺完整模型(模型1)。Beauty 模型5 训练至 epoch 37 被中断,test 结果(0.0012)为未训练完成权重,不具参考意义。

**四大核心发现**:

1. **默认参数完美复现论文(3 个数据集)**——Beauty 0.0392 vs 论文 0.0403、Sports 0.0224 vs 0.0228、Toys 0.0474 vs 0.0474(**完全一致**)。证明代码主干(SASRec 骨干 + 双专家集成)正确,评测协议修复后无泄漏。

2. **论文参数套到代码上反而崩溃(Beauty 模型6、Toys 模型1/2)**——Beauty 从 0.0392 暴跌到 0.0151(-62%),Toys 从 0.0474 暴跌到 0.0052(-89%)。根因:代码对三个辅助损失的实现语义与论文不同,直接套论文超参会放大错误信号:
   - `--deno_weight 0.1`(默认 1e-4 的 1000 倍):代码 $L_{deno}$ 是「表示重构损失」MSE(adaptive, raw.detach()),目标是让加噪后编码逼近原始编码,权重越大越**抵消扩散增强效果**;论文 $L_{deno}$ 是「噪声预测损失」‖ε_θ−ε‖²,语义完全不同。
   - `--expl_weight 0.01`(默认 1e-4 的 100 倍):代码 $L_{expl}$ 是负熵 -mean(H(p)),权重过大强迫 probs 均匀化,干扰主任务。
   - `--temperature 0.07`(默认 1.0):InfoNCE 温度越小,相似度除以 τ 后数值越大,对比损失越尖锐、训练越不稳定(Toys 模型3 用 τ=0.5 已部分恢复,说明 τ 是主要崩溃因素之一)。

3. **稀疏序列复现成功(Sports/Yelp)**——`--shorten_seq_to 5`(论文定义)与 `--max_seq_length 5`(仅截断张量)结果接近,但严格应以 `--shorten_seq_to 5` 为准:Sports 0.0204 vs 论文 0.0199、Yelp 0.0123 vs 论文 0.0124,基本吻合。两种写法在本数据集差距小,是因为 Sports/Yelp 用户历史本身较短,截断与张量限长效果趋同。

4. **ADC/HDA 消融做不出论文下降幅度**——去 ADC(模型4)NDCG@5 反而微升 0.001,伪消融(模型3)几乎不变。印证 §2.1.5 / §D.1.1 结论:代码把 ADC/HDA/C-A Router 三组件压进同一个 probs,自适应空间被 num_levels=2 + 相邻深度 + expl_weight=1e-4 三重压缩,消融自然不敏感。

---

#### C.9 ADARec(2) 对齐版代码三组实验(2026-09-06)

> 承接 §D 对原版「4 处语义差异」的批判:本地另建了 **ADARec(2)** 目录(`F:\recsys-research-training-zhangxinyu\experiment\ADARec(2)\src\`),把原版与论文不一致的 4 处全部补全对齐,再在 Sports_and_Outdoors 上跑了三组实验,检验「语义对齐后能否用论文超参复现、能否做出论文的消融下降」。

**对齐版补全的 4 处语义(相对原版 ADARec)**:

| # | 组件 | 原版实现 | 对齐版实现(models.py / modules.py) |
| --- | --- | --- | --- |
| 1 | ADC | 普通 `F.softmax`,无噪声无温度 | `gumbel_softmax`(modules.py L22-36):straight-through,前向 argmax 得 one-hot,反向沿 softmax 概率回传,τ_gs=0.5 |
| 2 | HDA | 仅前向加噪,加噪后直接再编码,无去噪网络 | `DenoisingNetwork`(modules.py L336-365):MLP 预测噪声 ε_θ(E_k, k),按 Eq.5 重建 ê_k=(E_k−√(1−ᾱ_k)ε_θ)/√ᾱ_k |
| 3 | HP-MoE | 双独立专家 + 固定 0.5/0.5 等权平均 | `ContentAwareRouter`(g_k,Eq.7)+ `gate_layer`(gate_u,Eq.10),元素级门控融合 h_u=gate_u⊙z_fin+(1−gate_u)⊙z_coar |
| 4 | L_deno | 表示重构 MSE(adaptive, raw.detach()) | 噪声预测 MSE ‖ε_θ·mask − ε·mask‖²(trainers.py L688-690),与论文 Eq.5 对齐 |

**三组实验命令与最终测试集 fused 指标(H@5 / N@5 / H@20 / N@20)**:

| Run | model_idx | 关键配置 | 早停 epoch | H@5 | N@5 | H@20 | N@20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 对齐版完整 + **论文超参**(deno=0.1/expl=0.01/τ=0.07/wd=1e-4/Tmax=10) | 132 | 0.0061 | 0.0043 | 0.0202 | 0.0080 |
| 2 | 2 | 对齐版完整 + **默认参数**(deno=1e-4/expl=1e-4/τ=1.0/wd=0/Tmax=50) | 233→234 | 0.0332 | 0.0221 | 0.0734 | 0.0334 |
| 3 | 3 | 对齐版 + 默认参数 + **`--ablate_hda`** | 277 | 0.0334 | 0.0222 | 0.0743 | 0.0337 |

> 论文 Table 1 Sports ADARec:H@5=0.0341 / N@5=0.0228 / H@20=0.0745 / N@20=0.0341。
> 论文 Table 4 消融 Sports:完整 0.0745/0.0341,w/o HDA 0.0681/0.0310(Δ=−0.0064/−0.0031),w/o HP-MoE 0.0645/0.0283(Δ=−0.0100/−0.0058),w/o ADC 0.0715/0.0328(Δ=−0.0030/−0.0013)。

**三大发现**:

1. **语义对齐后、默认参数即可复现论文(Run 2)**——四指标与论文差 ≤0.001(H@5 0.0332 vs 0.0341、N@5 0.0221 vs 0.0228、H@20 0.0734 vs 0.0745、N@20 0.0334 vs 0.0341)。与 §C.8 原版默认参数(Sports 0.0338/0.0224/0.0749/0.0339)几乎一致,说明**补全的 Gumbel-Softmax/去噪网络/门控路由在默认超参下并未带来可观测增益**——原版简化实现与论文对齐实现在最终指标上趋同。

2. **论文超参套到对齐版代码上仍然崩溃(Run 1)**——N@5 从 0.0221 暴跌到 0.0043(−81%),H@20 0.0734→0.0202。这**推翻了 §C.8 的旧归因**(旧结论把崩溃归咎于「L_deno 是表示重构损失而非噪声预测」):现在 L_deno 已对齐为噪声预测 MSE,崩溃依旧,说明**根因不是损失语义,而是论文超参本身过激**——`deno_weight=0.1`(默认的 1000×)、`expl_weight=0.01`(默认的 100×)、`temperature=0.07`(InfoNCE 过尖锐)叠加,在去噪网络尚未训好的早期即主导梯度、干扰主任务。论文 Table 4 的「最优 λexpl=0.01、λdeno=0.1」在独立复现中无法落地。

3. **HDA 消融做不出论文下降、方向甚至相反(Run 3 vs Run 2)**——去 HDA 后 H@20 0.0734→0.0743、N@20 0.0334→0.0337,**微升而非下降**,与论文 w/o HDA 的 −0.0064/−0.0031 完全相反。这延续并强化了 §D.1.1 的判断:即便语义对齐,扩散增强(HDA+去噪网络)在这套训练协议下对最终排序指标贡献 ≈0,论文消融表的下降幅度无法通过官方公开代码独立复现。

> **小结**:三组实验共同指向一个结论——论文报告的核心增益(扩散增强 + 自适应深度 + 门控 MoE)在复现中高度依赖论文那套「激进的辅助损失权重」,而该套权重在独立复现中反而导致崩溃;退到温和默认参数时,模型退化为「SASRec 骨干 + 双专家集成」即可达到论文主结果水平。换言之,**可复现的增益主要来自骨干与双专家集成,而非扩散/ADC/门控三件套**。

---

### D. 论文主张 vs 代码实现:批判性对照

> 本节对论文的 6 个核心主张逐一对照代码实现,给出"论文声称"、"代码实际"和"一致性判断"。

#### D.1 六主张逐一对照

| #   | 主张描述                                                    | 代码实际                                            | 一致性      | 说明                                    |
| --- | ------------------------------------------------------- | ----------------------------------------------- | -------- | ------------------------------------- |
| 1   | ADC 用 Gumbel-Softmax 实现自适应深度选择                          | 普通 softmax,无 Gumbel 噪声,温度机制缺失                   | **不一致**  | 论文声称 Gumbel-Softmax 采样,代码为确定性 softmax |
| 2   | HDA 通过反向去噪网络解析意图层次                                      | 仅前向加噪,无独立去噪网络;加噪嵌入直接经 Transformer 编码            | **不一致**  | 论文暗示迭代反向去噪,代码是"加噪+再编码"简化版             |
| 3   | HP-MoE 通过门控/路由动态分配专家                                    | 两专家完全独立,无跨专家门控;推理融合为等权平均                        | **不一致**  | 代码更接近"双模型集成",非"混合专家路由"                |
| 4   | $L_{deno}$ 是预测噪声 MSE $\|\epsilon_\theta - \epsilon\|^2$ | $L_{deno}$ = MSE(adaptive_output, raw.detach()) | **不一致**  | 代码是"表示重构损失",非"噪声预测损失"                 |
| 5   | 扩散步数 $T_u$ 自适应选择(0 到 $k$ 连续)                            | 仅在 `[4,5]`(e5)或 `[9,10]`(e10)两个离散深度中加权混合        | **部分一致** | 自适应体现在"两深度间的加权比例",而非"0到k连续选择"         |
| 6   | 专家融合是可学习门控网络                                            | 无门控网络,融合权重固定为 0.5/0.5                           | **不一致**  | 代码是两独立模型的推理平均                         |

> **2026-09-06 更新**:上述 4 处「不一致」已在 ADARec(2) 对齐版代码中逐项补全(Gumbel-Softmax / 去噪网络 / 门控路由 / 噪声预测损失),但三组 Sports 实验表明:默认参数下对齐版与原版指标趋同、论文超参仍崩溃、HDA 消融仍做不出下降(详见 §C.9)。即「语义不一致」并非复现偏差的主因,论文的核心增益主要来自骨干 + 双专家集成。

#### D.1.1 补充辨析:代码把 ADC/HDA/C-A Router 三组件挂到同一个 probs 上(消融失效的根因)

论文中 **ADC(生成端)**、**C-A Router(解析端)**、**HDA(扩散增强)** 是三个职责不同、可独立消融的模块,但代码把它们耦合进同一个 `controller + probs`,且 `probs` 的唯一消费方就是 HDA 的加噪加权,形成**三层耦合**:

| | 论文 ADC | 论文 C-A Router | 论文 HDA | 代码实际 |
| --- | --- | --- | --- | --- |
| **职责** | 决定加噪深度 $T_u$(生成端) | 决定各层融合权重 $g_k$(解析端) | 用 $T_u$ 做加噪+去噪(增强) | 一个 probs 同时干三件事 |
| **归属** | HDA | HP-MoE | 独立模块 | 无归属(合并) |
| **输出** | 离散深度(整数) | 逐层门控(向量) | 多噪声水平表示 | `(B,2)` softmax |

代码里 `probs` 的双重身份:前向加权求和 = C-A Router 的活(Eq.7/8),反向负熵探索损失 = ADC 的活(保持选深度多样性)。而 **`probs` 的唯一消费方就是 HDA 的加噪加权**(forward 里 controller→probs→对每个 depth 加噪→sum→再编码),所以:

- **去 HDA 会连带废掉 ADC**:若 adaptive=raw 且不调用 controller,则 probs=None → 探索损失归零 → ADC 不再被训练;
- **去 ADC 会连带废掉 HDA 分层语义**:固定单一深度后 HDA 退化为固定单步加噪,不再产生意图层次;
- **去 C-A Router 也废掉 ADC/HDA**:三者都在同一 forward 链上,无法独立拆解。

再加上三重压缩——`num_levels=2`(深度只有相邻两档)、`expl_weight=1e-4`(探索信号趋近零)、融合固定 0.5/0.5(无门控)——导致 **ADC/HDA/C-A Router 三组件实际都被简化成"几乎不影响前向"的装饰**,真正起作用的只剩 HP-MoE 双专家集成(两套独立参数)+ SASRec 骨干。这解释了本地消融(去 ADC 模型4)未复现论文 -3.9% 下降、指标反微升 ~0.001 的根因。详见 §2.1.5。

#### D.2 算法级语义变化总结

**三个语义变化**(超出实现细节范畴):

1. **ADC 前向/后向合并**:论文"前向 argmax 选深度,后向概率回传"改为"所有深度软混合,probs 同时参与前向计算和反向损失"。

2. **HDA 去噪网络省略**:论文"前向加噪 → 反向去噪网络 → 去噪表示"改为"前向加噪 → Transformer 再编码"。

3. **HP-MoE 路由简化**:论文"动态路由分配专家"改为"两专家独立处理 → 平均"。

#### D.3 代码与论文一致的方面

| 方面      | 代码行为                                                 | 对应论文内容              |
| ------- | ---------------------------------------------------- | ------------------- |
| 骨干网络    | SASRec(Transformer 编码器)                              | Section 2,ELCRec 继承 |
| DDPM 调度 | 线性 $\beta$ 调度,`beta_start=1e-4, beta_end=0.02, T=50` | Section 3.2         |
| 双专家结构   | model_e5(depth=4,5) + model_e10(depth=9,10)          | Section 3.3         |
| 对比学习    | InstanceCL (NCE) + IntentCL (PCL) + 距离损失             | Section 3.4         |
| 稀疏性实验   | `--shorten_seq_to N` 截断历史序列                          | Table 2, Figure 6   |
| 评测指标    | HIT@5/20, NDCG@5/20,重排式评测                            | Section 4.1         |

## 附录:ADARec 代码实践速查

> 本节浓缩 ADARec 代码的实操知识(运行命令、超参差异、评测协议、已知 Bug 与修复状态),供快速参考。详细分析见上方各节。

### A.1 运行命令(典型)

**Sports 重跑示例**(OOM 修复后,conda python312):
```
D:\.conda\envs\python312\python.exe main.py \
  --data_name Sports_and_Outdoors \
  --model_idx 1 --gpu_id 0 \
  --output_dir output/Sports_and_Outdoors/ \
  --batch_size 256 \
  --contrast_type Hybrid \
  --num_intent_cluster 256 \
  --seq_representation_type mean \
  --intent_cf_weight 0.1 \
  --num_hidden_layers 2 \
  --enable_diffusion_aug \
  --hidden_size 128 \
  --attention_probs_dropout_prob 0.5 \
  --hidden_dropout_prob 0.5 \
  --enable_adaptive_diffusion \
  --dual_expert \
  --epochs 400
```

**Beauty 运行**(hidden_size=256, layers=1, trade_off=10, prototype=shift,其余同上):
```
--hidden_size 256 --num_hidden_layers 1
```

**断点续训**(先确保 `*_resume.pt` 存在且两专家未全停):
```
...同上...
  --resume
```

### A.2 超参对照(Beauty / Sports / Toys / Yelp)

| 参数              | Beauty / Yelp / ml-1m | Sports / Toys |
| --------------- | -------------------- | -------------- |
| `hidden_size`     | 256                  | 128            |
| `num_hidden_layers` | 1                   | 2              |
| `num_intent_clusters` | 256               | 512            |
| `trade_off`       | 10                   | -              |
| `prototype`       | shift                | -              |
| `patience`        | 40(均适用)             | 40             |
| `beta_start/end`  | 1e-4 / 0.02(均适用)     | 同             |
| `diffusion_t_max` | 50(均适用)             | 50             |

### A.3 评测协议(易出错)

| 场景              | train_matrix 正确设置             | 监控指标        | 说明                                  |
| --------------- | -------------------------- | ----------- | ----------------------------------- |
| 训练期验证          | `valid_rating_matrix`(训练物品,不含验证) | 各专家 **raw** NDCG@20 | 早停依据;原版与本地都打印 e5/e10/fused 三行指标 |
| 训练结束最终测试      | `test_rating_matrix`(训练+验证物品)     | **fused** HIT@5/NDCG@5 | main.py L230 自动设置                   |
| `--do_eval` 加载权重评测 | 必须手动 `trainer.args.train_matrix = test_rating_matrix` | **fused** 指标 | Bug #6:忘记此步导致验证物品泄漏到候选池,NDCG 被压低  |

> ⚠️ `--do_eval` 默认用 `valid_rating_matrix`(不符合论文测试协议),必须手动切换。

### A.4 已知 Bug 与修复状态

| #   | 描述                                                            | 文件:行号                | 状态                                                            |
| --- | ------------------------------------------------------------- | -------------------- | ------------------------------------------------------------- |
| #1  | `index` 未定义(datasets.py `_data_sample_rec_task`,NameError)    | `datasets.py` L124   | ✅ 已修复:改为 `self.test_neg_items[user_id]`                       |
| #2  | IntentCL 遍历 nn.Parameter 无 `.query()` 方法(AttributeError)      | `trainers.py` L674   | ⚠️ 未改:模型 training=False 时不跑扩散,强设 True 反而错;属论文↔代码语义偏差,非崩溃      |
| #3  | Softmax vs Gumbel-Softmax 差异(代码用 softmax,非论文 Gumbel)          | `models.py` ADC      | i️ 记录:不影响复现结果                                                 |
| #4  | OOM:numpy 密集 `toarray()` 累积 host RAM(epoch ~189)              | `trainers.py` L891   | ✅ 已修复:稀疏索引 `rating_pred[batch_train.nonzero()] = 0`           |
| #5  | `--do_eval` `torch.load` 缺 `_e5/_e10.pt` 后缀 FileNotFoundError | `main.py` L188       | ✅ 已修复:显式拼接后缀                                                  |
| #6  | `--do_eval` 未切 `train_matrix` 为 `test_rating_matrix`(验证物品泄漏)  | `main.py` do_eval 分支 | ✅ 已修复:加载权重后设 `trainer.args.train_matrix = test_rating_matrix` |
|     |                                                               |                      |                                                               |

### A.5 早停行为

- **监控指标**:各专家 raw NDCG@20(`np.array(scores_e5[-1:])` 取 4 元素列表的第 4 位),**不是** fused NDCG@5。
- **触发机制**:双独立 EarlyStopper(e5/e10 各一个),`compare(score)` 用 `score > best_score + delta`,delta=0,patience=40。**两专家均触发才结束训练**(`if e5_stopped and e10_stopped: return True`)。
- **与论文一致性**:与原版仓库(Cxx-0/ADARec)逐行一致,非偏差。
- **resume snapshot 全停**:若快照中 `e5_stopped=True` 且 `e10_stopped=True`,恢复后 `valid_epoch` 立刻 return True,`--resume` 即空跑。须删 `_resume.pt` + 去 `--resume` 才能真重训。

### A.6 断点续训机制

- 快照文件:`*_resume.pt`(`checkpoint_prefix + "_resume.pt"`)。
- 快照内容:model_e5/e10 state_dict、优化器 e5/e10 state_dict、两 EarlyStopper 的 `best_score`(numpy)、`e5/e10_stopped` 标志。
- `load_resume_ckpt` 用 `map_location=self.device` + `weights_only=False`(含 numpy 必崩于 PyTorch 2.6+ 默认 True)。
- main.py 中须去掉 `if args.dual_expert:` 守卫才会在单专家模式下也保存(当前代码守卫仍存在,单专家 run 不会自动生成 resume.pt)。

### A.7 环境注意

| 项         | 说明                                                                          |
| --------- | --------------------------------------------------------------------------- |
| Python 环境 | `D:\.conda\envs\python312\python.exe`(QClaw 自带 python 无 torch)              |
| PyTorch   | torch 2.11.0 + CUDA 12.8(`torch.__version__`)                               |
| 代码目录      | `F:\recsys-research-training-zhangxinyu\experiment\ADARec\src\`             |
| 仓库来源      | 原址 `Cxx-0/ADARec`(sha 90faad2),原 ZhaoChao52 已重定向                            |
| 日志文件      | `--output_dir` 下;`.log` stdout,`.err` stderr,进度 `.txt` append 模式(含多 run 残留) |
| 编码注意      | 日志/笔记文件 UTF-8;PowerShell 控制台 GBK,写文件用工具避免乱码                                 |
