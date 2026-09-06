# 名词解释笔记
---

## A. 推荐系统通用术语

### A.1 Top-N 推荐（Top-N Recommendation）

- **类型**：任务
- **相关方向**：序列推荐、隐式反馈学习
- **具体含义**：给定用户历史行为，从候选物品中生成一个有序推荐列表（前 N 项），评价指标通常为 HIT@K / NDCG@K。
- **与评分预测的区别**：Top-N 使用隐式反馈（点击/购买），目标是排序；评分预测使用显式评分，目标是数值精度。

### A.2 序列推荐（Sequential Recommendation）

- **类型**：任务 / 研究领域
- **相关方向**：自回归建模、Transformer、扩散模型
- **具体含义**：根据用户的行为时间序列，预测下一个交互物品。强调行为的顺序性和时序依赖。
- **代表模型**：SASRec（Transformer）、GRU4Rec（RNN）、BERT4Rec（Masked LM）。

### A.3 隐式反馈（Implicit Feedback）

- **类型**：数据
- **相关方向**：推荐损失函数、负采样
- **具体含义**：用户行为间接反映偏好，如点击、购买、播放、收藏等。特点是只有正反馈有标注，缺失交互 ≠ 明确负反馈。
- **常用处理**：将隐式交互转为 $\{0,1\}$ 二值矩阵，未交互物品通过负采样获得训练信号。

### A.4 负采样（Negative Sampling）

- **类型**：训练技术
- **相关方向**：BPR 损失、对比学习
- **具体含义**：在隐式反馈场景中，从大量未交互物品中采样一部分作为负例，解决"全量负例计算成本过高"的问题。
- **ADARec 中的应用**：`datasets.py` 中每个训练样本包含一个正样本 `target_pos` 和多个负样本 `target_neg`（数量由 batch 内其他用户物品充当）。

### A.5 早停（Early Stopping）

- **类型**：训练技术
- **相关方向**：泛化、过拟合
- **具体含义**：监控验证集指标，若连续若干 epoch 无改善则停止训练，防止过拟合。`patience`=容忍多少 epoch 无改善，`delta`=改善阈值。
- **ADARec 中的特殊性**：双专家各自有独立早停器；监控指标是 raw NDCG@20，不是 fused NDCG@5。

---

## B. ADARec 核心模块

### B.1 ADC（Adaptive Depth Controller，自适应深度控制器）

- **类型**：模型组件
- **相关方向**：扩散增强、MoE
- **具体含义**：根据输入序列 embedding，输出一个概率向量 $\pi = (\pi_1, \pi_2)$，决定用多少噪声强度（深度 4 或 5）增强该样本。
- **论文描述**：用 Gumbel-Softmax 做离散采样，$\pi$ 是 one-hot 近似。
- **代码实际**（Bug 记录）：`models.py` L308 用的是普通 `F.softmax`，无 Gumbel 噪声，无温度重参数化，本质是 softmax 软混合。
- **输出概率的三个去向**：
  1. **前向加权**：`adaptive = sum(probs * depth_output)` 作为增强表示
  2. **探索损失**：$-(\log \pi_1 + \log \pi_2)/2$ 负熵，鼓励均匀
  3. **CL 分支**：在 IntentCL 中丢弃，不参与计算

### B.2 HDA（Hybrid Diffusion Augmentation，混合扩散增强）

- **类型**：数据增强技术
- **相关方向**：DDPM、前向扩散
- **具体含义**：对 Transformer 中间层输出做前向扩散（加噪），再编码回模型。核心是闭式公式，不训练独立去噪网络。
- **论文描述**：Forward Diffusion → Reverse Denoise Network → Recovered Representation。
- **代码实际**（Bug 记录）：只有前向加噪，无独立去噪 MLP；加噪结果直接喂给 Transformer 再编码。
- **数学形式**：$E_t = \sqrt{\bar{\alpha}_t} \cdot E_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$

### B.3 HP-MoE（Hybrid-Pruning MoE，混合剪枝专家混合）

- **类型**：模型架构
- **相关方向**：MoE 路由
- **具体含义**：两个独立 SASRec 专家（depth=4,5 和 depth=9,10），根据序列自适应选择使用哪个。
- **论文描述**：可学习门控网络动态路由。
- **代码实际**（Bug 记录）：两个专家完全独立，无共享参数；融合用固定 0.5/0.5 等权平均，无门控网络。
- **与经典 MoE 的区别**：经典 MoE 共享主干 + 门控；ADARec 代码是两个独立模型，接近"双模型集成"。

### B.4 意图聚类（Intent Clustering）

- **类型**：表示学习 / 辅助任务
- **相关方向**：KMeans、对比学习、聚类假设
- **具体含义**：将用户序列表示聚类为 K 个意图簇，用簇中心作为意图级别的表示。对比学习损失（IntentCL）拉近序列与对应簇中心、推远其他簇中心。
- **K 设置**：Beauty/Yelp/ml-1m = 256；Sports/Toys = 512。
- **EMA 更新**：`new_center = 0.99 * old_center + 0.01 * batch_mean`

### B.5 骨干网络：SASRec

- **类型**：基础模型
- **相关方向**：Transformer、自回归序列建模
- **具体含义**：SASRec（Self-Attentive Sequential Recommendation）是基于 Transformer 的序列推荐模型，用 Masked Self-Attention 保证时序信息不泄漏。
- **ADARec 继承**：`modules.py` 的 `SelfAttention` / `Layer` / `Encoder` 即 SASRec 实现，ELCRec 继承自 SASRec。

---

## C. 扩散模型相关

### C.1 DDPM（Denoising Diffusion Probabilistic Model，去噪扩散概率模型）

- **类型**：生成模型框架
- **相关方向**：前向扩散、反向去噪、噪声调度
- **具体含义**：两阶段：①前向过程逐步加噪；②反向过程学习去噪分布。训练目标通常为噪声预测 MSE $\| \epsilon_\theta - \epsilon \|^2$。
- **ADARec 的简化**：只有前向加噪，没有独立去噪网络；损失是表示重构而非噪声预测。

### C.2 前向扩散（Forward Diffusion）

- **类型**：扩散过程（前半段）
- **相关方向**：噪声调度、闭式采样
- **具体含义**：在已知噪声调度 $\{\alpha_t\}$ 下，任意时刻 $t$ 的带噪表示可由初始表示 $E_0$ 闭式计算，无需迭代。
- **ADARec 形式**：$E_t = \sqrt{\bar{\alpha}_t} E_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$

### C.3 β 调度（Beta Schedule）

- **类型**：扩散超参数
- **相关方向**：DDPM 收敛速度
- **具体含义**：控制每步前向扩散保留原信号的比例。ADARec 用**线性调度**：`beta_start=1e-4`，`beta_end=0.02`，`T_max=50`。
- **与余弦调度的区别**：线性调度 $\beta_t$ 从小到大线性增长；余弦调度变化更平滑。ADARec 未用官方余弦调度。

### C.4 闭式公式（Closed-Form Formula）

- **类型**：数学技巧
- **相关方向**：前向扩散计算效率
- **具体含义**：可直接用代数式计算，无需迭代或采样。在扩散模型中特指"不需要逐步迭代，只需一步即可从 $E_0$ 算出 $E_t$"。
- **ADARec 中体现**：`forward_diffusion(E0, t, noise)` 单步完成加噪，无需 for 循环。

### C.5 Gumbel-Softmax

- **类型**：可微离散采样
- **相关方向**：重参数化、离散选择
- **具体含义**：用 Gumbel 噪声 + Softmax 构造可微的 one-hot 近似，支持离散决策的反向传播。温度 $\tau$ 控制软硬程度：$\tau \to 0$ 时趋向 one-hot。
- **论文描述**：ADC 控制器用 Gumbel-Softmax。
- **代码实际**：用的是普通 Softmax（`F.softmax(logits)`），无 Gumbel 噪声，无温度机制。

### C.6 嵌入坍塌（Embedding Collapse）

- **类型**：训练问题
- **相关方向**：扩散模型、自监督学习、表示学习
- **具体含义**：扩散模型训练中，所有样本的嵌入趋向同质化（坍塌到相似区域），导致区分性丧失。ADARec 的核心研究问题。
- **ADARec 的解决方案**：通过 IntentCL（PCL）保持意图级别的区分性，通过距离损失约束双专家表示接近。

---

## D. 对比学习相关

### D.1 InstanceCL（Instance-level Contrastive Learning）

- **类型**：对比学习范式
- **相关方向**：InfoNCE、正负样本构造
- **具体含义**：同一样本的两个视图（如原始序列与增强序列）为正例，batch 内不同样本为负例。温度 $\tau$ 控制负例区分度。
- **ADARec 中的实现**：`modules.py` L170-190 `NCELoss`，$\tau=0.07$。

### D.2 IntentCL（Intent-level Contrastive Learning）

- **类型**：对比学习范式
- **相关方向**：聚类、ProtoContras
- **具体含义**：序列表示拉近对应意图簇中心，推远其他簇中心。与 InstanceCL 的区别是"意图"级别而非"实例"级别。
- **ADARec 中的实现**：`modules.py` L192-215 `PCLoss`（ProtoContras 的简化版）。
- **⚠️ 代码 Bug**：遍历 `nn.Parameter` 无 `.query()` 方法（IntentCL 分支），导致 AttributeError；`training=False` 时此分支不运行，故不影响正常推理。

### D.3 InfoNCE

- **类型**：对比学习损失函数
- **相关方向**：互信息估计
- **具体含义**：$-\log \frac{\exp(s^+ / \tau)}{\sum \exp(s / \tau)}$，是噪声对比估计的变体，理论上与互信息下界相关。
- **ADARec 中的应用**：InstanceCL 和 IntentCL 均用 InfoNCE 形式。

### D.4 温度参数（Temperature）

- **类型**：超参数
- **相关方向**：对比学习、注意力
- **具体含义**：控制 softmax / InfoNCE 的锐度。$\tau$ 大 → 分布更均匀（更难区分负例）；$\tau$ 小 → 分布更尖锐（更容易学到大间隔）。
- **ADARec 中的值**：InstanceCL $\tau=0.07$；IntentCL $\tau=0.07$。

### D.5 正样本 / 负样本（Positive / Negative Sample）

- **类型**：对比学习核心概念
- **相关方向**：数据增强、负采样
- **具体含义**：
  - **正样本**：与锚点来自同一实例/具有相同标签的样本（如原始序列与其扩散增强版本）。
  - **负样本**：与锚点不同的样本（如 batch 内其他用户序列）。
- **ADARec 中的构造**：InstanceCL 的正样本 = `input_emb` 与 `augmented_emb`；负样本 = batch 内其他用户的序列表示。

---

## E. 评测指标

### E.1 HIT@K（Hit Rate at K）

- **类型**：推荐评测指标
- **相关方向**：Top-N 推荐、排序
- **具体含义**：推荐列表前 K 项中是否命中目标物品（0/1 平均）。等价于 Recall@K（在单目标场景）。
- **ADARec 中的 K**：K=5 和 K=20。
- **代码**：`utils.py` L129。

### E.2 NDCG@K（Normalized Discounted Cumulative Gain at K）

- **类型**：推荐评测指标
- **相关方向**：排序质量、位置折扣
- **具体含义**：归一化折扣累计增益。既看是否命中，也看命中位置是否靠前（前部权重更高）。值域 [0,1]，1 为完美排序。
- **ADARec 中的 K**：K=5 和 K=20。
- **代码**：`utils.py` L130-131。

### E.3 重排式评测（Reranking-style Evaluation）

- **类型**：评测协议
- **相关方向**：候选集构造、评测公平性
- **具体含义**：评测时将目标物品放入候选集（包含正例和大量负例），用模型打分排序，根据排名计算指标。候选集构造方式影响指标绝对值（不同候选集不可直接比较）。
- **ADARec 中的细节**：评测前将训练交互从候选池中清零（`rating_pred[batch_train.nonzero()] = 0`），防止已交互物品占用排名位置。

### E.4 早停指标（Early Stopping Metric）

- **类型**：训练监控指标
- **相关方向**：过拟合防控
- **具体含义**：用于判断是否触发早停的评测指标。ADARec 中是**各专家 raw NDCG@20**，不是 fused NDCG@5（用户实际看到的融合指标）。
- **⚠️ 重要区别**：日志打印的 raw NDCG@5 与 fused NDCG@5 不是同一个数；早停只看 raw NDCG@20。

---

## F. 模型训练与优化

### F.1 BPR 损失（Bayesian Personalized Ranking Loss）

- **类型**：推荐损失函数
- **相关方向**：隐式反馈、排序学习
- **具体含义**：$\mathcal{L} = -\log \sigma(\hat{r}^+ - \hat{r}^-)$，鼓励正样本分数高于负样本分数。用 sigmoid 转换为概率损失。
- **代码**：`trainers.py` L527-531 `bpr_loss(pred_pos - pred_neg).mean()`。

### F.2 欧氏距离（Euclidean Distance）

- **类型**：度量
- **相关方向**：KMeans、损失函数
- **具体含义**：$\| \mathbf{a} - \mathbf{b} \|_2 = \sqrt{\sum_d (a_d - b_d)^2}$，或平方形式 $\| \mathbf{a} - \mathbf{b} \|^2$ 用于 KMeans 和 IntentCL。

### F.3 EMA（Exponential Moving Average，指数移动平均）

- **类型**：模型更新策略
- **相关方向**：KMeans 中心更新、模型稳定性
- **具体含义**：$c_{\text{new}} = \lambda \cdot c_{\text{old}} + (1-\lambda) \cdot c_{\text{batch}}$，常用 $\lambda=0.99$。使簇中心平滑更新，减少噪声影响。
- **代码**：`modules.py` L230-244，KMeans 的簇中心更新。

### F.4 patience（早停耐心值）

- **类型**：训练超参数
- **相关方向**：过拟合、泛化
- **具体含义**：允许验证指标"无改善"的最多 epoch 数。ADARec 中 `patience=40`。
- **⚠️ 陷阱**：若 `delta=0`，任何微涨都会重置计数器，导致早停几乎永不触发（Beauty 的实际情况：跑了 341/400 从未触发早停）。

### F.5 `weights_only`（PyTorch 加载安全参数）

- **类型**：PyTorch API 行为
- **相关方向**：模型加载、兼容性
- **具体含义**：PyTorch 2.6+ 的 `torch.load` 默认 `weights_only=True`（只允许加载纯权重张量），遇到 numpy 数组或自定义类会报错。
- **ADARec 中的坑**：resume snapshot 包含 numpy 数组（early stopper 的 best_score），必须 `weights_only=False`。

---

## G. 数据与评测协议

### G.1 训练集 / 验证集 / 测试集划分

- **类型**：实验范式
- **相关方向**：时序划分、数据泄漏
- **具体含义**：按时间顺序划分；训练用前 t-2 时刻交互；验证用 t-1 时刻交互；测试用 t 时刻交互。
- **ADARec 中的实现**：`utils.py` `item_list[:-2]` = 训练物品，`item_list[:-1]` = 验证物品，`item_list` = 候选物品。

### G.2 验证物品泄漏（Test Item Leakage in Validation）

- **类型**：评测 Bug
- **相关方向**：评测公平性
- **具体含义**：验证阶段（计算验证指标时）未将验证物品从候选池中移除，导致验证目标物品仍在候选中与自身竞争，压低 NDCG。
- **ADARec Bug #6**：--do_eval 模式忘记切换 `train_matrix = test_rating_matrix`，导致验证物品泄漏。
- **修复**：`main.py` do_eval 分支加载权重后加 `trainer.args.train_matrix = test_rating_matrix`。

### G.3 候选集构造

- **类型**：评测协议
- **相关方向**：负采样、Top-N 推荐
- **具体含义**：测试时从哪些物品中挑选目标。ADARec 的候选集 = 训练集所有物品（含训练交互）；评测时将已交互物品分数清零来模拟真实推荐。
- **⚠️ 注意**：候选集不同，同一模型的 NDCG@K 不可比较。

---

## H. ADARec 特有概念

### H.1 自适应深度 $T_u$（Adaptive Depth）

- **类型**：模型设计
- **相关方向**：动态计算图
- **具体含义**：根据用户 $u$ 的序列特征，自适应决定加噪强度（对应选 depth=4/5 或 depth=9/10）。
- **代码实际**：只有两个离散深度，不是"0 到 k 连续选择"；且自适应体现在两个深度的加权比例，不体现在专家路由。

### H.2 双专家（Dual Expert）

- **类型**：模型架构
- **相关方向**：MoE、模型集成
- **具体含义**：两个独立的 SASRec 模型：`model_e5`（depth=4,5）和 `model_e10`（depth=9,10），各自独立训练，独立早停。
- **与 HP-MoE 论文描述的差异**：论文描述有门控网络动态选择；代码是固定等权平均。

### H.3 `raw_output` vs `adaptive_output`

- **类型**：模型分支
- **相关方向**：扩散增强
- **具体含义**：
  - `raw_output`：原始 Transformer 输出，未加扩散增强。
  - `adaptive_output`：经自适应深度控制 + 扩散增强后的 Transformer 输出。
- **损失中的角色**：`adaptive.detach()` 逼近 `raw_output`，鼓励扩散分支预测"干净"表示。

### H.4 探索损失（Exploration Loss）

- **类型**：辅助损失
- **相关方向**：多臂老虎机、强化学习探索
- **具体含义**：控制器输出的负熵，鼓励探索（不锁定单一深度），防止表示坍塌。
- **ADARec 中的值**：权重 `exploration_weight = 1e-4`（论文建议 0.01）。

### H.5 重构损失 vs 噪声预测损失

- **类型**：扩散训练目标
- **相关方向**：DDPM
- **具体含义**：
  - **重构损失**（代码）：$\| \text{ADP}(E_t) - \text{RAW}(E_0) \|^2$，扩散输出预测"干净表示"。
  - **噪声预测损失**（论文描述）：$\| \epsilon_\theta - \epsilon \|^2$，扩散输出预测注入的噪声本身。
- **两者区别**：重构损失让扩散分支学习"去噪"，噪声预测损失让扩散分支学习"噪声分布"；语义不同，效果相近。

---

## I. 其他技术术语

### I.1 Transformer 编码器

- **类型**：模型组件
- **相关方向**：Self-Attention、位置编码
- **具体含义**：Multi-Head Self-Attention + 前馈网络的堆叠，是 BERT / GPT / SASRec 的核心。

### I.2 位置编码（Position Embedding）

- **类型**：序列建模组件
- **相关方向**：Transformer
- **具体含义**：为序列中每个位置分配一个向量，使模型感知位置顺序。ADARec 用可学习的位置嵌入（`nn.Parameter`）。

### I.3 LayerNorm

- **类型**：归一化技术
- **相关方向**：深度学习优化
- **具体含义**：在特征维度上做均值方差归一化，稳定训练。ADARec 用自定义 `LayerNorm`（RMSNorm 变体）。

### I.4 Dropout

- **类型**：正则化技术
- **相关方向**：过拟合防控
- **具体含义**：训练时随机置零部分神经元输出。ADARec 的 `attention_probs_dropout_prob=0.5`，`hidden_dropout_prob=0.5`。

### I.5 Adam 优化器

- **类型**：优化算法
- **相关方向**：深度学习训练
- **具体含义**：自适应学习率优化器，结合 Momentum 和 RMSProp。ADAREC 用 Adam，`weight_decay=1e-4`。

### I.6 `.toarray()` vs 稀疏操作

- **类型**：内存优化
- **相关方向**：推荐系统数据表示
- **具体含义**：
  - `.toarray()`：稀疏矩阵 → 密集数组，高内存占用。
  - `sparse.nonzero()`：直接获取非零索引，无需转密集。
- **OOM 教训**：双专家验证时 `rating_pred = train_matrix[batch].toarray()` 产生 (256, 18359) int32 数组 ≈ 17.9 MiB/批，累积导致 host RAM 耗尽。

### I.7 重参数化技巧（Reparameterization Trick）

- **类型**：概率编程技巧
- **相关方向**：VAE、Gumbel-Softmax
- **具体含义**：将随机采样节点从模型参数中分离，通过噪声变量注入，保持反向传播可通。Gumbel-Softmax 用此技巧实现离散采样的可微化。

---

## J. 具身智能（Embodied AI）

### J.1 具身智能（Embodied Intelligence）

- **类型**：研究范式
- **相关方向**：机器人、多模态大模型、VLA
- **具体含义**：让智能体（机器人）通过与物理世界交互来感知、理解并行动，形成「身体—环境—任务」闭环的智能。区别于纯软件智能（对话、图像生成），强调物理实体的感知-决策-行动闭环。2025 年首次写入中国政府工作报告，被纳入「十五五」未来产业布局。
- **三段式结构**：大脑（AI 大模型：语言交互 / 环境感知 / 任务决策）+ 小脑（运动控制算法：运动协调 / 身体平衡）+ 本体（机械结构）。

### J.2 VLA（Vision-Language-Action，视觉-语言-动作大模型）

- **类型**：模型范式
- **相关方向**：端到端机器人、多模态大模型
- **具体含义**：将视觉（Vision）、语言（Language）、动作（Action）统一到单个大模型中端到端输出，直接从感知映射到机器人动作，是当前具身智能的主流技术路线。
- **代表**：Figure Helix、英伟达 GR00T N1、智元 ViLLA 架构（VLM + MoE）。

### J.3 Sim-to-Real（仿真到现实迁移）

- **类型**：训练技术
- **相关方向**：机器人、世界模型、强化学习
- **具体含义**：在仿真环境训练的策略迁移到真实机器人时存在「仿真-现实鸿沟」（域差异），需通过域随机化、真实数据微调等缓解。
- **局限**：当前纯生成式世界模型（如 Genie 3）迁移到物理机器人效果仍有限。

### J.4 具身智能等级划分

- **类型**：分级标准
- **相关方向**：产业落地、能力分级
- **具体含义**：L1 完全人控 → L2 基础辅助 → L3 具身智能 + 训练监督（2026 年迈入 L3 初阶量产）→ L4 自成长智能 → L5 完全自主智能。

---

## K. AI Agent 与多智能体

### K.1 AI Agent（智能体）

- **类型**：系统范式
- **相关方向**：大模型应用、工具调用、自动化
- **具体含义**：能感知环境、自主决策、调用工具并执行行动的 AI 系统，核心是「感知—决策—行动—记忆」认知闭环。2025 年被称为「智能体元年」。
- **范式转移**：Agent 从单点效率工具（AI Agent）进化为系统生产力引擎（Agentic AI）。

### K.2 多智能体系统（Multi-Agent System）

- **类型**：系统架构
- **相关方向**：协作、分工、涌现行为
- **具体含义**：多个智能体协同完成任务。三类架构：层级式（管理者分解任务，如 CrewAI）、平等式（无中心协商，如 AutoGen）、混合式。
- **局限**：调用倍增、通信拥塞、错误传播，需有界角色 / 共享状态规则 / 冲突消解 / 预算控制 / 失败处理。

### K.3 MCP（Model Context Protocol，模型上下文协议）

- **类型**：开放协议
- **相关方向**：工具连接、标准化、生态
- **具体含义**：Anthropic 提出的模型上下文协议，标准化大模型与外部工具 / 数据源的连接方式，2025/12 成为 Linux Foundation Agentic AI Foundation 项目。
- **对比**：A2A（Google 的 Agent-to-Agent 协议）用于智能体间互操作，2025/7 移交 Linux Foundation 治理。

### K.4 AgentOps

- **类型**：工程实践
- **相关方向**：可观测性、评估、治理
- **具体含义**：对智能体运行做可观测性 / 评估 / 治理的工程实践，工具（如 agentops.ai）捕获模型调用、工具使用、延迟、token 消耗。

---

## L. 世界模型（World Model）

### L.1 世界模型（World Model）

- **类型**：模型范式
- **相关方向**：视频生成、机器人、自动驾驶
- **具体含义**：学习环境动态规律、能预测「下一步会怎样」的模型。区别于帧预测（只知「看起来怎样」），世界模型知道「会做什么」（因果、物理）。
- **技术流派四类**：① 视频生成 / 交互仿真派（Sora 2、Genie 3、Runway GWM-1）；② 3D 空间智能派（World Labs Marble）；③ 物理 AI 基础设施派（NVIDIA Cosmos）；④ JEPA / 潜在空间派（V-JEPA 2、Dreamer）。

### L.2 动作条件生成（Action-Conditioned Generation）

- **类型**：模型机制
- **相关方向**：世界模型、机器人
- **具体含义**：以「动作」作为条件输入生成未来状态，是 2025 年世界模型的关键转折——模型从「观察者」升级为可交互仿真环境与机器人大脑基础。

### L.3 复合误差累积（Compounding Error）

- **类型**：模型缺陷
- **相关方向**：世界模型、长程预测
- **具体含义**：多步 rollout 预测时误差逐步累积，10 步尚可、100 步通常失败，是 Dreamer 家族等世界模型的共同限制。

### L.4 世界模型四大开放难题

- **类型**：研究挑战
- **相关方向**：因果推理、不确定性估计
- **具体含义**：① 因果推理缺失（学「B 跟随 A」而非「A 导致 B」，OOD 预测崩溃）；② 复合误差累积；③ 数据效率（仍需海量视频语料，人类少量经验即可泛化物理）；④ 不确定性估计不成熟（校准的「我不知道」尚缺，新条件下过信危险）。

---

## M. 大模型与推荐系统（LLM×Rec）

### M.1 大语言模型推荐（LLM-based Recommendation）

- **类型**：研究范式
- **相关方向**：生成式推荐、提示学习、微调
- **具体含义**：将大语言模型（LLM）的世界知识、语义理解、推理与 scaling law 能力引入推荐系统。经典「WHERE + HOW」框架：WHERE 指 LLM 在推荐管线五阶段（特征工程 / 特征编码 / 打分排序 / 用户交互 / 管线控制）的角色；HOW 指训练与推理策略二分（是否调 LLM、推理是否用 CRM）。
- **关键发现**：直接 ICL / prompt 效果差（ChatGPT 对推荐任务 AUC ≈ 0.5 即瞎猜），需用推荐数据 tuning（如 TALLRec 用 LLaMA-7B + LoRA）。

### M.2 生成式推荐（Generative Recommendation）

- **类型**：新范式
- **相关方向**：语义 ID、自回归生成、scaling law
- **具体含义**：将推荐从「判别打分」重构为「生成任务」——生成物品的语义 Token / ID 序列，而非对候选集打分。范式转移主线：ID → 语义 Token → 统一 Transformer → scaling laws。
- **代表**：HSTU、OneRec（端到端生成式替代级联检索+排序）、TIGER（语义 ID 生成式检索）。

### M.3 LLMERS（LLM-Enhanced Recommender System）

- **类型**：技术路线
- **相关方向**：推理期成本、延迟约束
- **具体含义**：强调消除「推理期使用 LLM」以适配在线延迟 / 内存成本，仅在训练阶段用 LLM 增强。按增强组件分三 taxonomy：Knowledge Enhancement / Interaction Enhancement / Model Enhancement。

---

## N. 扩散序列推荐前沿

### N.1 扩散推荐（Diffusion-based Recommendation）

- **类型**：模型范式
- **相关方向**：DDPM、去噪、生成式推荐
- **具体含义**：将扩散模型用于推荐（表征去噪、物品生成、序列增强）。两大路线：① 交互扩散（DiffRec，学习用户交互的生成过程）；② 序列扩散（DiffuRec，把物品表示建模为分布）。
- **ADARec 的定位**：扩散作为「数据增强」手段，用于从稀疏序列重建用户意图的层次表示，而非直接生成物品。

### N.2 嵌入坍塌（Embedding Collapse）

- **类型**：训练问题
- **相关方向**：扩散推荐、自监督
- **具体含义**：扩散去噪过程中所有物品嵌入趋于收敛到同一区域，区分度丧失。ADRec 用「per-token 独立噪声」缓解（HR@20 +15.45%），ADARec 用意图级对比学习（IntentCL）保持区分性。
- **注意区分**：ADRec（KDD 2025，解物品嵌入坍塌）与 ADARec（AAAI 2026，解用户意图坍塌）名称相近但研究对象不同。

---

## O. 工业界生成式推荐与大推荐模型 Scaling Law

> 范式转移主线：物品 ID → 语义 Token → 统一 Transformer → Scaling Law。传统「召回-粗排-精排-重排」级联被端到端生成式大模型逐步替代。

### O.1 Scaling Law（扩展律）

- **类型**：规律
- **相关方向**：大模型、算力
- **具体含义**：模型性能随参数规模 / 训练算力 / 数据规模增大而按幂律提升。早期推荐模型（DLRM）不体现 Scaling Law——传统「稀疏扩展」（加大 embedding 行/维度）无法增强高阶特征交互、访存密集难用 GPU 算力；HSTU 等工作证明推荐模型在生成式框架下也能获得 Scaling Law 收益。
- **关键转折**：把推荐重定义为「Seq2Seq 序列转导任务」（统一 token 化），使推荐受益于与 GPT/LLaMA 一致的 Scaling Law。

### O.2 HSTU（Hierarchical Sequential Transducer Unit）

- **类型**：模型架构（Meta，ICML 2024）
- **相关方向**：生成式推荐、Scaling Law
- **具体含义**：Meta 首个展示推荐 Transformer 类架构具备有利缩放性质的工作，将用户行为/物品特征/上下文统一序列化为按时间排序的 token 做自回归生成。
- **四大提速来源**：① **去 softmax**（pointwise 非线性激活替代 softmax，因推荐词表非平稳、每天新增物品；还产生稀疏 GEMM）；② **Stochastic Length**（训练随机采子序列，解决用户历史长度重尾分布，长上下文泛化优于 RoPE 外推）；③ **减少线性投影**（Transformer 块从 6 层线性减到 2 层）；④ **M-FALCON**（微批缓存用户 KV cache 摊还）。
- **线上效果**：1.5 万亿参数，ranking 场景 A/B 提升 12.4%，数十亿用户平台多面上线。
- **警示**：Synerise 复现发现简单 BaseModel 在 Amazon Books 反超 HSTU 55-292%，说明 HSTU 的取舍针对 Meta 十亿级流式场景，不自动迁移到小/异分布数据集。

### O.3 ULTRA-HSTU（HSTU 2.0）

- **类型**：模型架构（Meta，arXiv 2602.16986）
- **相关方向**：生成式推荐、稀疏注意力、系统协同设计
- **具体含义**：HSTU 的模型-系统端到端协同设计升级。核心是 **SLA（Semi-Local Attention，半局部注意力）**：注意力掩码拆为局部窗口 K1 + 全局窗口 K2，复杂度由 O(L²) 降为 O((K1+K2)·L)。
- **其他优化**：输入序列合并 item 与 action 使长度减半（FLOPs 降 4×）、注意力截断、MoT（混合 transducer）、负载均衡随机长度采样、混合精度（BF16/FP8/INT4）。
- **规模**：18 层，服务 16k 序列；比常规模型训练缩放快 >5×、推理快 21×；Meta 生产全量部署，消费/互动指标 +4%~8%。
- **结论**：工业场景下 self-attention 严格优于 cross-attention（对比字节 STCA 用 cross-attention 仅 2-4 层）。

### O.4 OneRec（端到端生成式推荐）

- **类型**：系统范式（快手，KDD 2025）
- **相关方向**：生成式推荐、语义 ID、MoE、RL 对齐
- **具体含义**：用单个端到端 Encoder-Decoder 生成式模型替代「召回-粗排-精排-重排」整条级联链路。
- **动机**：级联三大瓶颈——① 计算碎片化（精排 GPU MFU 仅 4.6%/11.2%，远低于 LLM 40%+）；② 优化目标冲突（数百个目标分散各阶段互相掣肘）；③ 与 AI 前沿脱节。
- **Tokenizer**：融合视频多模态 + 用户行为，用 **RQ-Kmeans** 分层量化为 3 层粗到细语义 ID。
- **Decoder**：MoE 增强 + Next-Token-Prediction 自回归（激活约 13% 参数），支持会话级/分段生成。
- **RL 对齐**：偏好+格式+业务奖励综合系统 + 个性化 P-Score，用改进 **ECPO 算法**（严格截断负优势梯度，因推荐只有一次展示机会）。
- **线上**：快手双端全量上线，观看时长提升（口径 +0.54%/+1.6% 并存），OPEX 仅传统方案 ~10%。

### O.5 token-based 排序模型（RankMixer / OneTrans / TokenMixer-Large）

- **类型**：工业排序模型（字节）
- **相关方向**：特征交互、硬件感知、MoE
- **具体含义**：把工业排序从「Memory-bound 的手工特征交叉」改造为「Compute-bound 的 token 化 Transformer」，是 token-based 推荐系列奠基工作。
- **RankMixer（KDD 2025）**：无参 **Multi-head Token Mixing**（跨 token 特征交叉，替代二次方 self-attention 避免注意力矩阵 Memory-bound）+ **Per-token FFN**（不同特征子空间独立参数）。MFU 从 4.5% 提到 45%；抖音全量上线。
- **OneTrans（WWW 2026）**：统一 tokenizer（序列特征按时间戳/意图优先级合并，插入 [SEP] 分隔不同行为序列；非序列特征分组/自动拆分）+ 混合参数化（S-token 共享参数、NS-token 专属参数）+ 因果注意力 + **跨请求 KV 缓存**。线上每用户 GMV +5.68%。
- **TokenMixer-Large（arXiv 2026）**：扩展到 7B/15B 参数，修复 RankMixer 残差设计缺陷（引入 Mixing & Reverting + 层间残差）+ Sparse-Pertoken MoE。MFU 达 60%，电商 GMV +2.98%。

### O.6 推理式大推荐模型（R²ec / RecZero）

- **类型**：模型范式（NeurIPS 2025）
- **相关方向**：LLM×Rec、强化学习、自主推理
- **具体含义**：把 DeepSeek-R1 的纯 RL 涌现推理引入推荐，让 LLM 先「推理」再「推荐」，而非把 LLM 当外部推理模块（解耦）或蒸馏 teacher。
- **R²ec（港理工）**：双头架构（lm_head 出推理 token + rec_head 出物品评分）+ **RecPO 无标注 RL 框架**（融合奖励 R = β·R_similarity + R_ranking，GRPO/RLOO + KL 正则）。Hit@5 相对 +68.67%、NDCG@20 +45.21%。
- **RecZero/RecOne（阿里系）**：受 DeepSeek-R1-Zero 启发纯 RL 涌现推理——RecZero 纯 RL GRPO（规则奖励 = 格式 + 准确率，无 teacher 无标注）；RecOne 冷启动 SFT（DeepSeek-R1 当教师生成推理+评分，预测错时喂真实评分重写自洽解释）+ RL。
- **命名甄别**：RecZero（阿里，评分预测 GRPO 推理）≠ ReZero（Menlo，RAG 的 Retry-Zero），≠ R²ec（港理工双头）。

### O.7 RLVR（Reinforcement Learning with Verifiable Rewards，可验证奖励强化学习）

- **类型**：训练范式
- **相关方向**：推理模型、数学/代码
- **具体含义**：用客观可自动验证的奖励信号（数学对错、代码单元测试通过率）做 RL，无需训练奖励模型。DeepSeek-R1 用 GRPO + 规则奖励（数学）+ 编译/测试奖励（代码）在 base 模型上直接涌现反思、自我纠错等长推理行为，是推理模型（LRM）的核心训练方法。
