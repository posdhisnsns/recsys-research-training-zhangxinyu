# 名词解释笔记 — ADARec 核心术语

> 本文件收录 ADARec（AAAI 2026）论文与代码中出现的核心术语，附通俗解释、相关方向和代码出处。按主题分组，适合快速查阅。

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
