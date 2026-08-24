# 数学公式笔记 — ADARec 中的核心数学

> 本文件从 ADARec 论文与代码中提取核心数学公式，标注对应公式编号、变量含义、代码实现位置。适合作为 ADARec 数学基础的速查手册。

---

## 1. 序列推荐基础

### 1.1 SASRec 前向编码

$$E^{(l)} = \text{TransformerBlock}(E^{(l-1)}), \quad l = 1, 2, \dots, L$$

**含义**：输入嵌入 $E^{(0)} \in \mathbb{R}^{T \times D}$ 经过 $L$ 层 Transformer 编码器逐层变换，得到序列表示 $E^{(L)}$。每层包含：Multi-Head Self-Attention（MHSA） + 前馈网络（FFN），均带残差连接和 LayerNorm。

**代码**：`modules.py` L117-140 `class Layer`，`models.py` L143-149 `class Encoder`。

### 1.2 预测分数（点积）

$$\hat{r}_{u,i} = \mathbf{h}_u \cdot \mathbf{e}_i = \sum_{d=1}^{D} h_{u,d} \cdot e_{i,d}$$

**含义**：用户表示 $\mathbf{h}_u \in \mathbb{R}^D$ 与物品嵌入 $\mathbf{e}_i \in \mathbb{R}^D$ 的点积作为偏好分数，用于排序。实际代码中 $\mathbf{h}_u$ 取 Transformer 输出中最后一个有效位置的表示。

**代码**：`trainers.py` L533-534 `torch.sum(user_emb * item_emb, dim=-1)`。

---

## 2. 自适应扩散增强（ADC）

### 2.1 控制器输出（Gumbel-Softmax，论文）

$$\pi_i = \frac{\exp\left((\log \alpha_i + g_i) / \tau\right)}{\sum_{j=1}^{M} \exp\left((\log \alpha_j + g_j) / \tau\right)}$$

**含义**（论文 Section 3.2）：用 Gumbel-Softmax 得到离散 one-hot 分布 $\pi$ 作为深度选择软变量，$g_i \sim \text{Gumbel}(0,1)$ 是重参数化噪声，$\tau$ 是温度参数。

**⚠️ 代码实际**：ADARec 代码 `models.py` L308 用的是普通 `F.softmax(logits, dim=-1)`，**没有** Gumbel 噪声和温度机制，等价于 $\tau=1$ 的简化版本：

$$\pi_i = \frac{\alpha_i}{\sum_j \alpha_j}$$

### 2.2 前向扩散（闭式，HDA）

$$E_t = \sqrt{\bar{\alpha}_t} \cdot E_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**含义**（论文 Section 3.2 / ADARec 代码 `diffusion_utils.py`）：在噪声调度表 $\{\alpha_t\}$ 下，干净嵌入 $E_0$ 经过 $t$ 步扩散后变为带噪嵌入 $E_t$。$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$。

**代码**：`diffusion_utils.py` L10-13 `forward_diffusion(E0, t, noise)`。

### 2.3 线性噪声调度

$$\beta_t = \beta_{\text{start}} + \frac{t}{T_{\max} - 1} \cdot (\beta_{\text{end}} - \beta_{\text{start}})$$

$$\alpha_t = 1 - \beta_t$$

**含义**：从 $\beta_1 = 1e^{-4}$ 线性增长到 $\beta_{50} = 0.02$，$T_{\max} = 50$。ADARec 用**线性调度**而非余弦调度。

**代码**：`diffusion_utils.py` L4-9 `get_noise_schedule`（`diffusion_t_max=50`）。

### 2.4 自适应加权混合（代码实现）

$$\mathbf{v}_{\text{adaptive}} = \sum_{i=1}^{M} \pi_i \cdot \mathbf{v}_i$$

**含义**：给定控制器输出的概率向量 $\pi = (\pi_1, \pi_2)$，将 $M=2$ 个深度层（depth=4 和 depth=5）的输出加权求和，得到自适应混合表示。

**代码**：`models.py` L176-194 `if probs is not None` 分支，$\pi$ 来自 `F.softmax(logits)`。

### 2.5 探索损失（负熵）

$$L_{\text{expl}} = -\frac{1}{B} \sum_{u=1}^B \sum_{i=1}^M \pi_{u,i} \ln \pi_{u,i} = -\mathbb{H}(\pi)$$

**含义**：鼓励控制器输出**均匀分布**（最大熵），避免总是选择同一个深度。代码中计算两个深度的负熵。

**代码**：`trainers.py` L660-662 `loss = -(torch.log(prob[...,0]) + torch.log(prob[...,1])) / 2`（对应 $M=2$ 的均匀熵）。

### 2.6 去噪损失（表示重构）

$$L_{\text{deno}} = \mathbb{E}_t \left[ \| \text{ADP}(E_t) - \text{RAW}(E_0) \|^2 \right]$$

**含义**：扩散分支输出 $\text{ADP}(E_t)$ 预测"干净"目标 $\text{RAW}(E_0)$，用 MSE 衡量。这是**表示重构损失**，而非噪声预测（与标准 DDPM 不同）。

**⚠️ 注意**：论文描述为"噪声预测"，但代码实现是"表示重构"，两者不等价。

**代码**：`trainers.py` L660 `F.mse_loss(adaptive.detach(), raw_output)`（`adaptive.detach()` 阻断梯度回传控制器）。

---

## 3. 双专家混合（HP-MoE）

### 3.1 专家前向

$$\mathbf{h}_u^{(e5)} = \text{Encoder}_{e5}(\text{item_seq}_u), \quad \mathbf{h}_u^{(e10)} = \text{Encoder}_{e10}(\text{item_seq}_u)$$

**含义**：两个独立的 SASRec 编码器，分别用 depth=4,5 和 depth=9,10 层。两者参数独立，**不是**共享主干。

**代码**：`models.py` L156-169 `class DualExpertModel`，`self.model_e5` / `self.model_e10` 是两个独立 `SASRecModel` 实例。

### 3.2 固定权重融合（代码）

$$\hat{r}_{u,i}^{\text{fused}} = \frac{1}{2} \cdot \hat{r}_{u,i}^{(e5)} + \frac{1}{2} \cdot \hat{r}_{u,i}^{(e10)}$$

**含义**：两个专家的预测分数直接等权平均。**没有**可学习门控网络。

**⚠️ 注意**：论文描述为可学习门控，但代码用固定 0.5/0.5，等价于"双模型集成"而非"MoE 路由"。

**代码**：`trainers.py` L895-901 `pred_fused = (pred_e5 + pred_e10) / 2`。

---

## 4. 对比学习（CL）

### 4.1 InstanceCL（InfoNCE）

$$L_{\text{InstanceCL}} = -\frac{1}{B} \sum_{u=1}^B \log \frac{\exp(\mathbf{h}_u^+ \cdot \mathbf{h}_u / \tau)}{\sum_{j=0}^{K} \exp(\mathbf{h}_u \cdot \mathbf{h}_j / \tau)}$$

**含义**：同一用户的原始视图 $\mathbf{h}_u$ 与增强视图 $\mathbf{h}_u^+$ 为正例，与 batch 内其他用户的表示为负例。温度参数 $\tau$ 控制负例区分度。

**代码**：`modules.py` L170-190 `class NCELoss`（$\tau$ 默认 0.07），`trainers.py` L672-674 调用。

### 4.2 IntentCL（PCL）

$$L_{\text{IntentCL}} = \frac{1}{B \cdot K} \sum_{u=1}^B \sum_{k=1}^K \log \frac{\exp(\|\mathbf{h}_u - \mathbf{c}_k\|^2 / \tau)}{\sum_{j=1}^K \exp(\|\mathbf{h}_u - \mathbf{c}_j\|^2 / \tau)}$$

**含义**：将序列表示拉近意图簇中心 $\mathbf{c}_k$，推远其他簇中心。$\|\cdot\|^2$ 是欧氏距离平方，$\tau$ 是 InfoNCE 温度。

**代码**：`modules.py` L192-215 `class PCLoss`；`trainers.py` L675 调用（代码中 `center_center_distance` 算簇中心间距离，用于损失计算）。

### 4.3 距离损失（辅助正则）

$$L_{\text{dist}} = \frac{1}{M^2} \sum_{i,j} \| \mathbf{v}_i^{(e5)} - \mathbf{v}_j^{(e10)} \|^2$$

**含义**：鼓励两个专家的输出表示尽可能接近，增强融合的一致性。

**代码**：`trainers.py` L647-653，计算两专家所有深度层输出之间的均方距离。

---

## 5. 总损失函数

### 5.1 完整加权损失

$$L_{\text{total}} = \underbrace{L_{\text{rec}}}_{\text{推荐损失}} + \lambda_{\text{deno}} \underbrace{L_{\text{deno}}}_{\text{去噪损失}} + \lambda_{\text{expl}} \underbrace{L_{\text{expl}}}_{\text{探索损失}} + \lambda_{\text{cl}} \underbrace{L_{\text{cl}}}_{\text{CL损失}}$$

**含义**：四部分损失的加权和。$L_{\text{rec}}$ 是推荐主损失（交叉熵），其余为辅助损失。

**默认值**（Beauty/Sports 超参表）：
$$\lambda_{\text{deno}} = 0.0001,\quad \lambda_{\text{expl}} = 0.0001,\quad \lambda_{\text{cl}} = 1$$

**⚠️ 注意**：论文建议 $\lambda_{\text{expl}} = 0.01$，$\lambda_{\text{deno}} = 0.1$，但代码实际用 $1e-4$，相差 100 倍。

**代码**：`trainers.py` L657-675，权重从 `self.diffusion_weight`（对应 $\lambda_{\text{deno}}$）和 `self.exploration_weight`（对应 $\lambda_{\text{expl}}$）读取。

### 5.2 推荐损失（交叉熵）

$$L_{\text{rec}} = -\frac{1}{B \cdot |\mathcal{N}_u|} \sum_{u \in \mathcal{B}} \sum_{i \in \mathcal{N}_u} \ln \sigma(\hat{r}_{u,i}^+ - \hat{r}_{u,i}^-)$$

**含义**：正负样本对的排名交叉熵损失（BPR 变体）。$\hat{r}_{u,i}^+$ 是目标物品分数，$\hat{r}_{u,i}^-$ 是负采样物品分数，$\sigma$ 是 sigmoid 函数。

**代码**：`trainers.py` L527-531 `bpr_loss(pred_pos - pred_neg).mean()`。

---

## 6. 评测指标

### 6.1 HIT@K

$$\text{HIT}@K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \mathbb{1}\left[ \text{rank}_u(i^+) \leq K \right]$$

**含义**：推荐列表长度 $K$ 内是否命中目标物品（0/1 指标），也叫 Hit Rate。

**代码**：`utils.py` L129-132 `get_sample_scores`，返回 4 元素列表 `[HIT@5, NDCG@5, HIT@20, NDCG@20]`。

### 6.2 NDCG@K

$$\text{NDCG}@K = \frac{1}{\text{IDCG}@K} \sum_{i=1}^{K} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$$

**含义**：Discounted Cumulative Gain，按排名位置给命中物品折扣打分，归一化后最大为 1。ADARec 中 relevant=1（目标物品），故 DCG 即 $\frac{1}{\log_2(\text{rank}+1)}$。

**代码**：`utils.py` L130-131，按 rank 计算位置折扣 NDCG。

---

## 7. KMeans 聚类

### 7.1 目标函数

$$\min_{\{\mathbf{c}_k\}} \sum_{n=1}^N \min_k \| \mathbf{h}_n - \mathbf{c}_k \|^2$$

**含义**：将 $N$ 个序列表示分配到 $K$ 个簇，最小化到簇中心的欧氏距离平方之和。

**代码**：`models.py` L47-64 `class KMeans`，`modules.py` L219-271 `def forward`。$K$ 由 `num_intent_clusters` 控制（Beauty=256，Sports/Toys=512）。

### 7.2 意图嵌入计算

$$\mathbf{c}_k = \frac{1}{|\mathcal{S}_k|} \sum_{\mathbf{h} \in \mathcal{S}_k} \mathbf{h}$$

**含义**：每个意图簇中心是簇内所有序列表示的均值。代码用 EMA 更新：`new_center = ema_decay * old_center + (1 - ema_decay) * batch_mean`。

**代码**：`modules.py` L230-244 EMA 更新逻辑。

---

## 8. 公式与代码对照速查表

| 公式编号 | 含义 | 论文位置 | 代码位置 |
|---------|------|---------|---------|
| §2.1 | SASRec 前向 | Section 2 | `modules.py` L117-140 |
| §2.2 | 预测点积 | Section 2 | `trainers.py` L533-534 |
| §3.1 | Gumbel-Softmax（论文） | Section 3.2 | — |
| §3.1' | Softmax（代码） | — | `models.py` L308 |
| §3.2 | 前向扩散 | Section 3.2 | `diffusion_utils.py` L10-13 |
| §3.3 | 线性 β 调度 | Section 3.2 | `diffusion_utils.py` L4-9 |
| §3.4 | 自适应加权混合 | Section 3.2 | `models.py` L176-194 |
| §3.5 | 探索损失（负熵） | Section 3.4 | `trainers.py` L660-662 |
| §3.6 | 去噪损失（MSE） | Section 3.4 | `trainers.py` L660 |
| §4.1 | 双专家独立编码 | Section 3.3 | `models.py` L156-169 |
| §4.2 | 等权融合 | Section 3.3 | `trainers.py` L895-901 |
| §5.1 | InstanceCL | Section 3.4 | `modules.py` L170-190 |
| §5.2 | IntentCL（PCL） | Section 3.4 | `modules.py` L192-215 |
| §5.3 | 距离损失 | Section 3.4 | `trainers.py` L647-653 |
| §5.4 | 总损失 | Section 3.4 | `trainers.py` L657-675 |
| §5.5 | 推荐损失（BPR CE） | Section 3.1 | `trainers.py` L527-531 |
| §6.1 | HIT@K | Section 4.1 | `utils.py` L129-132 |
| §6.2 | NDCG@K | Section 4.1 | `utils.py` L130-131 |
| §7.1 | KMeans 目标 | Section 3.3 | `models.py` L47-64 |
| §7.2 | 意图簇中心 | Section 3.3 | `modules.py` L230-244 |
