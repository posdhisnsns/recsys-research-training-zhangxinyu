# 数学公式笔记 — 通用经典公式与论文特殊公式

> 本文件将数学公式分为两大部分：
>
> - **第一部分：通用经典公式** —— 推荐系统与深度学习中的公共数学基础，脱离具体论文仍然成立，可用于任何相关模型。
> - **第二部分：论文特殊公式** —— 具体论文中特有的设计，按论文排序（从 §1 开始），每篇论文内部的公式亦各自从 (1) 开始排序。
>
> 每部分下再按主题分小节。

---

# 第一部分：通用经典公式

> 这些公式是推荐系统 / 深度学习的公共数学基础，适用于各类模型，是经典教材与通用框架中的标准内容。

---

## 1. 序列推荐基础

### 1.1 SASRec 前向编码

$$E^{(l)} = \text{TransformerBlock}(E^{(l-1)}), \quad l = 1, 2, \dots, L$$

**含义**：输入嵌入 $E^{(0)} \in \mathbb{R}^{T \times D}$ 经过 $L$ 层 Transformer 编码器逐层变换，得到序列表示 $E^{(L)}$。每层包含：Multi-Head Self-Attention（MHSA） + 前馈网络（FFN），均带残差连接和 LayerNorm。

**通用性**：Transformer 编码器是序列建模的通用结构，广泛用于 SASRec、BERT4Rec 等序列推荐模型。

**代码思路**：

```python
class Layer(nn.Module):
    def __init__(self, config):
        self.attention = SelfAttention(config)      # MHSA
        self.feed_forward = Intermediate(config)    # FFN
        self.attention_norm = LayerNorm(config)
        self.ffn_norm = LayerNorm(config)

    def forward(self, x, attention_mask):
        attn_out = self.attention(x, attention_mask)
        x = self.attention_norm(x + attn_out)        # 残差 + LayerNorm
        ffn_out = self.feed_forward(x)
        x = self.ffn_norm(x + ffn_out)               # 残差 + LayerNorm
        return x

class Encoder(nn.Module):
    def __init__(self, config):
        self.layer = nn.ModuleList([Layer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x, attention_mask):
        for layer in self.layer:
            x = layer(x, attention_mask)
        return x
```

### 1.2 预测分数（点积）

$$\hat{r}_{u,i} = \mathbf{h}_u \cdot \mathbf{e}_i = \sum_{d=1}^{D} h_{u,d} \cdot e_{i,d}$$

**含义**：用户表示 $\mathbf{h}_u \in \mathbb{R}^D$ 与物品嵌入 $\mathbf{e}_i \in \mathbb{R}^D$ 的点积作为偏好分数，用于排序。实际代码中 $\mathbf{h}_u$ 取 Transformer 输出中最后一个有效位置的表示。

**通用性**：点积打分是推荐系统、信息检索、对比学习中最常见的相似度度量。

**代码思路**：

```python
# 取序列最后一位作为用户表示
user_emb = sequence_output[:, -1, :]           # (B, D)

# 全量物品打分:用户表示 @ 物品嵌入矩阵转置
item_emb = model.item_embeddings.weight         # (num_items, D)
rating_pred = torch.matmul(user_emb, item_emb.T)  # (B, num_items)

# 或逐元素点积求和(单物品)
score = torch.sum(user_emb * item_emb, dim=-1)
```

---

## 2. 对比学习（通用经典）

### 2.1 InstanceCL（InfoNCE）

$$L_{\text{InstanceCL}} = -\frac{1}{B} \sum_{u=1}^B \log \frac{\exp(\mathbf{h}_u^+ \cdot \mathbf{h}_u / \tau)}{\sum_{j=0}^{K} \exp(\mathbf{h}_u \cdot \mathbf{h}_j / \tau)}$$

**含义**：同一用户的原始视图 $\mathbf{h}_u$ 与增强视图 $\mathbf{h}_u^+$ 为正例，与 batch 内其他用户的表示为负例。温度参数 $\tau$ 控制负例区分度。

**通用性**：InfoNCE 是对比学习的通用损失，广泛用于 SimCLR、MoCo、CL4SRec、SGL 等模型。

**代码思路**：

```python
class NCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, view1, view2):
        # L2 归一化
        view1 = F.normalize(view1, dim=-1)          # (B, D)
        view2 = F.normalize(view2, dim=-1)          # (B, D)

        # 相似度矩阵
        logits = torch.matmul(view1, view2.T) / self.temperature  # (B, B)

        # 对角线为正样本,标签即 [0,1,...,B-1]
        labels = torch.arange(view1.size(0), device=view1.device)
        loss = self.cross_entropy(logits, labels)
        return loss
```

### 2.2 IntentCL（PCL）

$$L_{\text{IntentCL}} = \frac{1}{B \cdot K} \sum_{u=1}^B \sum_{k=1}^K \log \frac{\exp(\|\mathbf{h}_u - \mathbf{c}_k\|^2 / \tau)}{\sum_{j=1}^K \exp(\|\mathbf{h}_u - \mathbf{c}_j\|^2 / \tau)}$$

**含义**：将序列表示拉近意图簇中心 $\mathbf{c}_k$，推远其他簇中心。$\|\cdot\|^2$ 是欧氏距离平方，$\tau$ 是 InfoNCE 温度。

**通用性**：PCL（Prototypical Contrastive Learning）是通用的原型对比学习框架，可用于任何需要聚类原型的场景。

**代码思路**：

```python
class PCLoss(nn.Module):
    def __init__(self, temperature=0.07):
        self.temperature = temperature

    def forward(self, features, centers):
        # features: (B, D), centers: (K, D)
        # 计算到各簇中心的欧氏距离平方
        dist = torch.cdist(features, centers) ** 2   # (B, K)

        # 距离越小越相似,取负号作为 logits
        logits = -dist / self.temperature            # (B, K)

        # 每个样本最接近的簇作为正类
        labels = torch.argmin(dist, dim=-1)          # (B,)
        loss = F.cross_entropy(logits, labels)
        return loss
```

### 2.3 距离损失（辅助正则）

$$L_{\text{dist}} = \frac{1}{M^2} \sum_{i,j} \| \mathbf{v}_i^{(e5)} - \mathbf{v}_j^{(e10)} \|^2$$

**含义**：鼓励两个分支的输出表示尽可能接近，增强融合的一致性。

**通用性**：表示对齐 / 一致性正则是多分支、多视图模型的通用正则手段（如 Mean Teacher、BYOL 的预测对齐）。

**代码思路**：

```python
# v_e5: (M, B, D), v_e10: (M, B, D),M 为深度层数
dist_loss = 0.0
for i in range(M):
    for j in range(M):
        dist_loss += F.mse_loss(v_e5[i], v_e10[j])
dist_loss = dist_loss / (M * M)
```

---

## 3. 推荐损失（通用经典）

### 3.1 BPR 交叉熵损失

$$L_{\text{rec}} = -\frac{1}{B \cdot |\mathcal{N}_u|} \sum_{u \in \mathcal{B}} \sum_{i \in \mathcal{N}_u} \ln \sigma(\hat{r}_{u,i}^+ - \hat{r}_{u,i}^-)$$

**含义**：正负样本对的排名交叉熵损失（BPR 变体）。$\hat{r}_{u,i}^+$ 是目标物品分数，$\hat{r}_{u,i}^-$ 是负采样物品分数，$\sigma$ 是 sigmoid 函数。

**通用性**：BPR（Bayesian Personalized Ranking）是推荐系统排序任务的通用损失，用于矩阵分解、神经协同过滤、序列推荐等。

**代码思路**：

```python
def bpr_loss(pos_score, neg_score):
    # pos_score, neg_score: (B,)
    return -F.logsigmoid(pos_score - neg_score).mean()

# 或直接交叉熵形式
loss = F.cross_entropy(pred, target)             # pred: (B, num_items)
```

---

## 4. 评测指标（通用经典）

### 4.1 HIT@K

$$\text{HIT}@K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \mathbb{1}\left[ \text{rank}_u(i^+) \leq K \right]$$

**含义**：推荐列表长度 $K$ 内是否命中目标物品（0/1 指标），也叫 Hit Rate。

**通用性**：HIT@K / Recall@K 是推荐、检索、排序任务的通用评测指标。

**代码思路**：

```python
def hit_at_k(ranked_items, true_item, k):
    return int(true_item in ranked_items[:k])

# 批量:按分数排序取 top-k
_, topk_idx = torch.topk(rating_pred, k, dim=-1)     # (B, k)
hit = (topk_idx == target_pos.unsqueeze(-1)).any(dim=-1).float().mean()
```

### 4.2 NDCG@K

$$\text{NDCG}@K = \frac{1}{\text{IDCG}@K} \sum_{i=1}^{K} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$$

**含义**：Discounted Cumulative Gain，按排名位置给命中物品折扣打分，归一化后最大为 1。relevant=1（目标物品）时，DCG 即 $\frac{1}{\log_2(\text{rank}+1)}$。

**通用性**：NDCG 是信息检索与推荐系统的通用排序质量指标。

**代码思路**：

```python
def ndcg_at_k(ranked_items, true_item, k):
    if true_item not in ranked_items[:k]:
        return 0.0
    rank = ranked_items[:k].index(true_item) + 1     # 从 1 开始
    dcg = 1.0 / np.log2(rank + 1)
    idcg = 1.0
    return dcg / idcg
```

---

## 5. KMeans 聚类（通用经典）

### 5.1 目标函数

$$\min_{\{\mathbf{c}_k\}} \sum_{n=1}^N \min_k \| \mathbf{h}_n - \mathbf{c}_k \|^2$$

**含义**：将 $N$ 个表示分配到 $K$ 个簇，最小化到簇中心的欧氏距离平方之和。

**通用性**：KMeans 是无监督聚类的通用算法，广泛用于原型学习、向量量化、意图发现。

**代码思路**：

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=K, n_init=10)
labels = kmeans.fit_predict(features)            # (N,)
centers = kmeans.cluster_centers_                # (K, D)
```

### 5.2 簇中心计算

$$\mathbf{c}_k = \frac{1}{|\mathcal{S}_k|} \sum_{\mathbf{h} \in \mathcal{S}_k} \mathbf{h}$$

**含义**：每个簇中心是簇内所有表示的均值。代码用 EMA 更新：`new_center = ema_decay * old_center + (1 - ema_decay) * batch_mean`。

**通用性**：EMA（指数移动平均）是深度学习训练中更新统计量的通用手段（如 BatchNorm 的 running_mean、MoCo 的动量编码器）。

**代码思路**：

```python
# 计算 batch 内每个簇的均值
for k in range(K):
    mask = (labels == k)
    if mask.sum() > 0:
        batch_mean = features[mask].mean(dim=0)          # (D,)
        # EMA 更新
        centers[k] = ema_decay * centers[k] + (1 - ema_decay) * batch_mean
```

---

## 6. 扩散模型基础（通用经典）

> 以下公式是标准 DDPM 扩散模型的通用数学，任何扩散模型都适用。

### 6.1 前向扩散（闭式）

$$E_t = \sqrt{\bar{\alpha}_t} \cdot E_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**含义**：在噪声调度表 $\{\alpha_t\}$ 下，干净数据 $E_0$ 经过 $t$ 步扩散后变为带噪数据 $E_t$。$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$。

**通用性**：这是 DDPM 的闭式前向加噪公式，一步直接从 $E_0$ 跳到任意时刻 $t$，无需逐步迭代。

**代码思路**：

```python
def forward_diffusion(E_0, t, alphas_cumprod):
    # E_0: (B, T, D), t: (B,), alphas_cumprod: (Tmax,)
    sqrt_alpha = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1)      # (B,1,1)
    sqrt_one_minus = torch.sqrt(1. - alphas_cumprod[t]).view(-1, 1, 1)
    noise = torch.randn_like(E_0)                                   # (B,T,D)
    E_t = sqrt_alpha * E_0 + sqrt_one_minus * noise
    return E_t
```

### 6.2 线性噪声调度

$$\beta_t = \beta_{\text{start}} + \frac{t}{T_{\max} - 1} \cdot (\beta_{\text{end}} - \beta_{\text{start}})$$

$$\alpha_t = 1 - \beta_t$$

**含义**：从 $\beta_1 = 1e^{-4}$ 线性增长到 $\beta_{50} = 0.02$，$T_{\max} = 50$。线性调度是扩散模型中最简单的一种噪声调度方式（另有余弦调度、二次调度等）。

**通用性**：噪声调度是扩散模型的通用组件，决定加噪速度。

**代码思路**：

```python
def get_noise_schedule(T_max=50, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, T_max)   # (Tmax,)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)         # ᾱ_t
    return betas, alphas, alphas_cumprod
```

---

## 7. Gumbel-Softmax 与熵正则（通用经典）

### 7.1 Gumbel-Softmax（可微离散采样）

$$\pi_i = \frac{\exp\left((\log \alpha_i + g_i) / \tau\right)}{\sum_{j=1}^{M} \exp\left((\log \alpha_j + g_j) / \tau\right)}$$

**含义**：用 Gumbel-Softmax 得到离散 one-hot 分布 $\pi$ 作为软变量，$g_i \sim \text{Gumbel}(0,1)$ 是重参数化噪声，$\tau$ 是温度参数。

**通用性**：Gumbel-Softmax 是深度学习中可微离散采样的通用技巧，用于 VAE 离散隐变量、MoE 路由、NAS 架构搜索等。

**简化版本**：若去掉 Gumbel 噪声和温度机制，等价于 $\tau=1$ 的普通 softmax：

$$\pi_i = \frac{\alpha_i}{\sum_j \alpha_j}$$

**代码思路**：

```python
# 完整 Gumbel-Softmax(PyTorch 内置)
pi = F.gumbel_softmax(logits, tau=0.5, hard=True)   # one-hot,可微

# 或手动实现
def gumbel_softmax(logits, tau=0.5):
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y = F.softmax((logits + g) / tau, dim=-1)
    # straight-through:前向 one-hot,反向软概率
    index = y.argmax(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(y).scatter_(-1, index, 1.0)
    return y_hard - y.detach() + y

# 简化版(普通 softmax)
pi = F.softmax(logits, dim=-1)
```

### 7.2 探索损失（负熵）

$$L_{\text{expl}} = -\frac{1}{B} \sum_{u=1}^B \sum_{i=1}^M \pi_{u,i} \ln \pi_{u,i} = -\mathbb{H}(\pi)$$

**含义**：鼓励输出**均匀分布**（最大熵），避免总是选择同一个选项。

**通用性**：熵正则是强化学习 / 门控网络 / MoE 路由中的通用探索手段，防止坍缩到单一选择。

**代码思路**：

```python
# 负熵:熵越大,loss 越小,鼓励均匀分布
entropy = -torch.sum(pi * torch.log(pi + 1e-8), dim=-1)   # (B,)
exploration_loss = -entropy.mean()                         # 标量
```

---

# 第二部分：论文特殊公式

> 以下公式是具体论文中特有的设计，**按论文排序（从 §1 开始）**，每篇论文内部的公式**各自从 (1) 开始排序**。
>
> 命名约定：公式编号采用「论文节号 + 论文内公式号」，如 §1-(3) 表示第 1 篇论文的第 (3) 个公式。

---

## 1. ADARec（自适应扩散增强推荐）

> **论文核心**：用控制器动态混合不同深度的 Transformer 输出，结合扩散增强与混合专家（MoE），从单条稀疏序列重建用户意图层次。

### 1.0 符号约定

| 符号 | 含义 | CLI 默认值 |
| --- | --- | --- |
| $B$ | batch_size | 256 |
| $T$ | max_seq_length | 50（若 shorten_seq_to 则 = shorten+3） |
| $D$ | hidden_size | 128 |
| $H$ | num_attention_heads | 2 |
| $L$ | head_size = $D/H$ | 64 |
| $K$ | num_intent_clusters | 512（Toys/Sports）/ 256（Yelp/Beauty/ml-1m） |
| $M$ | controller_num_levels | 2（硬编码） |
| $T_{\max}$ | diffusion_t_max | 50 |
| $E^{0}_{u}$ | 用户序列嵌入 | $\mathbb{R}^{L \times D}$ |
| $V_{l_u}$ | 归一化序列长度嵌入 | 原文未给明确形状，仅称向量 |
| $T_u$ | 选定离散扩散深度 | 整数 |
| $\tau_{gs}$ | Gumbel-Softmax 温度 | 0.5（论文） |

---

### 1.1 ADC（自适应深度控制器）

#### §1-(1) 上下文编码

$$h_c = \text{Pool}(\text{GRU}([E^{0}_{u}; V_{l_u}]))$$

- **符号**：$E^{0}_{u}$（用户输入序列嵌入，$\mathbb{R}^{L \times D}$）；$V_{l_u}$（归一化序列长度嵌入）；GRU 捕获时序依赖；Pool 池化。
- **形状**：$h_c$ 为 GRU 池化后的上下文向量（原文未给维度，推断为 $\mathbb{R}^{D}$ 或 controller 隐藏维）。
- **运算**：拼接 $[E^{0}_{u}; V_{l_u}]$ → GRU → Pool。
- **输出**：$h_c$（序列级上下文表征）。
- **作用**：为自适应深度选择提供序列级上下文。
- **代码思路**：

```python
class AdaptiveDepthController(nn.Module):
    def __init__(self, hidden_size):
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.length_embedding = nn.Embedding(512, hidden_size)

    def forward(self, sequence_emb, sequence_lengths):
        # sequence_emb: (B, T, D)
        gru_out, _ = self.gru(sequence_emb)              # (B, T, D)
        fused = gru_out[:, -1, :]                        # (B, D) 取最后隐状态

        lengths_clipped = torch.clamp(sequence_lengths, max=511)
        length_emb = self.length_embedding(lengths_clipped)  # (B, D)
        return fused, length_emb
```

#### §1-(2) 深度 logits

$$\text{logits}_{u} = W_{\sigma} \cdot h_c + b_{\sigma}$$

- **符号**：$W_{\sigma}$、$b_{\sigma}$ 为可学习参数；$h_c$ 为 §1-(1) 输出。
- **运算**：线性变换。
- **输出**：$\text{logits}_{u}$（跨 $0,\dots,T_{\max}$ 共 $T_{\max}+1$ 个可能深度的 logits）。
- **作用**：为深度选择提供各深度打分。
- **代码思路**：

```python
# 拼接序列上下文与长度嵌入
concat = torch.cat([fused, length_emb], dim=-1)      # (B, 2D)
logits = self.output_layer(concat)                   # (B, num_levels) = (B, 2)
```

#### §1-(3) Gumbel-Softmax 深度选择

$$p_u = \text{Softmax}((\text{logits}_u + g) / \tau_{gs})$$

- **符号**：$g$ 为来自 $\text{Gumbel}(0,1)$ 的 i.i.d. 样本；$\tau_{gs}$ 为 softmax 温度。
- **运算**：前向取 argmax 得离散深度 $T_u$，反向用软概率 $p_u$（Gumbel-Softmax 估计器）。
- **输出**：$p_u$（$T_{\max}+1$ 维深度选择概率分布）；$T_u$（选定的离散深度）。
- **作用**：在 $T_{\max}+1$ 个离散深度中软/硬选择增强深度，实现自适应深度控制且前向可微。
- **代码思路**：

```python
# 完整 Gumbel-Softmax(论文描述)
p_u = F.gumbel_softmax(logits, tau=0.5, hard=True)  # one-hot,可微

# 代码实际实现(普通 softmax,无 Gumbel 噪声)
probs = F.softmax(logits, dim=-1)                    # (B, 2)
```

---

### 1.2 HDA（分层扩散增强）

#### §1-(4) 前向加噪

$$E^{k}_{u} = \sqrt{\bar{\alpha}_{k}} \cdot E^{0}_{u} + \sqrt{1 - \bar{\alpha}_{k}} \cdot \epsilon_{k}$$

- **符号**：$\bar{\alpha}_{k}$ 为 DDPM 累积保留系数（线性方差调度）；$E^{0}_{u}$（干净序列嵌入 $\mathbb{R}^{L \times D}$）；$\epsilon_{k} \sim \mathcal{N}(0, I)$（与 $E^{0}_{u}$ 同形状的标准高斯噪声）。
- **形状**：$E^{k}_{u}$ 与 $E^{0}_{u}$ 同形，$\mathbb{R}^{L \times D}$。
- **运算**：闭式前向加噪。
- **输出**：$E^{k}_{u}$（第 $k$ 步噪声表征）。
- **作用**：构造从干净到高度噪声的扩散视图，为去噪/意图层次提供多噪声水平输入。
- **代码思路**：

```python
def forward_diffusion(E_0, t, alphas_cumprod):
    sqrt_alpha = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1)
    sqrt_one_minus = torch.sqrt(1. - alphas_cumprod[t]).view(-1, 1, 1)
    noise = torch.randn_like(E_0)
    E_t = sqrt_alpha * E_0 + sqrt_one_minus * noise
    return E_t
```

#### §1-(5) 反向去噪

$$\hat{E}^{k}_{u} = \frac{\sqrt{1 - \bar{\alpha}_{k}} \cdot \epsilon_{\theta}(E^{k}_{u}, k)}{\sqrt{\bar{\alpha}_{k}}}$$

- **符号**：$\epsilon_{\theta}$ 为可学习去噪网络（原文指明实现为 MLP）；$k$ 为噪声水平（同时作条件输入）。
- **形状**：$\hat{E}^{k}_{u}$ 与 $E^{k}_{u}$ 同形，$\mathbb{R}^{L \times D}$。
- **运算**：一步反向去噪。
- **输出**：$\hat{E}^{k}_{u}$（第 $k$ 步去噪表征）。
- **作用**：从各噪声步重建意图表征——高 $k$ → 粗粒度意图（信息被压缩，$I(\hat{E}_k; E_u) \ll H(E_u)$）；低 $k$ → 细粒度意图（信息近全保留，$I(\hat{E}_k; E_u) \approx H(E_u)$）；整条轨迹构成意图层次 $A_u$。
- **代码思路**：

```python
# 论文描述:去噪网络预测噪声,再重建
class DenoisingNetwork(nn.Module):
    def __init__(self, hidden_size):
        self.net = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, E_k, k):
        # E_k: (B, T, D), k: (B,)
        k_emb = k.float().unsqueeze(-1).unsqueeze(-1).expand(-1, E_k.size(1), 1)
        x = torch.cat([E_k, k_emb], dim=-1)
        eps_pred = self.net(x)                          # (B, T, D)
        return eps_pred

# 重建:ê_k = sqrt(1-ᾱ_k) * ε_θ / sqrt(ᾱ_k)
eps_pred = denoise_net(E_k, k)
E_hat = torch.sqrt(1 - alphas_cumprod[k]).view(-1,1,1) * eps_pred \
        / torch.sqrt(alphas_cumprod[k]).view(-1,1,1)

# 代码实际实现:无去噪网络,加噪后直接过 Transformer 再编码
# final_sequence_emb = sum(weighted_diffused_embs)
# adaptive_output = item_encoder(final_sequence_emb)
```

---

### 1.3 HP-MoE（分层解析混合专家）

#### §1-(6) 粗粒度对比损失

$$\mathcal{L}_{\text{cont}} = \sum_{k = \lfloor T_u/2 \rfloor}^{T_u - 1} - \log \left[ \frac{ \exp(\text{sim}(z_{u,k}, z_{u,k+1}) / \tau) }{ \sum_{z_j \in \mathcal{B}_k} \exp(\text{sim}(z_{u,k}, z_j) / \tau) } \right]$$

- **符号**：$z_{u,k}$（粗粒度表征）；$z_{u,k+1}$ 为正样本；$\mathcal{B}_k$ 含 $z_{u,k+1}$ 与同批其他用户的粗粒度表征（负样本）；$\tau$ 温度；$k$ 范围 $\lfloor T_u/2 \rfloor,\dots,T_u-1$。
- **运算**：基于 InfoNCE 的排斥性对比损失（repulsive contrastive loss）。
- **输出**：标量损失。
- **作用**：强制粗粒度专家学习"用户不变本质"（user-invariant essence），使不同粗粒度层次一致、并与同批其他用户区分。
- **代码思路**：

```python
def repulsive_contrastive_loss(z_k, z_k1, z_others, tau=0.07):
    # z_k, z_k1: (B, D) 同一用户相邻粗粒度层
    # z_others: (B, B, D) 或 (B, D) 同批其他用户的粗粒度表征
    z_k = F.normalize(z_k, dim=-1)
    z_k1 = F.normalize(z_k1, dim=-1)
    pos_sim = torch.sum(z_k * z_k1, dim=-1) / tau       # (B,)
    neg_sim = torch.matmul(z_k, z_others.T) / tau       # (B, B)
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    labels = torch.zeros(z_k.size(0), dtype=torch.long, device=z_k.device)
    return F.cross_entropy(logits, labels)
```

#### §1-(7) 内容感知门控

$$g_k = \text{Sigmoid}(\text{MLP}([\text{Pool}(\hat{E}^{k}_{u}); \text{emb}(k)]))$$

- **符号**：$\text{Pool}(\hat{E}^{k}_{u})$（第 $k$ 层去噪表征的池化，即内容表征）；$\text{emb}(k)$（噪声水平嵌入）；MLP 多层感知机。
- **形状**：$g_k$ 原文未明确给出（推断为逐层次标量或向量，取值 $[0,1]$）。
- **运算**：拼接 → MLP → Sigmoid。
- **输出**：$g_k$（第 $k$ 层门控概率）。
- **作用**：内容感知路由——动态决定每个层次 $k$ 偏向细粒度专家（$g_k$ 大）还是粗粒度专家（$g_k$ 小）。
- **代码思路**：

```python
class ContentAwareRouter(nn.Module):
    def __init__(self, hidden_size):
        self.level_emb = nn.Embedding(T_max, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, E_hat_k, k):
        # E_hat_k: (B, T, D)
        pooled = E_hat_k.mean(dim=1)                    # (B, D) 内容表征
        k_emb = self.level_emb(k)                       # (B, D) 噪声水平嵌入
        concat = torch.cat([pooled, k_emb], dim=-1)     # (B, 2D)
        g_k = torch.sigmoid(self.mlp(concat))           # (B, 1)
        return g_k
```

#### §1-(8) 聚合细粒度表征

$$z_{\text{fin}} = \frac{ \sum_{k=0}^{T_u} g_k \cdot E_{\text{fin}}(\hat{E}^{k}_{u}) }{ \sum_{k=0}^{T_u} g_k + \varepsilon }$$

- **符号**：$g_k$（§1-(7) 输出）；$E_{\text{fin}}$（细粒度专家，Transformer）；$\hat{E}^{k}_{u}$（HDA 输出）；$\varepsilon$ 数值稳定项。
- **运算**：按门控对各层细粒度专家输出加权聚合。
- **输出**：$z_{\text{fin}}$（聚合细粒度表征）。
- **作用**：强调内容相关层次。
- **代码思路**：

```python
# 逐层过细粒度专家,按 g_k 加权
weighted_sum = 0.0
weight_total = 0.0
for k in range(T_u + 1):
    expert_out = E_fin(E_hat[k])                    # (B, T, D)
    weighted_sum += g_k[k] * expert_out
    weight_total += g_k[k]
z_fin = weighted_sum / (weight_total + 1e-8)
```

#### §1-(9) 聚合粗粒度表征

$$z_{\text{coar}} = \frac{ \sum_{k=0}^{T_u} (1 - g_k) \cdot E_{\text{coar}}(\hat{E}^{k}_{u}) }{ \sum_{k=0}^{T_u} (1 - g_k) + \varepsilon }$$

- **符号**：同 §1-(8)，权重用 $(1 - g_k)$；$E_{\text{coar}}$（粗粒度专家）。
- **运算**：按反向门控聚合各层粗粒度专家输出。
- **输出**：$z_{\text{coar}}$（聚合粗粒度表征）。
- **作用**：按反向门控聚合各层粗粒度专家输出。
- **代码思路**：

```python
weighted_sum = 0.0
weight_total = 0.0
for k in range(T_u + 1):
    expert_out = E_coar(E_hat[k])
    weighted_sum += (1 - g_k[k]) * expert_out
    weight_total += (1 - g_k[k])
z_coar = weighted_sum / (weight_total + 1e-8)
```

#### §1-(10)(11) 最终融合

$$\text{gate}_u = \text{Sigmoid}(W_g \cdot z_{\text{coar}} + b_g)$$

$$h_u = \text{gate}_u \odot z_{\text{fin}} + (1 - \text{gate}_u) \odot z_{\text{coar}}$$

- **符号**：$\text{gate}_u$（§1-(10)，由粗粒度聚合表征算最终门控）；$z_{\text{fin}}$、$z_{\text{coar}}$（§1-(8)(9) 输出）；$\odot$ 逐元素乘。
- **运算**：先算最终门控，再逐元素加权融合。
- **输出**：$h_u$（最终融合用户表征，送预测层）。
- **作用**：平衡稳定的长期主题（粗粒度，$z_{\text{coar}}$）与瞬时细节（细粒度，$z_{\text{fin}}$），得到综合用户表征。
- **代码思路**：

```python
# 论文描述:门控融合
gate_u = torch.sigmoid(self.gate_layer(z_coar))     # (B, D)
h_u = gate_u * z_fin + (1 - gate_u) * z_coar        # (B, D)

# 代码实际实现:两专家预测等权平均
pred_fused = (pred_e5 + pred_e10) / 2.0             # (B, num_items)
```

---

### 1.4 总损失

#### §1-(12)(13)

$$\mathcal{L}_{\text{tot}} = \mathcal{L}_{\text{rec}} + \lambda_{\text{deno}} \cdot \mathcal{L}_{\text{deno}} + \lambda_{\text{cont}} \cdot \mathcal{L}_{\text{cont}} + \lambda_{\text{expl}} \cdot \mathcal{L}_{\text{expl}}$$

$$\mathcal{L}_{\text{rec}} = - \sum_{i \in \mathcal{I}} p(i) \log(y_{u,i})$$

- **符号**：$\mathcal{L}_{\text{rec}}$（推荐损失，§1-(13)）；$\mathcal{L}_{\text{deno}}$（去噪 MSE）；$\mathcal{L}_{\text{cont}}$（§1-(6) 对比）；$\mathcal{L}_{\text{expl}}$（探索负熵）；$\lambda$ 加权系数。
- **运算**：§1-(12) 为四项加权和；§1-(13) 为标准交叉熵。
- **输出**：标量总损失。
- **作用**：联合优化推荐精度、意图层次重建质量、粗粒度不变性、深度探索多样性。
- **代码思路**：

```python
# 推荐损失:交叉熵
rec_loss = F.cross_entropy(rating_pred, target)     # 或 BPR 变体

# 总损失
total_loss = (rec_loss
              + self.diffusion_weight * reconstruction_loss
              + self.exploration_weight * exploration_loss
              + self.cf_weight * cl_loss)
total_loss.backward()
optimizer.step()
```

#### §1-(14) 探索损失（负熵）

$$\mathcal{L}_{\text{expl}} = - \sum_{i} p_{u,i} \log p_{u,i}$$

- **符号**：$p_{u,i}$ 为 ADC 输出的深度选择概率（§1-(3) 的 $p_u$ 分量）。
- **运算**：负熵。
- **输出**：标量损失。
- **作用**：防止 ADC 坍缩到只选浅深度，鼓励深度选择多样性。
- **代码思路**：

```python
if probs is not None:
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # (B,)
    exploration_loss = -torch.mean(entropy)                         # 标量
```

#### §1-(15) 去噪损失（噪声预测 MSE）

$$\mathcal{L}_{\text{deno}} = \| \epsilon_{\theta}(E^{k}_u, k) - \epsilon_k \|^2$$

- **符号**：$\epsilon_{\theta}$ 为去噪网络；$E^{k}_u$ 为第 $k$ 步噪声表征；$\epsilon_k$ 为对应时刻采样噪声。
- **运算**：MSE。
- **输出**：标量损失。
- **作用**：优化可学习去噪网络 $D_{\theta}$ 准确估计噪声，支撑 HDA 意图层次重建质量。
- **代码思路**：

```python
# 论文描述:噪声预测 MSE
eps_pred = denoise_net(E_k, k)
deno_loss = F.mse_loss(eps_pred, eps_k)

# 代码实际实现:表示重构 MSE(adaptive 逼近 raw)
deno_loss = F.mse_loss(adaptive_output, raw_output.detach())
```

---

### 1.5 ADARec 公式速查表

| 编号 | 模块 | 公式 | 作用 |
| --- | --- | --- | --- |
| §1-(1) | ADC | $h_c = \text{Pool}(\text{GRU}([E^{0}_{u}; V_{l_u}]))$ | 序列级上下文 |
| §1-(2) | ADC | $\text{logits}_{u} = W_{\sigma} \cdot h_c + b_{\sigma}$ | 各深度打分 |
| §1-(3) | ADC | $p_u = \text{Softmax}((\text{logits}_u + g) / \tau_{gs})$ | 深度选择 |
| §1-(4) | HDA | $E^{k}_{u} = \sqrt{\bar{\alpha}_{k}} E^{0}_{u} + \sqrt{1 - \bar{\alpha}_{k}} \epsilon_{k}$ | 前向加噪 |
| §1-(5) | HDA | $\hat{E}^{k}_{u} = \sqrt{1 - \bar{\alpha}_{k}} \epsilon_{\theta}(E^{k}_{u}, k) / \sqrt{\bar{\alpha}_{k}}$ | 反向去噪 |
| §1-(6) | HP-MoE | $\mathcal{L}_{\text{cont}} = \sum_k -\log[\cdots]$ | 粗粒度对比 |
| §1-(7) | HP-MoE | $g_k = \text{Sigmoid}(\text{MLP}([\text{Pool}(\hat{E}^{k}_{u}); \text{emb}(k)]))$ | 内容感知门控 |
| §1-(8) | HP-MoE | $z_{\text{fin}} = \sum_k g_k E_{\text{fin}} / \sum_k g_k$ | 细粒度聚合 |
| §1-(9) | HP-MoE | $z_{\text{coar}} = \sum_k (1-g_k) E_{\text{coar}} / \sum_k (1-g_k)$ | 粗粒度聚合 |
| §1-(10) | HP-MoE | $\text{gate}_u = \text{Sigmoid}(W_g z_{\text{coar}} + b_g)$ | 最终门控 |
| §1-(11) | HP-MoE | $h_u = \text{gate}_u \odot z_{\text{fin}} + (1-\text{gate}_u) \odot z_{\text{coar}}$ | 最终融合 |
| §1-(12) | 总损失 | $\mathcal{L}_{\text{tot}} = \mathcal{L}_{\text{rec}} + \lambda_{\text{deno}}\mathcal{L}_{\text{deno}} + \lambda_{\text{cont}}\mathcal{L}_{\text{cont}} + \lambda_{\text{expl}}\mathcal{L}_{\text{expl}}$ | 联合优化 |
| §1-(13) | 总损失 | $\mathcal{L}_{\text{rec}} = -\sum_i p(i) \log(y_{u,i})$ | 推荐交叉熵 |
| §1-(14) | 探索 | $\mathcal{L}_{\text{expl}} = -\sum_i p_{u,i} \log p_{u,i}$ | 负熵探索 |
| §1-(15) | 去噪 | $\mathcal{L}_{\text{deno}} = \|\epsilon_{\theta}(E^{k}_u, k) - \epsilon_k\|^2$ | 噪声预测 MSE |

---

### 1.6 ADARec 公式与代码对照

| 编号 | 论文公式 | 代码位置 | 一致性 |
| --- | --- | --- | --- |
| §1-(1) | $h_c = \text{Pool}(\text{GRU}([E^{0}_{u}; V_{l_u}]))$ | `modules.py:AdaptiveDepthController.forward` | ✅ 一致 |
| §1-(2) | $\text{logits}_{u} = W_{\sigma} h_c + b_{\sigma}$ | 同上 `output_layer` | ✅ 一致 |
| §1-(3) | Gumbel-Softmax | `models.py:forward` L308 | ⚠️ 代码用普通 softmax，无 Gumbel 噪声与温度 |
| §1-(4) | 前向加噪 | `diffusion_utils.py:forward_diffusion` | ✅ 一致 |
| §1-(5) | 反向去噪 | `models.py:forward` L181~194 | ⚠️ 代码无独立去噪网络，加噪后直接再编码 |
| §1-(6) | 粗粒度对比损失 | `modules.py:PCLoss` / `trainers.py` L675 | ⚠️ 代码为 InstanceCL + IntentCL，语义有偏差 |
| §1-(7) | 内容感知门控 | — | ⚠️ 代码无此模块 |
| §1-(8) | 细粒度聚合 | `models.py:forward` L176~194 | ⚠️ 代码为两深度 softmax 加权 |
| §1-(9) | 粗粒度聚合 | 同上 | ⚠️ 同上 |
| §1-(10) | 最终门控 | — | ⚠️ 代码无此模块 |
| §1-(11) | 最终融合 | `trainers.py` L838 | ⚠️ 代码为固定 0.5/0.5 等权平均 |
| §1-(12) | 总损失 | `trainers.py:_train_one_expert` L657~675 | ✅ 加权结构一致 |
| §1-(13) | 推荐交叉熵 | `trainers.py` L527~531 | ✅ 一致（BPR 变体） |
| §1-(14) | 探索损失 | `trainers.py` L660~662 | ✅ 一致（负熵） |
| §1-(15) | 去噪损失 | `trainers.py` L660 | ⚠️ 代码为表示重构 MSE，非噪声预测 |

> **说明**：代码与论文存在 4 处主要语义偏差——§1-(3) 用普通 softmax 替代 Gumbel-Softmax、§1-(5) 省略去噪网络、§1-(7)(10) 无门控路由、去噪损失为表示重构而非噪声预测。详见 ADARec - 通过自适应扩散与混合专家缓解意图坍塌.md §D 论文主张 vs 代码实现。

---
