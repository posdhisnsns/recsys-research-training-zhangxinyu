# Python 学习笔记 — ADARec 代码中的语法与技巧

> 本文件记录 ADARec 代码（`experiment/ADARec/src/`）涉及的 Python / PyTorch 语法点，来源为实际代码行号，供对照查阅。

---

## 1. PyTorch 基础

### 1.1 `nn.Parameter`

```python
# modules.py L48
self.length_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
```

- `nn.Parameter(tensor)` 是 `torch.Tensor` 的子类，自动注册为模型可学习参数（自动加入 `model.parameters()`）。
- 与普通 `self.xxx = torch.zeros(...)` 的区别：后者不会自动被 `parameters()` 收录，须手动 `register_buffer`。

### 1.2 `F.pad` — 序列填充

```python
# modules.py L73-76
input_mask = None
if max_len > seq_len:
    input_mask = F.pad(
        torch.zeros((seq_emb.size(0), max_len - seq_len), device=seq_emb.device, dtype=torch.long),
        (0, 0), "constant", 1
    )
input_mask = torch.cat([input_mask, input_mask], dim=-1)  # L76
```

- `F.pad(input, pad, mode="constant", value=0)`：`pad` 是 (左, 右, 上, 下) 元组，对于 1D tensor `(L,)` → pad=(左, 右)；对于 2D `(B, L)` → pad=(左, 右)。
- `mode="constant"` 用常数填充；`value` 指定填充值。
- 此处 `dtype=torch.long` 决定 padding token id=0。

### 1.3 `torch.gather` — 按索引收集

```python
# modules.py L101-102
item_emb = self.item_embeddings(input_ids)
sequence_emb = torch.gather(item_emb, dim=1, index=position_ids.unsqueeze(-1).expand_as(item_emb))
```

- `gather(input, dim, index)`：沿 `dim` 方向，用 `index` 中的索引值从 `input` 中取值。
- 此处 `index=position_ids.unsqueeze(-1).expand_as(item_emb)` 得到形状 `(B, T, D)`，在 `item_emb` 中按位置 id 取对应列，实现"将 position_id 转成位置向量"。

### 1.4 `F.dropout` 与 `self.training`

```python
# modules.py L105-106
if self.dropout_rate > 0:
    sequence_emb = self.dropout(F.pad(sequence_emb, ...))
```

- `nn.Dropout` / `F.dropout`：在 `self.training=True` 时随机置零，在 `eval()` 时关闭（所有元素乘 `(1-p)` 变为确定性缩放）。
- ADARec 中 `training=False`（如 `model.eval()`）时 dropout 关闭，扩散增强分支（`adaptive_output`）在 `training=False` 时被跳过。

### 1.5 `torch.arange` + `unsqueeze`

```python
# modules.py L78-79
position_ids = torch.arange(max_len, dtype=torch.long, device=seq_emb.device)
position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
```

- `torch.arange(start, end, step, device=...)`：在指定设备上生成一维等差张量。
- `unsqueeze(0)` 在第 0 维加 batch 维 → `(1, max_len)`；`expand_as(input_ids)` 广播到 `(B, T)`。

### 1.6 `register_buffer`

```python
# modules.py L39-40
self.register_buffer("partial_seq_emb", torch.zeros(1, 1, self.hidden_size))
self.register_buffer("attn_mask", torch.zeros(1, 1))
```

- `register_buffer(name, tensor)`：将 tensor 注册为模型成员，**不是**可学习参数，但会随 `model.to(device)` / `model.state_dict()` / `model.eval()` 自动迁移/切换（与 `nn.Parameter` 的区别）。
- 用途：持久化非参数状态（如 `running_mean`、look-up 表）。

---

## 2. 列表 / 字典操作

### 2.1 列表索引与 `append`

```python
# trainers.py L825
for i, expert_model in enumerate([self.model_e5, self.model_e10]):
    if not stopped_list[i]:
        loss = expert_model(...)[0]  # forward 返回 (loss, output)
```

- `enumerate(list)` 同时得到索引 `i` 和值，配合 `stopped_list[i]` 实现"按位跳过"。

### 2.2 列表推导式过滤

```python
# trainers.py L661
valid_experts = [expert for expert in ["e5", "e10"] if not getattr(self, f"{expert}_stopped")]
```

- `getattr(self, "e5_stopped")` 动态读取属性。
- `["e5", "e10"]` → `f"{expert}_stopped"` → `"e5_stopped"` / `"e10_stopped"`。

### 2.3 字典构造（Python 3.7+ 有序）

```python
# trainers.py L895-901
s_e5 = self.last_scores_e5
log_line = {
    f"{'Epoch': {epoch}, "
    f"'HIT@5_e5': '{s_e5[0]:.4f}', ...
}
```

- Python 3.7+ 字典保持插入顺序，此处用于构造结构化日志行。

### 2.4 `dict.get` 带默认值

```python
# main.py L102
lr = float(args.learning_rate) if args.learning_rate else self.default_lr
```

- `args.learning_rate` 为 `None` 时返回 `None`，`float(None)` 报错；用 `if-else` 兜底而非 `dict.get`（因为 argparse 属性不存在会抛 AttributeError）。

---

## 3. NumPy 相关

### 3.1 `numpy.array` 与 `[-1:]` 索引

```python
# utils.py L822
self.early_stopper_e10(np.array(scores_e10[-1:]), self.model_e10)
```

- `scores_e10` 是 `[HIT@5, NDCG@5, HIT@20, NDCG@20]` 列表。
- `scores_e10[-1:]` 取最后一个元素（`NDCG@20`），`np.array(...)` 包装成 shape `(1,)` 数组。
- `scores_e10[-1]` 与 `scores_e10[-1:]` 的区别：前者是标量，后者是 shape `(1,)` 的 1D 数组。

### 3.2 `numpy.concatenate`

```python
# trainers.py L655
np.concatenate([np.array(s) for s in scores])
```

- 将多个列表/数组沿轴 0 拼接，用于聚合多 expert 分数。

### 3.3 稀疏矩阵 `.nonzero()` + 索引

```python
# trainers.py L519
rating_pred[batch_train.nonzero()] = 0
```

- `rating_pred` 是形状 `(B, num_items)` 的密集张量。
- `batch_train.nonzero()` 返回 `(2, num_nonzero)` 的索引数组（行索引, 列索引）。
- 用密集索引 `rating_pred[batch_train.nonzero()]` 可一次写入多个位置（对应"将已交互物品分数清零"的推荐系统常用技巧）。
- 另一种写法：`rating_pred[batch_train.toarray().bool()] = 0`（触发 OOM 的旧写法）。

---

## 4. PyTorch 张量操作

### 4.1 `torch.clamp` — 数值截断

```python
# trainers.py L630
prob = torch.clamp(prob, min=1e-9)
```

- `torch.clamp(input, min, max)` 将 input 限制在 `[min, max]` 范围内，防止 log / 除零。

### 4.2 `torch.log` 与 `torch.exp`

```python
# trainers.py L631
prob = torch.clamp(prob, min=1e-9)
loss = -(torch.log(prob[..., 0]) + torch.log(prob[..., 1])) / 2
```

- `torch.log` 即 $\ln(x)$；`torch.exp` 即 $e^x$。
- 信息熵计算：`H = -sum(p * log(p))`。

### 4.3 `tensor.detach()` — 截断梯度

```python
# trainers.py L660
L_deno = self.diffusion_weight * F.mse_loss(adaptive.detach(), raw_output)
```

- `tensor.detach()` 返回一个与计算图分离的同值 tensor，**阻断梯度传播**。
- 用途：ADARec 中 `adaptive.detach()` 使 `L_deno` 不反向影响控制器（ADC），只更新主 Transformer。

### 4.4 `tensor.expand_as` — 广播扩展（惰性）

```python
# modules.py L83
input_mask = input_mask.unsqueeze(1).bool()
extended_attention_mask = (1.0 - input_mask.unsqueeze(1).expand_as(item_emb)) * self.negtive_inf
```

- `tensor.expand_as(other)` 创建一个视图（不复制数据），形状扩展到与 `other` 一致；要求原维度能被 broadcast。
- 注意：`expand` 系列只生成视图，无法在原张量上写入。

### 4.5 `F.softmax` 与 `dim` 参数

```python
# models.py L308
probs = F.softmax(logits, dim=-1)  # 在最后一维做 softmax
```

- `dim=-1` 即最后一维；`F.softmax(logits, dim=-1)` 使所有通道和为 1。
- ADARec 的 ADC 控制器用 `F.softmax` 得到 `probs` 作为加权系数（**不是** Gumbel-Softmax，无噪声和重参数化）。

### 4.6 `F.mse_loss`

```python
# trainers.py L660
L_deno = self.diffusion_weight * F.mse_loss(adaptive.detach(), raw_output)
```

- `F.mse_loss(input, target, reduction='mean')`：默认返回 `mean((input - target)^2)`。
- ADARec 中 `adaptive.detach()` 来自扩散分支，`raw_output` 来自原始分支，两者做 MSE 即"去噪损失"。

### 4.7 `torch.nn.functional.pad`

```python
# modules.py L105
sequence_emb = self.dropout(F.pad(sequence_emb, (0, 0, 1, 0), "constant", 0))
```

- `F.pad(tensor, pad, mode, value)`：`pad`=(左, 右, 上, 下)，对应填充宽度。
- 此处 `(0, 0, 1, 0)` 表示在上面（序列最前面）加 1 个 padding token，下方不填，序列维度不变，只增加时间维长度。

---

## 5. Python 特殊语法

### 5.1 `*` 解包参数

```python
# trainers.py L625
loss = loss_fn(*expert_outputs)
```

- `*expert_outputs` 将列表/元组解包为多个独立参数传给函数。
- 配合 `enumerate`：`for i, x in enumerate(list)` 得到下标和值。

### 5.2 `isinstance` 类型检查

```python
# datasets.py L79
if isinstance(sample, dict):
    for key in sample:
        sample[key] = sample[key].to(self.device)
```

- `isinstance(obj, tuple)` 检查对象是否属于指定类型（或其子类），返回布尔值。

### 5.3 条件表达式（三元）

```python
# modules.py L105-106
if self.dropout_rate > 0:
    sequence_emb = self.dropout(F.pad(sequence_emb, ...))
```

- `if-else` 控制流，不是三元表达式；`dropout_rate > 0` 做守卫判断。

### 5.4 字符串格式化（f-string）

```python
# trainers.py L895-901
f"{'Epoch': {epoch}, ...}"  # 错误的 f-string 写法
f"Epoch: {epoch}, HIT@5_e5: {s_e5[0]:.4f}"  # 正确写法
```

- ⚠️ **常见错误**：在 f-string 嵌套字典字面量时，`f"{'key': value}"` 中的 `{}` 是字典字面量语法，与 f-string 冲突。正确做法：用普通字符串拼接或将字典构造移出 f-string。
- ADARec 代码日志行用 `f"..{epoch}.."` 而非字典格式。

### 5.5 `zip` 并行迭代

```python
# trainers.py L825
for i, expert_model in enumerate([self.model_e5, self.model_e10]):
    if not stopped_list[i]:
        ...
```

- `enumerate` 与列表索引配合，`zip` 用于同时遍历多个等长序列。
- `zip(stoppe5_list, stopped_list)` 适合并行迭代两个列表。

---

## 6. 调试与工具函数

### 6.1 `py_compile` — 无执行检查

```python
import py_compile
py_compile.compile("src/trainers.py", doraise=True)
```

- 只做语法检查，不执行代码；`doraise=True` 遇语法错误抛 `py_compile.PyCompileError`。
- ADARec 每次修改代码后用此验证语法正确。

### 6.2 `torch.no_grad()` — 推理上下文

```python
# trainers.py L886
with torch.no_grad():
    scores_e5, _ = self.iteration(epoch, self.eval_dataloader, ...)
```

- `torch.no_grad()` 禁用梯度计算，减少显存/内存，用于推理和评测。

### 6.3 `torch.cuda.is_available()` / `.device`

```python
# main.py L23
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# main.py L41
self.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
```

- `torch.cuda.is_available()` 检查 CUDA 是否可用。
- `torch.device` 指定张量/模型的目标设备。

### 6.4 `scipy.sparse.csr_matrix`

```python
# trainers.py L519（已废弃的 toarray 写法）
train_matrix = self.args.rating_matrix
rating_pred = train_matrix[batch_user_index].toarray()  # 旧写法，OOM 风险
# 正确写法（稀疏操作）：
rating_pred = np.zeros((batch_size, num_items))  # 用稀疏索引
rating_pred[batch_train.nonzero()] = 0
```

- `toarray()` 将稀疏矩阵转为密集 numpy 数组，对大矩阵内存消耗巨大。
- 正确做法：用稀疏索引直接操作，或保持稀疏格式。

---

## 7. 常见错误与规避

### 7.1 `NameError: name 'xxx' is not defined`

```python
# datasets.py（Bug #1，修复前）
index = self.test_neg_items[user_id]  # ❌ index 在此处未定义
# 修复后：
neg_items = self.test_neg_items[user_id]  # ✅
```

- 常见于循环/分支中引用了未声明的变量名。

### 7.2 `weights_only=False`（PyTorch 2.6+）

```python
# trainers.py L644
torch.load(resume_path, map_location=self.device, weights_only=False)
```

- PyTorch 2.6+ 默认 `weights_only=True`，但 resume snapshot 包含 numpy 数组，numpy 无法被 `weights_only` 处理，须显式设 `False`。

### 7.3 字典/列表索引越界

```python
# utils.py L822（Bug 修复前）
self.early_stopper_e10(np.array(scores_e10[-1:]), ...)  # scores_e10 是 [4元素] 列表
# 若 scores_e10 为 None（已停专家未评测），-1: 索引会抛 TypeError
# 修复：valid_epoch 对已停专家也执行评测（跳过早停器更新），保证 last_scores 始终有效
```

### 7.4 `torch.load` 缺后缀

```python
# main.py（Bug #5，修复前）
torch.load(checkpoint_prefix)  # ❌ 缺 _e5.pt / _e10.pt
# 修复后：
torch.load(checkpoint_prefix + "_e5.pt")
torch.load(checkpoint_prefix + "_e10.pt")
```

---

# 二、推荐系统 / AI 通用 Python 语法（对应 recsys.md）

> 以下语法点对应 `recsys.md` 知识体系，覆盖从数据处理 → 矩阵分解 → 图神经网络 → 评测指标 → 向量检索的推荐系统实现常用写法。

## 8. 稀疏矩阵与用户-物品交互矩阵

### 8.1 `scipy.sparse` 三种格式

```python
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix

# COO：三元组 (row, col, data)，适合从 (user, item) 列表构造
rows, cols, data = [0, 0, 1, 2], [1, 3, 2, 1], [1, 1, 1, 1]
mat_coo = coo_matrix((data, (rows, cols)), shape=(3, 5))

# CSR：压缩行存储，适合行切片、矩阵乘法
mat_csr = mat_coo.tocsr()
mat_csr[0]          # 取第 0 行的稀疏向量
mat_csr[0].nonzero()  # 第 0 行的非零列索引

# LIL：链表存储，适合逐元素增量构建（循环内赋值）
mat_lil = lil_matrix((100, 100))
mat_lil[5, 10] = 1.0
mat_lil = mat_lil.tocsr()  # 构建完成后转 CSR 加速运算
```

- **COO**（Coordinate）：适合一次性批量构造。
- **CSR**（Compressed Sparse Row）：适合「取某行」和「矩阵乘法」这类行操作。
- **LIL**（List of Lists）：适合循环内逐元素插入，之后转 CSR。

### 8.2 稀疏矩阵乘法与点积

```python
import numpy as np
# 用户矩阵 U: (n_users, k)，物品矩阵 V: (n_items, k)
# 交互矩阵 R ≈ U @ V.T
scores = U @ V.T          # 稠密打分矩阵 (n_users, n_items)

# 稀疏矩阵之间的乘法
R_hat = (U_sparse @ V_sparse.T).toarray()
```

- 交互矩阵通常极稀疏（99%+ 为空），存储时用 `csr_matrix` 而非稠密 `np.ndarray`，否则内存爆炸。

## 9. 相似度计算与矩阵分解

### 9.1 余弦相似度

```python
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(mat_csr)   # 物品-物品相似度矩阵

# 手动实现（等价）：先 L2 归一化再点积
from sklearn.preprocessing import normalize
mat_norm = normalize(mat_csr, norm='l2', axis=1)
sim = mat_norm @ mat_norm.T
```

### 9.2 矩阵分解（SVD / ALS）

```python
import numpy as np
from sklearn.decomposition import TruncatedSVD

# SVD：R ≈ U Σ Vᵀ
U, S, Vt = np.linalg.svd(R.toarray(), full_matrices=False)

# TruncatedSVD：只保留前 k 个奇异值（降维，适合大规模稀疏矩阵）
svd = TruncatedSVD(n_components=50)
U_low = svd.fit_transform(mat_csr)   # 用户低维表示

# ALS（交替最小二乘，implicit 库，隐式反馈）
import implicit
model = implicit.als.AlternatingLeastSquares(factors=50)
model.fit(mat_csr.T)   # 注意 implicit 以「物品×用户」为行
```

- **SVD**：显式评分矩阵分解。
- **ALS**：隐式反馈下交替固定一侧求解另一侧，能处理大规模且天然并行。
- **TruncatedSVD**：截断奇异值，是 SVD 的降维近似。

## 10. PyTorch 模型范式（深度推荐）

### 10.1 `nn.Embedding` 查表

```python
import torch.nn as nn
self.user_emb = nn.Embedding(num_users, hidden_size)
self.item_emb = nn.Embedding(num_items, hidden_size)

user_vec = self.user_emb(user_ids)   # (B, hidden)
item_vec = self.item_emb(item_ids)   # (B, hidden)
```

- `nn.Embedding` 是「索引 → 向量」的查表，等价于一个可学习的权重矩阵 `(vocab, dim)`。

### 10.2 `nn.DataLoader` + `Dataset`

```python
from torch.utils.data import Dataset, DataLoader

class RecDataset(Dataset):
    def __init__(self, users, items, labels):
        self.users, self.items, self.labels = users, items, labels
    def __len__(self): return len(self.users)
    def __getitem__(self, i):
        return self.users[i], self.items[i], self.labels[i]

loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=4)
for users, items, labels in loader:
    ...
```

- `num_workers` 并行加载加速，Windows 下多进程需包在 `if __name__ == '__main__':` 里。

### 10.3 `forward` 与 `training` 分支

```python
class Model(nn.Module):
    def forward(self, x):
        if self.training:
            x = self.dropout(x)   # 只在训练时 dropout
        return x
```

- `self.training` 随 `model.train()` / `model.eval()` 自动切换，控制 dropout、BN 等训练/推理差异。

## 11. 图神经网络（推荐用）

### 11.1 邻接矩阵与度归一化

```python
import numpy as np
# 用户-物品二部图邻接矩阵 A
A = np.zeros((n_nodes, n_nodes))
# ... 填入边 ...

# 对称归一化 D^{-1/2} A D^{-1/2}（GCN 经典）
D = np.diag(1.0 / np.sqrt(A.sum(axis=1)))
A_norm = D @ A @ D
```

### 11.2 消息传递（简化 LightGCN 一步）

```python
# 邻域聚合：新表示 = 聚合邻居表示
for layer in range(n_layers):
    E = A_norm @ E   # E: (n_nodes, dim)，一次矩阵乘法即一层传播
final = sum(E_layers) / n_layers   # LightGCN 用各层平均
```

- LightGCN 的核心就是「去掉特征变换和非线性激活，只保留邻域聚合」，聚合用稀疏矩阵乘法即可表达。

## 12. 评测指标实现

### 12.1 Recall@K / Hit@K

```python
def recall_at_k(ranked_items, true_item, k):
    return int(true_item in ranked_items[:k])

def hit_at_k(ranked_items, true_items, k):
    # true_items 可为单个或多个
    return int(len(set(ranked_items[:k]) & set(true_items)) > 0)
```

### 12.2 NDCG@K

```python
import numpy as np
def ndcg_at_k(ranked_items, true_item, k):
    if true_item not in ranked_items[:k]:
        return 0.0
    rank = ranked_items[:k].index(true_item) + 1   # 位置从 1 开始
    dcg = 1.0 / np.log2(rank + 1)
    idcg = 1.0   # 单目标时理想 DCG = 1（目标排第 1）
    return dcg / idcg
```

- **位置折扣**：`1/log2(rank+1)`，排名越靠前贡献越大。
- **IDCG**（理想 DCG）：目标排在第 1 位的 DCG；单目标场景恒为 1，多目标需按理想排序计算。

### 12.3 高效批量计算（用矩阵）

```python
# 全量排序：给每个用户对所有物品打分，取 top-k
scores = model.get_all_scores()          # (n_users, n_items)
scores[mask_train] = -np.inf             # 屏蔽已交互物品（防止占用排名）
topk = np.argpartition(-scores, k, axis=1)[:, :k]  # 每行取最大的 k 个
```

- `np.argpartition` 是部分排序，比 `np.argsort` 快（只需前 k 个有序）。

## 13. 负采样与数据划分

### 13.1 随机负采样

```python
import numpy as np
def sample_negatives(n_items, n_neg, exclude):
    negs = set()
    while len(negs) < n_neg:
        c = np.random.randint(1, n_items)   # 0 通常为 padding
        if c not in exclude:
            negs.add(c)
    return list(negs)
```

- 需排除已交互物品（`exclude`），否则把正样本误当负样本。

### 13.2 `torch.multinomial` 按流行度采样

```python
torch.multinomial(item_popularity, num_samples, replacement=True)
```

- `item_popularity` 为归一化的流行度权重，实现「热门物品更可能被采为负例」的流行度采样。

### 13.3 时序划分（避免时间泄漏）

```python
# 按时间排序后，最后一条做测试、倒数第二条做验证、其余训练
seq_sorted = sorted(seq, key=lambda x: x['time'])
train, valid, test = seq_sorted[:-2], [seq_sorted[-2]], [seq_sorted[-1]]
```

- **时间泄漏**：随机打散会让「未来行为」进入训练，指标虚高。必须按时间顺序划分。

## 14. 向量检索（ANN）

### 14.1 FAISS 索引

```python
import faiss
import numpy as np
dim = 64
index = faiss.IndexFlatL2(dim)      # 精确 L2 检索
index = faiss.IndexFlatIP(dim)      # 内积（配合 L2 归一化 = 余弦）
index.add(item_vectors)              # 加入物品向量

D, I = index.search(query_vectors, k=10)   # 返回距离 + 索引
```

- `IndexFlatL2` 精确但慢；`IndexIVFFlat` / `IndexHNSW` 是近似索引，牺牲少量精度换速度。

## 15. 损失函数实现

### 15.1 BPR 损失

```python
import torch
import torch.nn.functional as F
def bpr_loss(pos_score, neg_score):
    return -F.logsigmoid(pos_score - neg_score).mean()
```

- 鼓励正样本得分高于负样本，`logsigmoid` 数值稳定。

### 15.2 InfoNCE 损失

```python
def info_nce(sim_matrix, temperature=0.07):
    # sim_matrix: (B, B)，对角线为自身（正样本）
    logits = sim_matrix / temperature
    labels = torch.arange(sim_matrix.size(0))   # 对角线索引
    return F.cross_entropy(logits, labels)
```

- 把「自身」当正类，其余当负类，做 softmax 交叉熵；`temperature` 控制分布锐度。

### 15.3 负熵（探索损失）

```python
def entropy_loss(probs):
    probs = torch.clamp(probs, min=1e-9)
    return (probs * torch.log(probs)).sum(dim=-1).mean()   # 熵 H
# 探索损失 = -H，鼓励分布更均匀
```

- 最小化负熵 = 最大化熵，防止控制器/门控坍缩到单一选择。


---

## 16. 序列推荐模型实现（对应 recsys.md §5.1）

### 16.1 GRU4Rec：RNN 建模会话内依赖

```python
import torch.nn as nn
# 输入: (B, T) 的 item id 序列
self.item_emb = nn.Embedding(num_items, hidden_size)
self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)  # batch_first 让输入为 (B,T,D)
self.fc = nn.Linear(hidden_size, num_items)

def forward(self, seq):
    emb = self.item_emb(seq)          # (B,T,D)
    out, h = self.gru(emb)            # out (B,T,D), h (1,B,D) 最后一个隐状态
    logits = self.fc(out)             # (B,T,num_items)，取最后一步预测下一物品
    return logits[:, -1, :]
```

- `nn.GRU(input_size, hidden_size, batch_first=True)`：`batch_first=True` 时输入形状 `(B, T, D)`，否则默认 `(T, B, D)`（RNN 类默认 time-first，易踩坑）。
- 多层用 `num_layers=2`；`dropout` 参数仅在 `num_layers>1` 时生效。

### 16.2 SASRec：单向 Transformer + 因果掩码

```python
import torch
import torch.nn.functional as F

def self_attention(Q, K, V, attn_mask, scale):
    scores = Q @ K.transpose(-2, -1) / scale      # (B,T,T)
    scores = scores.masked_fill(attn_mask == 0, -1e9)  # 因果掩码：只看过去
    return F.softmax(scores, dim=-1) @ V

# 因果掩码构造（上三角为 0，屏蔽未来）
T = seq_len
attn_mask = torch.tril(torch.ones(T, T)).bool()   # 下三角全 1
```

- **因果掩码**：`torch.tril` 取矩阵下三角，让位置 t 只能关注 ≤ t 的位置（模拟"从左到右"预测下一物品）。
- `masked_fill(mask==0, -1e9)` 把被屏蔽位置的分数设为极小值，softmax 后权重≈0。

### 16.3 BERT4Rec：双向 + 完形填空（Cloze）

```python
# 与 SASRec 的区别：不用因果掩码，改用 Cloze 掩码随机遮住部分 item
masked_seq = seq.clone()
mask_prob = 0.2
mask = (torch.rand(seq.shape) < mask_prob) & (seq != pad_id)
masked_seq[mask] = mask_id           # 被遮位置替换为 [MASK] token
logits = model(masked_seq)            # 双向注意力，预测被遮位置
loss = F.cross_entropy(logits[mask], seq[mask], ignore_index=pad_id)
```

- 双向注意力**不做**因果掩码（过去未来都能看），但训练时要**遮住未来物品**，否则目标泄漏。

### 16.4 DIN：目标相关的注意力加权

```python
def target_attention(user_hist_emb, target_emb, mask):
    # user_hist_emb: (B,T,D)，target_emb: (B,D)
    att = (user_hist_emb * target_emb.unsqueeze(1)).sum(-1)  # (B,T) 逐元素点积
    att = att.masked_fill(mask == 0, -1e9)
    att = F.softmax(att, dim=-1).unsqueeze(-1)               # (B,T,1)
    return (att * user_hist_emb).sum(dim=1)                   # (B,D) 加权求和
```

- **目标注意力**：用候选物品（target）当 query，给历史物品分配权重，只聚焦与当前候选相关的历史。

### 16.5 DIEN：GRU + 兴趣演化（AUGRU）

```python
# DIEN 在 GRU 后接注意力，再用 AUGRU（注意力门控 GRU）更新兴趣演化
att_score = ...  # 与 DIN 相同的目标注意力分数 (B,T)
# AUGRU：用注意力分数缩放更新门，控制兴趣更新强度
```

- DIEN 相比 DIN 多了「兴趣演化」：用 GRU 追踪兴趣随时间的变化，再用注意力门控让演化聚焦目标。

---

## 17. 图神经网络推荐（对应 recsys.md §5.2，补充 NGCF）

### 17.1 LightGCN 邻域聚合（无参）

```python
import scipy.sparse as sp
# A: (n_nodes, n_nodes) 归一化邻接矩阵（对称归一化 D^{-1/2} A D^{-1/2}）
def lightgcn_propagate(emb, adj, n_layers):
    embeds = [emb]
    for _ in range(n_layers):
        emb = adj @ emb           # 一次稀疏矩阵乘法 = 一层邻域聚合
        embeds.append(emb)
    return sum(embeds) / (n_layers + 1)   # 各层取平均（LightGCN 核心）
```

### 17.2 NGCF 特征变换 + 非线性（对比）

```python
# NGCF 相比 LightGCN 多了可学习权重 W 和非线性激活
def ngcf_propagate(emb, adj, W1, n_layers):
    embeds = [emb]
    for _ in range(n_layers):
        emb = adj @ emb
        emb = torch.tanh(W1(emb))         # 特征变换 + tanh 非线性（LightGCN 删掉这步）
        embeds.append(emb)
    return torch.cat(embeds, dim=-1)        # NGCF 拼接各层（LightGCN 是求和平均）
```

- LightGCN 的结论：对协同过滤而言，`特征变换 + 非线性激活` 几乎无益，删掉后更轻量且效果更好。

---

## 18. 对比学习增强视图（对应 recsys.md §5.3）

### 18.1 序列增强（CL4SRec：裁剪/掩码/重排）

```python
import random
def augment(seq, mode, mask_id=0):
    if mode == 'crop':        # 裁剪：随机保留连续一段
        l = random.randint(2, len(seq))
        start = random.randint(0, len(seq) - l)
        return seq[start:start+l]
    if mode == 'mask':        # 掩码：随机遮住部分位置
        mask = torch.rand(len(seq)) < 0.3
        return seq.masked_fill(mask, mask_id)
    if mode == 'reorder':     # 重排：随机打乱局部
        l = random.randint(2, len(seq))
        start = random.randint(0, len(seq) - l)
        perm = seq[start:start+l][torch.randperm(l)]
        return torch.cat([seq[:start], perm, seq[start+l:]])
```

### 18.2 图增强（SGL：边/节点扰动）

```python
import scipy.sparse as sp
def edge_dropout(adj, drop_rate):
    # 随机删边：按概率保留边
    rows, cols = adj.nonzero()
    keep = torch.rand(len(rows)) > drop_rate
    return sp.csr_matrix((adj.data[keep], (rows[keep], cols[keep])), shape=adj.shape)
```

- 对比学习核心是「同一对象造两个增强视图 → 拉近、不同对象推远」，损失用 InfoNCE（见 §15.2）。

---

## 19. 扩散推荐（对应 recsys.md §5.4）

### 19.1 前向加噪（闭式公式）

```python
import torch
def forward_diffusion(x0, t, betas):
    # betas: 预计算的噪声调度 (T,)
    alpha_bar = torch.cumprod(1 - betas, dim=0)      # ᾱ_t = ∏(1-β)
    alpha_bar_t = alpha_bar[t].view(-1, 1)           # 取 t 时刻
    eps = torch.randn_like(x0)                        # 采样噪声
    xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps
    return xt, eps
```

- **闭式公式**：`x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε`，一步直接从 x_0 跳到任意时刻 t，无需逐步迭代。
- **线性 β 调度**：`betas = torch.linspace(1e-4, 0.02, T)`（ADARec 用 T=50）。

### 19.2 反向去噪（训练目标 = 预测噪声）

```python
def denoise_loss(model, x0, betas):
    t = torch.randint(0, len(betas), (x0.shape[0],))   # 随机采样时刻
    xt, eps = forward_diffusion(x0, t, betas)          # 加噪
    eps_pred = model(xt, t)                            # 网络预测噪声
    return F.mse_loss(eps_pred, eps)                   # ‖ε_θ − ε‖²
```

- **重参数化技巧**：直接让网络预测「加进去的噪声 ε」，而非预测 x_0，训练更稳定。
- 推理时从纯噪声 x_T 出发，逐步用 `x_{t-1} = (x_t − β_t·ε_θ/√(1-ᾱ_t))/√(1-β_t) + σ_t·z` 去噪还原。

### 19.3 注意：ADARec 的「扩散」是简化版

```python
# ADARec 只有前向加噪 + 表示重构损失，没有独立去噪网络：
L_deno = F.mse_loss(adaptive.detach(), raw_output)   # 让加噪表示逼近原始表示
```

- 与标准 DDPM「预测噪声 ε」不同，ADARec 的扩散损失本质是「表示重构」（详见 python.md §4.6）。

---

## 20. 生成式检索语义码本（对应 recsys.md §5.5，TIGER）

### 20.1 RQ-VAE 语义 ID（残差量化）

```python
# 思路：用 KMeans 分层量化物品向量 → 得到多级语义 ID
from sklearn.cluster import KMeans
def build_codebook(item_embs, n_levels, n_codes):
    codes = []
    residual = item_embs
    for _ in range(n_levels):
        km = KMeans(n_clusters=n_codes)
        idx = km.fit_predict(residual)          # 每层一个离散码
        codes.append(idx)
        residual = residual - km.cluster_centers_[idx]   # 残差进入下一层
    # 物品的语义 ID = 各层码拼接，如 (3, 12, 7)
    return torch.stack(codes, dim=-1)
```

- **语义码本**：把连续物品向量量化成多级离散 ID，让生成模型能「逐 token 生成物品 ID」。
- 生成式检索的损失与语言模型一致：`F.cross_entropy(logits, semantic_id)`，逐步生成每一层码。

---

## 21. 大模型推荐 LoRA（对应 recsys.md §5.7）

### 21.1 LoRA 低秩微调

```python
from peft import LoraConfig, get_peft_model, TaskType
config = LoraConfig(
    r=8,                       # 低秩维度
    lora_alpha=16,             # 缩放系数
    target_modules=["q_proj", "v_proj"],   # 只在注意力 Q/V 加低秩旁路
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(base_llm, config)
# 只训练 lora 参数（约原参数的 1%），冻结主干
model.print_trainable_parameters()
```

- **核心思想**：冻结大模型主干，只训练两个低秩矩阵 `ΔW = B·A`（r ≪ d），参数量和显存大幅下降。
- 推荐任务（TALLRec）把「用户历史 + 物品」写成自然语言序列，用 LoRA 微调后做下一 token 预测。

### 21.2 Prompt 构造（推荐 → 文本）

```python
def build_prompt(user_seq, candidate):
    # 把推荐样本写成自然语言
    return f"用户浏览过：{user_seq}。请判断用户是否喜欢：{candidate}？"
# 输出 yes/no 或候选打分，用语言模型 token 概率做排序
```

---

## 22. 多任务推荐（对应 recsys.md §5.8）

### 22.1 MMoE 门控专家

```python
import torch
import torch.nn.functional as F
class MMoE(nn.Module):
    def __init__(self, d, n_experts, n_tasks):
        self.experts = nn.ModuleList([nn.Linear(d, d) for _ in range(n_experts)])
        self.gates = nn.ModuleList([nn.Linear(d, n_experts) for _ in range(n_tasks)])  # 每任务一个门控
    def forward(self, x):
        expert_out = torch.stack([e(x) for e in self.experts], dim=1)   # (B, n_experts, d)
        outs = []
        for gate in self.gates:
            w = F.softmax(gate(x), dim=-1).unsqueeze(-1)   # (B, n_experts, 1)
            outs.append((expert_out * w).sum(dim=1))        # 加权融合
        return outs   # n_tasks 个输出，各任务用不同专家组合
```

### 22.2 ESMM 链式（点击 → 转化）

```python
# ESMM 用「点击率 CTR × 转化率 CVR = 转化率 CTCVR」链式建模，缓解样本选择偏差
loss = bce(ctr_pred, y_click) + bce(ctcvr_pred, y_purchase)
# ctcvr_pred = ctr_pred * cvr_pred，让 CVR 也能用全量曝光样本训练
```

- 核心：CVR 只有「点击过的样本」才有标签，存在选择偏差；ESMM 通过共享 CTR 塔 + CTCVR 链式目标让 CVR 用全量样本学。

---

## 23. CTR 特征交叉（对应 recsys.md §0.2 经典）

### 23.1 DeepFM：FM 二阶交叉

```python
def fm_layer(emb):   # emb: (B, F, D) 稀疏特征 embedding
    # 一阶：线性项
    first = emb.sum(dim=1)
    # 二阶：FM 交叉 —— (sum)^2 - sum(^2)
    sum_square = emb.sum(dim=1) ** 2      # (B, D)
    square_sum = (emb ** 2).sum(dim=1)    # (B, D)
    second = 0.5 * (sum_square - square_sum)   # 显式建模两两特征交互
    return first, second
```

- FM 二阶交叉的核心是「和的平方减平方的和」技巧，避免显式遍历所有特征对（O(F²) 降到 O(F)）。

### 23.2 DCN：交叉网络

```python
def cross_network(x0, x, W, b):
    # 交叉层：x_{l+1} = x0 ⊙ (W·x_l + b) + x_l，显式建模高阶特征交互
    return x0 * (W @ x + b) + x
```

- DCN 的交叉网络每一层都让「原始输入 x0」与「当前表示」做逐元素乘，能显式捕获任意阶的特征交互。
