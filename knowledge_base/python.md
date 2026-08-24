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
