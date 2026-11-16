# MuPP：通过多偏好填充增强短序列

> 论文：MuPP: Multi-Preference Padding for Sequential Recommendation，SIGIR 2026（pp. 3854–3859）。[https://doi.org/10.1145/3805712.3809950](https://doi.org/10.1145/3805712.3809950)
> 代码：https://github.com/kangvic0615/MuPP（MIT 许可，基于 RecBole 二次开发）
> 更新记录：2026-10-25 首次整理；2026-10-31 补入 Beauty 与 Video Games 的本地训练、验证/测试指标和日志路径。原文 PDF 仍未获取，论文主表数值继续留空，不把本地结果当作论文结果。

---

# 一、文献阅读

## 1. 研究背景

序列推荐通常把每个用户的历史截断或补齐到固定长度 $L$，短序列用特殊的 0 填充。填充位在注意力里被 mask 掉，既不计损失也不参与打分，等于**输入预算被浪费**：本地 Beauty 的平均长度约 7.7、Video Games 约 9.3，在 $L=50$ 的标准设置下大部分位置都是 padding。

前作 Repeated Padding（RecSys 2024；扩展版 RepPad+，[arXiv:2403.06372](https://arxiv.org/abs/2403.06372)）把这个“闲置空间”直接用起来：训练时把原始序列重复若干次填到 padding 位，并指出这是**无参数、无超参、模型无关**的序列级增强；扩展版摘要报告在七个数据集、多类骨干上的平均提升最高为 SASRec 35.34%、GRU4Rec 84.11%。

但重复填充注入的是**同一物品同一嵌入的多个副本**：第 2 块和第 3 块看到的是完全相同的表示，只有位置编码不同；能复制的次数也被长度限死（最多 $\lfloor L/|S|\rfloor$ 次）。MuPP 要解决的正是这一点——**让填充内容带上不同的“偏好”表示**。

> 口径说明：上面的提升数字来自 RepPad+ 的 arXiv 摘要（v3，扩展版），不是 MuPP 自己的结果；MuPP 的动机表述未能核对原文，方法描述以下面的官方代码为准。

## 2. 研究目的

- 把“重复同一个物品”换成“同一物品的多个偏好视图”：每一块重复经过一个不同的可学习线性变换，得到多偏好填充。
- 用正交正则约束这些偏好视图之间的相似度，避免多个变换学成同一个映射。
- 保持预测头不变（内积 + 全物品交叉熵），新增参数只有 $K$ 个 $d\times d$ 线性层，推理时的网络结构不变。

## 3. 研究问题

原文 RQ 未能核对，下面按方法结构与实验配置归纳出三个可检验的问题：

1. 重复填充的收益来自“占满位置”，还是来自“重复的内容”？
2. 把每块重复换成不同的偏好变换，是否比对完全相同的副本做重复（RepPad）更好？
3. 偏好个数 $K$ 与正交强度 $\lambda_{orth}$ 是否敏感？

## 4. 研究方法

一条样本的完整数据流：

| 顺序 | 输入 | 处理 | 产物 |
|---|---|---|---|
| 1 | 原始序列 $S$（长度 $n$） | 采样重复次数 $r\sim\mathcal U\{1,\dots,\lfloor L/n\rfloor\}$ | $r$ |
| 2 | $S$、$r$ | 按块平铺到长度 $L$，超出部分补 0 | $\tilde S$ |
| 3 | $\tilde S$ | 分块变换：第 $b$ 块用 $W_{b\bmod K}$，batch 内最大重复数的最后一块用原始嵌入 | 嵌入序列 |
| 4 | 嵌入序列 | 加位置嵌入 → LayerNorm → dropout | $H^0$ |
| 5 | $H^0$ | 因果掩码 Transformer（`n_layers` 层） | $H$ |
| 6 | $H$ | 取第 $\min(nr,L)-1$ 个位置 | $z$ |
| 7 | $z$ | $Ez$ 做全物品打分，交叉熵 + 正交正则 | 损失 |

三个关键设计点：

- **重复次数是随机的**，每个样本、每次前向都重新采样，推理时也一样；
- **块内位置沿用原位置编码**：同一物品在第 $b$ 块里的位置是 $bn+u$，位置嵌入会随块变化，因此“同一物品的不同副本”天然带有不同的位置信号；
- **最后一块保留原始嵌入**，其余块经过 $W_{b\bmod K}$；$K=2$ 时两块交替使用 $W_0$、$W_1$。

## 5. 研究内容（公式分析）

> 原文公式编号未能核对。下式按 `recbole/model/sequential_recommender/mupp.py` 整理成等价形式，编号（A）–（E）为本笔记自拟，与代码逐行对应。

**（A）重复次数**

$$
r_u\sim\mathcal U\left\{1,\dots,\left\lfloor L/n_u\right\rfloor\right\}
$$

对应 `max_repeats = L // item_seq_len` 与 `torch.randint(1, max_repeats + 1)`。

**（B）平铺与掩码**

记 $b(t)=\lfloor t/n\rfloor$、$u(t)=t\bmod n$：

$$
\tilde S_t=
\begin{cases}
S_{u(t)}, & b(t)<r,\\
0, & \text{否则},
\end{cases}
\qquad
n'=\min(nr,\,L).
$$

**（C）分块偏好变换**

$$
e_t=
\begin{cases}
E[\tilde S_t], & b(t)=B_{\max}-1,\\
W_{b(t)\bmod K}\,E[\tilde S_t], & \text{否则},
\end{cases}
$$

其中 $B_{\max}=\max_u r_u$ 是**整个 batch 内**的最大重复数，而不是样本自己的 $r_u$（这一点的后果见第 8 节与第二节第 4 条）。

**（D）正交正则**（`orthogonality_regularization`）

$$
\mathcal L_{orth}=\frac{1}{C}\sum_{i<j}\left(\frac{\left\langle \mathrm{vec}(W_i),\mathrm{vec}(W_j)\right\rangle}{\|\mathrm{vec}(W_i)\|\,\|\mathrm{vec}(W_j)\|}\right)^{2},
\qquad C=\binom{K}{2}.
$$

$K=2$ 时只有一对权重，正则退化为“让两个变换的权重尽量正交”。

**（E）总损失**

$$
\mathcal L=\mathrm{CE}\big(\mathrm{softmax}(Ez),\,y\big)+\lambda_{orth}\,\mathcal L_{orth}.
$$

Beauty 取 $\lambda_{orth}=0.001$，Video Games 取 $0.003$。

**两个容易被忽略的实现细节：**

1. **$r=1$ 时输出等价于 SASRec**（唯一的块就是 $B_{\max}-1$，走原始嵌入；正交正则仍在总损失里，但不影响打分）。也就是说增强只在“确实发生了重复”时生效，短序列被重复的概率与幅度都更大。
2. **pad 嵌入不是零向量**：`nn.Embedding(..., padding_idx=0)` 只在构造时把第 0 行置零，紧接着 `self.apply(self._init_weights)` 对所有 Embedding 权重做 `normal_(0, 0.02)`，第 0 行因此变成随机向量；而 `padding_idx` 只屏蔽梯度，这个随机向量在训练中不再变化。所以代码里的“补 0”实际是“补一个固定的随机嵌入”。

## 6. 实验环境、数据集、参数配置

仓库只提供两个数据集的配置；论文主表是否只含这两个数据集未核对。

数据集规模（本地 `dataset/*.inter`；按 `user/item ≥ 5` 单遍过滤后自行统计，未与 RecBole 实际加载结果核对）：

| 数据集 | 用户数 | 物品数 | 交互数 | 平均长度 | 划分 |
|---|---:|---:|---:|---:|---|
| Amazon_Beauty | 26,360 | 13,484 | 204,031 | 7.74 | 随机 8:1:1（`RS`） |
| Amazon_Video_Games | 31,027 | 26,108 | 287,107 | 9.25 | 留一法（`LS`） |

超参数（`configs/Amazon_Beauty.yaml`、`configs/Amazon_Video.yaml`）：

| 项目 | Beauty | Video Games |
|---|---|---|
| 最大长度 | 50 | 50 |
| 层数 / 头数 | 2 / 4 | 4 / 8 |
| 隐藏维 / FFN 维 | 256 / 256 | 256 / 256 |
| dropout（hidden / attn） | 0.3 / 0.3 | 0.5 / 0.5 |
| batch size | 2048 | 4096 |
| 学习率 / 优化器 | 5e-4 / Adam | 5e-4 / Adam |
| 损失 | 全物品 CE（无负采样） | 同左 |
| $K$ / $\lambda_{orth}$ | 2 / 0.001 | 2 / 0.003 |
| 轮数 / 评估间隔 / 早停 | 300 / 2 / 5 | 300 / 2 / 5 |
| 指标 / 选模 | NDCG、MRR，topk=[3,5,10] / valid NDCG@10 | 同左 |
| 随机种子 | 2025 | 2025 |

**两套切分不一致**是这篇最需要注意的口径问题：Beauty 用随机 8:1:1，Video Games 用 leave-one-out。两者不能直接横向比较，也不宜和本目录其他笔记里同样叫 Beauty 的 5-core 版本（22,363 用户 / 12,101 物品）对照——本地这份 Beauty 是 2023 年的另一份快照，规模更大。

## 7. 完整实验结果表格

原文 PDF 无法获取，只从检索索引确认了主表的列名：

`Datasets | Metric | SASRec | CL4SRec | RepPad | MuPP | Improve`

| 数据集 | 指标 | SASRec | CL4SRec | RepPad | MuPP | Improve |
|---|---|---|---|---|---|---|
| （待补） | （待补） | — | — | — | — | — |

### 7.1 本地运行结果（seed 2025；不是论文表格）

两次运行均使用官方 RecBole 入口、`repeatable=True`、全物品排序、`valid_metric=NDCG@10`，模型在验证集 NDCG@10 最优轮保存后再测试。Beauty 是随机 8:1:1 切分（RS），Video Games 是按时间留一法（LS），不能把两行直接横向比较。

| 数据集 | 配置 | 最佳轮 | 验证 NDCG@10 | 验证 MRR@10 | 测试 NDCG@10 | 测试 MRR@10 | 日志 |
|---|---|---:|---:|---:|---:|---:|---|
| Amazon_Beauty | `configs/Amazon_Beauty.yaml`；batch 4096 | 21 | 0.0650 | 0.0531 | 0.0495 | 0.0409 | `experiment/MuPP/log/MuPP/MuPP-Amazon_Beauty-Oct-26-2026_22-24-54-b2fc0a.log` |
| Amazon_Video_Games | `configs/Amazon_Video.yaml`；batch 2048 | 19 | 0.0744 | 0.0564 | 0.0651 | 0.0493 | `experiment/MuPP/log/MuPP/MuPP-Amazon_Video_Games-Oct-27-2026_16-57-57-275b03.log` |

完整 top-k 指标如下，便于和 RecBole 的原始日志逐项核对：

| 数据集 | split | NDCG@3 | NDCG@5 | NDCG@10 | MRR@3 | MRR@5 | MRR@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| Amazon_Beauty | valid | 0.0493 | 0.0563 | 0.0650 | 0.0457 | 0.0495 | 0.0531 |
| Amazon_Beauty | test | 0.0386 | 0.0434 | 0.0495 | 0.0358 | 0.0384 | 0.0409 |
| Amazon_Video_Games | valid | 0.0498 | 0.0599 | 0.0744 | 0.0448 | 0.0504 | 0.0564 |
| Amazon_Video_Games | test | 0.0438 | 0.0526 | 0.0651 | 0.0394 | 0.0442 | 0.0493 |

本地结果只能说明当前代码、数据快照和配置下的运行情况。由于论文 PDF 和论文主表尚未核对，不能计算“相对论文提升”，也不能把这两次单 seed 结果写成 MuPP 的论文成绩。补齐论文主表后按 `(本地 − 论文) / 论文 × 100%` 补写提升/下降百分比，与 DuoRec、MELT 等笔记的记法保持一致。

其余表格（消融、$K$ 与 $\lambda_{orth}$ 敏感性、序列长度分组、效率对比）连列名都未确认，全部留空；**不补估计值**。

## 8. 结果分析（待原文核对）

原文数值缺失，这里只做机制层面的判断，供拿到 PDF 后逐条验证：

1. **相对 RepPad 的改动很小**：只多了一个“块 → 变换”的映射和一项正则，参数量增加 $Kd^2$（$K=2$、$d=256$ 时约 13 万），训练与推理结构不变。所以 MuPP 若真的超过 RepPad，收益只能来自“同一物品在不同块里给出不同 key/value”这一条。
2. **但“多偏好”的表述与实现有距离**：$W_i$ 是全局共享的线性层，不含物品或位置条件，严格说是“多组全局线性投影”，而不是“为每个物品生成多个子偏好”。这是审稿视角下最容易被追问的点。
3. **重复次数在推理时也随机**：`forward` 里没有 `self.training` 分支，训练和评估都会执行重复，测试输入本身被改写；同一检查点在测试集上重复评估会得到不同指标，指标带额外方差。原文若报告均值与方差，需要确认评估时是否固定重复次数。
4. **超短序列的重复幅度很大**：$n=5$ 时 $r$ 最大可取 10，同一物品在输入里出现 10 次，注意力可能被重复副本主导。RepPad 原论文对填不满的序列用滑动窗口截取子序列来补，MuPP 是整段重复，这个差别是否影响结果需要核对。
5. **两套切分**：Beauty 随机划分、Video Games 留一法，若原文把两个数据集汇总成“平均提升”，含义会被稀释。

**与已有笔记的联系**：MuPP 和 BASRec、[TADA](<TADA - 通过尾部感知序列增强改善长尾推荐.md>) 同属“输入侧增强”，但改的东西不同——BASRec 改序列内容（重排 / 替换 / 跨序列 mixup），TADA 改头尾物品的分布，MuPP 改的是**本来被模型丢弃的 padding 位**。它不引入对比损失、不需要额外标签、也不改变推理结构，因此在工程上更像一个“前置插件”：先把短序列用多偏好填满，再叠加替换 / 混合类增强，两类做法不冲突。

## 9. 结论与局限

**方法主张**（拿到原文后应改写为原文结论）：把 padding 从“零填充”换成“带多偏好变换的重复填充”，在不改预测头、不增加推理结构的前提下，利用被浪费的输入位置。

**局限**

- 偏好视图来自全局线性层，与“每个物品有多个偏好”的自然语义不完全对应；
- 重复次数随机且推理期同样随机，评估的可重复性差；
- 实现里“最后一块保留原始嵌入”用的是 batch 内最大重复数，只有当样本恰好取到 batch 最大值时才成立；
- 重复上限 $\lfloor L/n\rfloor$ 随序列变短而增大，超短序列的输入会被同一批物品反复填满。

**待补清单**（拿到原文后按此顺序补）：RQ 原文与式号、主表全部数值、消融（去多偏好 / 去正交正则）、$K$ 与 $\lambda_{orth}$ 的敏感性、按序列长度分组的结果、与 RepPad 的对照条件、效率对比。

---

# 二、代码复现

> 官方仓库：<https://github.com/kangvic0615/MuPP>（2025-09-12 初始提交，2026-04-13 更新 README）。
> 本地副本：`experiment/ELSE/MuPP/`，两份数据集已随仓库提供。
> **复现状态（2026-10-31）：Beauty 与 Video Games 主实验已跑通。** 两次运行均为 seed 2025；消融、序列长度分组、多 seed 和 RepPad 对照尚未进行。

## 1. 仓库结构

| 路径 | 内容 |
|---|---|
| `recbole/model/sequential_recommender/mupp.py` | 模型本体（212 行），继承 RecBole 的 `SequentialRecommender` |
| `configs/Amazon_Beauty.yaml`、`configs/Amazon_Video.yaml` | 两个数据集全部超参 |
| `dataset/` | `.inter` / `.item` 文件；Beauty 另有未被代码引用的 `semantic_augmentation.npy` |
| `run_recbole.py`、`run_mupp.sh` | RecBole 标准入口与两行启动脚本 |
| `recbole/` | 随仓库打包的 RecBole 源码；`--model MuPP` 由 `get_model` 的 importlib 扫描自动发现，不需要额外注册模型 |

## 2. 环境与运行

依赖按 `requirement.txt`：`torch==2.5.0`、`tensorboard>=2.5.0`、`ray>=1.13.0,<=2.6.3`、`hyperopt==0.2.5`、`colorlog`、`thop`、`plotly` 等。

```bash
pip install -e .            # 仓库根目录，按 README
sh run_mupp.sh              # 等价于下面这行（注意脚本里默认跑 Video Games）
python run_recbole.py --model MuPP --dataset Amazon_Video_Games --config_files configs/Amazon_Video.yaml
```

本地尝试（2026-10-25 首次冒烟，随后于 2026-10-26/27 完成主实验；`D:\.conda\envs\python312`：Python 3.12.12、torch 2.11.0+cu128、CUDA 可用）：

| 步骤 | 结果 |
|---|---|
| `from recbole.quick_start import run`（10-25） | 失败：`ModuleNotFoundError: No module named 'ray'` |
| 注入 `ray` 桩后重试（10-25） | 失败：`ModuleNotFoundError: No module named 'tensorboard'`（`recbole/utils/utils.py` 顶层导入） |
| 正式运行（10-26/27） | Beauty 与 Video Games 均完成；指标见第 7.1 节 |

首次冒烟失败的两个依赖都写在 `requirement.txt` 里；后续正式运行已具备所需依赖，但本地 torch 2.11 与仓库声明的 2.5.0 仍不一致。正式结果来自后续日志，不把 10-25 的失败尝试当作训练结果。

此外还有两个入口级的坑：

- `run_recbole.py` 的默认值是 `--dataset Amazon_Beatuy`（拼写错误）和 `--config_files configs/MuPP.yaml`（文件不存在），不显式传参直接报错；
- 仓库没有 `recbole/properties/model/MuPP.yaml`，模型参数完全依赖两份 `configs/*.yaml`，缺字段（例如 `num_item_representation`）会在构造模型时 `KeyError`。

## 3. 论文模块 → 代码位置

| 方法要素 | 代码位置 | 实现要点 |
|---|---|---|
| 最大重复次数 | `mupp.py:104` `max_repeats = torch.div(seq_length, item_seq_len)` | 即 $\lfloor L/n\rfloor$，再 `clamp(min=1)` |
| 重复次数采样 | `mupp.py:106-109` `torch.randint(1, max_rep+1)` | **逐样本 Python 循环**，无 `training` 分支（推理也随机） |
| 平铺与掩码 | `mupp.py:111-120` `block_indices` / `within_block_pos` / `gather` | 整除与取模构造块索引，`valid_mask` 之外置 0 |
| 实际序列长度 | `mupp.py:121-125` `new_seq_len` | $\min(nr,L)$，用于取输出位置 |
| 分块偏好变换 | `mupp.py:130-138` `representation_layer[i % K]` | 最后一块用原始嵌入，其余用线性变换 |
| 位置嵌入 + LayerNorm + dropout | `mupp.py:144-150` | 位置 id 覆盖平铺后的整段 $L$ |
| 因果掩码 Transformer | `mupp.py:151-155`、`get_attention_mask` | 标准 RecBole `TransformerEncoder` |
| 取序列表示 | `mupp.py:157` `gather_indexes(output, new_seq_len-1)` | 不用固定的最后一位 |
| 正交正则 | `mupp.py:160-175` | 权重展平后 L2 归一化，两两余弦平方取均值 |
| 全物品 CE + 正则 | `mupp.py:190-195` | `logits = z @ E^T`，`loss = CE + λ·L_orth` |
| 评估 | `mupp.py:206-212` `full_sort_predict` + 配置 | 全排序，`mode: full`，指标 NDCG / MRR |

## 4. 实现细节与可疑点

1. **推理期随机**：`forward` 每次都重新采样 `num_repeats`，没有 `if self.training` 判断。同一检查点在测试集上重复评估会得到不同指标，这既影响数值稳定性，也要和“推理结构不变”的说法分开看——推理次数不变，但结果不再确定。
2. **“最后一块保留原嵌入”用的是 batch 最大值**：`if i == max_rep_int - 1` 里的 `max_rep_int` 来自 batch 内最大重复数。若某样本的 $r_u$ 小于它，该样本**所有**块都会被变换，没有任何一块保留原始嵌入；反之只有取到最大值的样本才符合设计意图。若要严格实现“每样本最后一块保留原嵌入”，应改成 `i == num_repeats[u] - 1`。
3. **`torch.div` 得到浮点**：`max_repeats` 是 float 张量，靠 `int(...)` 取整，$L$ 能被 $n$ 整除的边界上依赖浮点表示，建议显式用整除。
4. **pad 嵌入不是零向量**：见第一节第 5 条第 2 点，`padding_idx=0` 的零初始化被 `_init_weights` 覆盖。
5. **配置里有未使用的键**：`mask_ratio`、`ft_ratio`、`pooling_mode` 在 `mupp.py` 中没有任何引用（疑似从其他模型复制）；`semantic_augmentation.npy` 同样未被代码引用。
6. **两套切分写死在配置里**：Beauty `RS [0.8,0.1,0.1]`、Video Games `LS valid_and_test`，同一套脚本无法横向比较。

## 5. 复现清单

| 实验 | 状态 | 备注 |
|---|---|---|
| Beauty 主实验 | 已完成 | test NDCG@10=0.0495，MRR@10=0.0409；最佳轮 21 |
| Video Games 主实验 | 已完成 | test NDCG@10=0.0651，MRR@10=0.0493；最佳轮 19 |
| 消融（去多偏好 / 去正交正则） | 未做 | 代码无开关；可用 `K=1`、`λ=0` 近似，但不是严格等价 |
| $K$ 与 $\lambda_{orth}$ 敏感性 | 未做 | 两份配置分别给了 0.001 与 0.003，可据此复跑 |
| 按序列长度分组（短序列收益） | 未做 | RecBole 自带 evaluator 不分组，需要自己加 |
| 与 RepPad 对照 | 未做 | 把 $K$ 设为 1 并令变换为恒等可近似，但不是严格等价 |

## 6. 复现风险与下一步

- **先补依赖再谈复现**：`tensorboard`、`ray`（≤2.6.3）必须装；本地 torch 2.11 与仓库声明的 2.5.0 不一致，若出现 `torch.load` 或张量 API 报错，优先怀疑版本。
- **固定重复次数做对照**：为排除第 4 节第 1 条的影响，建议加一个 `eval_num_repeats` 开关（例如固定 $r=\lfloor L/n\rfloor$），先确认评估稳定，再比较模型。
- **核对“最后一块”的语义**：是保留官方行为（batch 最大值判断）还是按设计意图改成 per-sample 判断，两者结果不可直接比较，必须在记录里写明取的是哪一种。
- **两个数据集切分不同**：报告时逐数据集标注 `RS` / `LS`，不要跨数据集汇总成“平均提升”。
- **Beauty 版本差异**：本地 Beauty 是 2023 快照（26,360 用户 / 13,484 物品），与本目录其他笔记里的 5-core Beauty（22,363 / 12,101）不是同一份数据，绝对值不可直接对比。
