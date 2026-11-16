# RepPad：Repeated Padding 数据增强学习笔记

> 论文： [Repeated Padding for Sequential Recommendation](https://arxiv.org/abs/2403.06372v2)（RecSys 2024）
>
> 扩展论文： [Repeated Padding+: Simple yet Effective Data Augmentation Plugin for Sequential Recommendation](https://arxiv.org/abs/2403.06372v3)
>
> 代码：[experiment/ELSE/RepPad](../../experiment/ELSE/RepPad/)
>
> 记录时间：2026-11-15

---

# 一、文献阅读

## 1. 研究背景

序列推荐通常把用户历史交互补零到固定长度。短序列左侧会出现大量 padding，模型真正能利用的交互很少，训练信号也比较稀疏。普通零填充只是在输入中补无效位置，并没有增加有效的行为模式。

RepPad 的想法很直接：把已有的短序列重复放入固定长度窗口，让相同的历史行为在多个位置参与训练。这样不需要增加新物品，也不改变原始交互顺序，只改变序列的排列和填充方式。

RepPad+ 进一步考虑中等长度序列。对这类序列，完整重复一次可能超出最大长度，因此从原序列中截取连续子序列，再与完整序列拼接，使更多用户能够获得重复填充增强。

## 2. 研究目的

1. 减少短序列中无效零填充带来的信息浪费。
2. 让同一历史行为在多个有效位置产生训练梯度。
3. 在不引入额外物品和复杂网络模块的情况下提升序列推荐效果。
4. 将增强策略做成可插入 SASRec、GRU4Rec 等骨干的简单插件。

## 3. 研究问题

1. 重复历史序列能否比普通零填充提供更有效的训练信号？
2. 是否需要在重复序列之间加入分隔符 0？
3. RepPad+ 的中等长度子序列处理能否覆盖普通 RepPad 无法增强的用户？
4. 方法在不同数据集和不同序列推荐骨干上是否稳定？

## 4. 研究方法

RepPad 的增强发生在训练样本构造阶段，模型结构和推荐损失基本保持不变。

### 4.1 序列划分

对用户序列

$$
S_u=[v_1,v_2,\ldots,v_L],
$$

代码按留出法构造训练样本：

- 训练输入：$x_u=S_u[:-3]$；
- 训练目标：$y_u=S_u[1:-2]$；
- 验证答案：$v_{L-1}$；
- 测试答案：$v_L$。

因此训练增强不会使用验证和测试物品，避免把留出目标提前放入训练输入。实现见 [`SASRec/src/datasets.py`](../../experiment/ELSE/RepPad/SASRec/src/datasets.py)。

### 4.2 普通零填充

原始方法先将训练输入左侧补 0：

$$
\tilde{x}_u=[0,\ldots,0,x_u].
$$

0 只表示 padding，不作为真实物品参与推荐目标。

### 4.3 RepPad：整段重复填充

设最大序列长度为 $N$，训练序列长度为 $l=|x_u|$。当 $l<N$ 时，代码先计算：

$$
m=\left\lfloor \frac{N}{l}\right\rfloor,
$$

再从 $1$ 到 $m$ 中随机采样重复次数 $r$。

#### 不带分隔符：`aug_type=1`

代码将完整训练序列重复拼接：

$$
\tilde{x}_u=x_u^{(r)}\Vert x_u,
$$

其中 $x_u^{(r)}$ 表示连续重复 $r$ 次。目标序列同步重复，保证输入和下一物品目标逐位置对齐。

#### 带分隔符：`aug_type=2`

在重复块之间加入 0：

$$
\tilde{x}_u=(x_u\Vert[0])^{(r)}\Vert x_u.
$$

这里的 0 是分隔符，不是新物品。经过最后的截断和左填充后，序列长度仍为 $N$。

### 4.4 RepPad+：处理中等长度序列

RepPad+ 先判断是否还有足够空间进行增强。

#### 带分隔符：`aug_type=3`

当

$$
|x_u|>N-2
$$

时不增强；当 $\lfloor N/|x_u|\rfloor\leq 1$ 时，令

$$
q=N-1-|x_u|,
$$

随机选取长度为 $q$ 的连续子序列 $x_{u,start:start+q}$，构造：

$$
\tilde{x}_u=x_{u,start:start+q}\Vert[0]\Vert x_u.
$$

如果序列较短，可以完整重复，和 RepPad 的带分隔符版本一致。

#### 不带分隔符：`aug_type=4`

对应条件是 $|x_u|>N-1$，中等长度时令

$$
q=N-|x_u|,
$$

构造：

$$
\tilde{x}_u=x_{u,start:start+q}\Vert x_u.
$$

这个版本不放分隔符，能在相同窗口内多保留一个有效物品位置。

### 4.5 推荐目标

模型对每个有效位置预测下一个物品。对正物品和负物品使用二元交叉熵：

$$
\mathcal{L}_{rec}=-\sum_{t\in\mathcal{T}}
\left[\log\sigma(h_t^\top e_{y_t^+})+
\log(1-\sigma(h_t^\top e_{y_t^-}))\right].
$$

其中 $h_t$ 是序列编码器在位置 $t$ 的表示，$e_{y_t^+}$ 和 $e_{y_t^-}$ 分别为正、负物品嵌入，$\mathcal{T}$ 是非 padding 的训练位置集合。RepPad 没有额外的增强损失，增强后的序列直接进入相同的推荐目标。

## 5. 模型与实验设置

### 5.1 骨干模型

本地完整结果主要来自 SASRec。模型包含物品嵌入、位置嵌入、LayerNorm、dropout 和因果 Transformer 编码器。日志中的默认设置为：

| 参数 | SASRec 本地设置 |
|---|---:|
| hidden size | 64 |
| Transformer 层数 | 2 |
| attention heads | 2 |
| hidden dropout | 0.5 |
| attention dropout | 0.5 |
| initializer range | 0.02 |
| max sequence length | 50 |
| optimizer | Adam |
| learning rate | 0.001 |
| weight decay | 0 |
| batch size | 256 |
| 最大 epoch | 500 |
| early stopping patience | 10 |
| validation start | epoch > 200 |
| seed | 2024 |

论文和仓库还提供 GRU4Rec 接口，但当前 `GRU4Rec/src/output` 没有本地训练日志，下面的结果表不把 GRU4Rec 当作已完成复现实验。

### 5.2 数据集

当前 SASRec 数据目录包含 Beauty、Home、Sports and Outdoors、Toys and Games、Yelp、LastFM 和 ml-1m。日志中的五组 Original/RepPad 对照为 Beauty、Home、Sports and Outdoors、Toys and Games、Yelp；LastFM 和 ml-1m 目前只有 RepPad+ 结果。

用户和物品经过预处理后映射为连续整数 ID，0 保留给 padding。训练、验证、测试使用同一用户序列的时间顺序进行留出。

### 5.3 评估指标

采用全物品排序，报告：

- Hit@5、NDCG@5；
- Hit@10、NDCG@10；
- Hit@20、NDCG@20。

验证阶段使用验证答案，测试阶段重新使用测试交互矩阵屏蔽已见物品。日志最后的 `Epoch: 0` 结果是加载最佳检查点后的测试结果，不是第 0 个训练 epoch 的结果。

---

# 二、代码复现

## 1. 代码结构

| 文件 | 作用 |
|---|---|
| [`SASRec/src/main.py`](../../experiment/ELSE/RepPad/SASRec/src/main.py) | 解析参数、读取数据、构造 DataLoader、训练、早停和测试 |
| [`SASRec/src/datasets.py`](../../experiment/ELSE/RepPad/SASRec/src/datasets.py) | 划分训练/验证/测试序列，执行 aug_type=0～4 的重复填充 |
| [`SASRec/src/models.py`](../../experiment/ELSE/RepPad/SASRec/src/models.py) | SASRec 物品/位置嵌入、因果注意力 mask 和编码器 |
| [`SASRec/src/trainers.py`](../../experiment/ELSE/RepPad/SASRec/src/trainers.py) | BCE 损失、训练循环、全物品评估、检查点保存加载 |
| [`SASRec/src/utils.py`](../../experiment/ELSE/RepPad/SASRec/src/utils.py) | 数据读取、负采样、指标计算和 EarlyStopping |
| [`GRU4Rec/src`](../../experiment/ELSE/RepPad/GRU4Rec/src) | GRU4Rec 版本代码，当前没有对应输出日志 |

## 2. 环境与运行命令

仓库没有单独的 `requirements.txt`。代码依赖 PyTorch、NumPy、tqdm、gensim，以及与当前 PyTorch 版本兼容的 SciPy。

在 SASRec 源码目录运行：

```powershell
cd F:\recsys-research-training-zhangxinyu\experiment\ELSE\RepPad\SASRec\src

# Original
python main.py --data_name=Beauty --aug_type=0 --model_idx=1

# RepPad：随机重复填充，不使用分隔符
python main.py --data_name=Beauty --aug_type=1 --model_idx=2

# RepPad：随机重复填充，重复块之间加入 0
python main.py --data_name=Beauty --aug_type=2 --model_idx=3

# RepPad+：带分隔符的中等长度子序列增强
python main.py --data_name=Beauty --aug_type=3 --model_idx=6

# RepPad+：不带分隔符的中等长度子序列增强
python main.py --data_name=Beauty --aug_type=4 --model_idx=7
```

更换 `--data_name` 即可运行 Home、Sports_and_Outdoors、Toys_and_Games、Yelp、LastFM 或 `ml-1m`。使用 GPU 0 时保留默认 `--gpu_id=0`；CPU 可加 `--no_cuda`。

代码的原始 README 也给出了 `aug_type=0/1/2` 的 RepPad 命令和 `aug_type=3/4` 的 RepPad+ 命令。

## 3. 训练流程对应代码

1. `main.py` 调用 `get_user_seqs` 读取已经按时间排序的用户序列，并设置 `item_size=max_item+2`。
2. `SASRecDataset.__getitem__` 根据 `data_type` 构造训练、验证或测试样本。
3. 训练样本先在 `datasets.py` 中执行对应的重复填充，再由 `_data_sample_rec_task` 左填充或截断到长度 50。
4. `Trainer.cross_entropy` 对非零正目标位置计算正负 BCE。
5. `SASRecModel.transformer_encoder` 构造 padding mask 和上三角未来 mask，保证位置 $t$ 不能读取未来交互。
6. 验证从 epoch 201 左右开始，`EarlyStopping` 根据验证指标保存最佳模型；训练结束后将测试交互矩阵用于全物品评估。

## 4. 实验结果

本目录中的结果分为两类：

- `src/output/Original`、`src/output/RepPad`、`src/output/RepPad+`：仓库随代码提供的官方结果文件；
- `src/output` 根目录下的 `SASRec-*.txt`：本地复现实验结果。

两类结果不能混在一起计算。官方文件用于对照论文和检查结果范围，本地文件用于判断当前环境是否复现出相同趋势。

### 4.1 官方结果：Original、RepPad 与 RepPad+

下面是官方结果目录中五个数据集的测试指标。格式为 `H@5/N@5/H@10/N@10/H@20/N@20`。

| 数据集 | Original | RepPad (`aug_type=1`) | RepPad+ 官方文件 |
|---|---|---|---|
| Beauty | 0.0370 / 0.0242 / 0.0591 / 0.0312 / 0.0919 / 0.0394 | 0.0503 / 0.0336 / 0.0749 / 0.0415 / 0.1076 / 0.0497 | 0.0512 / 0.0344 / 0.0762 / 0.0424 / 0.1089 / 0.0507 |
| Home | 0.0093 / 0.0060 / 0.0149 / 0.0078 / 0.0230 / 0.0098 | 0.0126 / 0.0082 / 0.0196 / 0.0104 / 0.0305 / 0.0132 | 0.0133 / 0.0088 / 0.0207 / 0.0112 / 0.0322 / 0.0141 |
| Sports_and_Outdoors | 0.0218 / 0.0143 / 0.0322 / 0.0176 / 0.0482 / 0.0217 | 0.0259 / 0.0171 / 0.0411 / 0.0220 / 0.0616 / 0.0271 | 0.0294 / 0.0194 / 0.0434 / 0.0239 / 0.0641 / 0.0291 |
| Toys_and_Games | 0.0479 / 0.0324 / 0.0721 / 0.0402 / 0.1017 / 0.0477 | 0.0570 / 0.0390 / 0.0803 / 0.0465 / 0.1107 / 0.0542 | 0.0622 / 0.0433 / 0.0866 / 0.0512 / 0.1172 / 0.0589 |
| Yelp | 0.0139 / 0.0088 / 0.0250 / 0.0123 / 0.0441 / 0.0171 | 0.0197 / 0.0124 / 0.0355 / 0.0174 / 0.0593 / 0.0234 | 0.0207 / 0.0130 / 0.0359 / 0.0179 / 0.0598 / 0.0239 |

官方结果显示，RepPad 相比 Original 在五个数据集上均提升，RepPad+ 通常继续提升。这里的 `RepPad+ 官方文件` 使用仓库附带日志中的配置标记，不能根据文件夹名称把它们当成本地复现结果。

### 4.2 本地复现结果：`src/output` 根目录

本地根目录日志中，`aug_type=3` 是带分隔符的 RepPad+，`aug_type=4` 是不带分隔符的 RepPad+。当前没有在本地根目录留下对应的 `aug_type=0` Original 和 `aug_type=1` RepPad 日志，因此不能用下面的本地结果计算相对 Original 的提升。

| 数据集 | 本地 `aug_type=3`：H@5/N@5/H@10/N@10/H@20/N@20 | 本地 `aug_type=4`：H@5/N@5/H@10/N@10/H@20/N@20 |
|---|---|---|
| Beauty | 0.0515 / 0.0341 / 0.0763 / 0.0421 / 0.1094 / 0.0503 | 0.0490 / 0.0323 / 0.0742 / 0.0404 / 0.1094 / 0.0493 |
| Home | 0.0125 / 0.0084 / 0.0197 / 0.0108 / 0.0311 / 0.0136 | 0.0123 / 0.0079 / 0.0197 / 0.0103 / 0.0305 / 0.0130 |
| Sports_and_Outdoors | 0.0268 / 0.0179 / 0.0414 / 0.0226 / 0.0622 / 0.0278 | 0.0264 / 0.0174 / 0.0415 / 0.0223 / 0.0608 / 0.0272 |
| Toys_and_Games | 0.0567 / 0.0390 / 0.0824 / 0.0472 / 0.1134 / 0.0550 | 0.0582 / 0.0399 / 0.0832 / 0.0480 / 0.1151 / 0.0560 |
| Yelp | 0.0208 / 0.0126 / 0.0362 / 0.0176 / 0.0605 / 0.0236 | 0.0209 / 0.0131 / 0.0352 / 0.0177 / 0.0593 / 0.0237 |
| LastFM | 0.0569 / 0.0399 / 0.0734 / 0.0453 / 0.1092 / 0.0544 | 0.0495 / 0.0317 / 0.0743 / 0.0395 / 0.1101 / 0.0482 |
| ml-1m | 0.1285 / 0.0828 / 0.2093 / 0.1086 / 0.3225 / 0.1372 | 0.1326 / 0.0860 / 0.2141 / 0.1121 / 0.3230 / 0.1396 |

### 4.3 本地结果与官方结果对比

本地和官方都运行了 SASRec 的 RepPad+，但日志来自不同目录和实验过程，不能直接当作同一随机种子的重复运行。以共同的五个数据集为例：

- Beauty：本地 `aug_type=3` 的 H@10/N@10/H@20/N@20 为 0.0763/0.0421/0.1094/0.0503，官方 RepPad+ 为 0.0762/0.0424/0.1089/0.0507，整体非常接近；
- Home：本地对应指标为 0.0197/0.0108/0.0311/0.0136，官方为 0.0207/0.0112/0.0322/0.0141，本地略低；
- Sports_and_Outdoors：本地为 0.0414/0.0226/0.0622/0.0278，官方为 0.0434/0.0239/0.0641/0.0291，本地有一定差距；
- Toys_and_Games：本地为 0.0824/0.0472/0.1134/0.0550，官方为 0.0866/0.0512/0.1172/0.0589，本地低于官方；
- Yelp：本地为 0.0362/0.0176/0.0605/0.0236，官方为 0.0359/0.0179/0.0598/0.0239，整体接近。

本地结果已经复现出 RepPad+ 的总体效果和大致数值范围，但由于官方结果和本地结果的随机状态、训练过程及检查点来源不同，不能据此判断每个指标完全复现。

## 5. 结果分析

### 5.1 官方结果分析

官方 Original、RepPad、RepPad+ 文件的共同趋势是：重复填充提高了短序列数据上的有效训练位置，RepPad 通常优于普通零填充，RepPad+ 在中等长度序列上进一步补充增强。Yelp 和 Home 的相对提升较明显，Toys_and_Games 的绝对数值更高但相对提升较小。

### 5.2 本地复现结果分析

本地 `aug_type=3` 与 `aug_type=4` 的结果没有固定的优劣关系。带分隔符版本在 Beauty、Home、Sports_and_Outdoors 和 Yelp 的部分 NDCG 指标更好；无分隔符版本在 Toys_and_Games、ml-1m 以及部分 H@10/H@20 指标更好。分隔符会占用一个序列位置，但能帮助模型区分重复块；无分隔符则能保留更多重复物品。

本地日志只有 RepPad+ 两种配置，没有本地 Original 和 RepPad 基线，因此这里不再把官方基线与本地结果混合计算提升比例。后续若需要严格复现，应在同一环境下补跑 `aug_type=0`、`aug_type=1`，再比较三种方法。

## 6. 代码检查记录

1. `aug_type=0～4` 在当前 `datasets.py` 中有明确分支；其他整数不会进入有效分支，训练样本变量可能未定义。
2. `aug_type=1/2` 使用 `int(max_len / len(input_ids))` 决定可重复次数，短序列有更多重复机会；输入长度接近 50 时不再重复。
3. `aug_type=3/4` 的中等长度分支使用随机连续子序列，随机起点由 `random.randint` 决定。
4. 训练样本中的目标序列和输入序列同步重复，避免输入与标签错位。
5. 评估阶段不执行增强，验证和测试仍使用原始留出序列；本地结果是“增强训练、原始序列评估”。
6. 当前 `main.py` 中 `set_seed(args.seed)` 被注释，日志虽然记录 `seed=2024`，但没有真正固定 Python、NumPy 和 PyTorch 的随机种子。这是复现实验时需要注意的随机性来源。
7. `star_valid` 实际条件是 `epoch > star_valid`，默认 200，因此第一次验证不是严格的 epoch 200，而是后续 epoch。
8. 官方 `RepPad+` 子目录中的部分日志标记为 `aug_type=5/6`，而当前数据集实现只支持 0～4。这说明官方结果文件和当前代码版本存在标记差异；这些文件仍按官方结果记录，不当作当前代码的本地输出。

## 7. 空白记录

### Bug 检查及修复

留空。

### 代码修改及优化

留空。

### 实验结果复现

官方目录已有 Original、RepPad、RepPad+ 结果；本地 `output` 根目录已有 SASRec RepPad+ 两种配置结果，尚未补跑本地 Original、RepPad 基线、多 seed 平均和 GRU4Rec 完整实验。

### 结果对比及分析

已完成官方结果与本地 RepPad+ 结果的分目录整理和对照；后续补跑本地 Original、RepPad 后，再进行同一环境下的三方法比较。

## 8. 总结

RepPad 的核心不是生成新物品，而是重复利用已有短序列，把原本的零填充位置变成有效训练位置。RepPad+ 再用连续子序列覆盖中等长度序列。官方结果显示两种方法都能提升总体 Hit 和 NDCG；本地结果已经接近官方 RepPad+ 的数值范围，但本地 Original、RepPad 基线尚未补跑，论文均值和本地单 seed 结果需要分开理解。
