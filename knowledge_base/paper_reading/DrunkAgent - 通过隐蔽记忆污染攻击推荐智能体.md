# DrunkAgent：通过隐蔽记忆污染攻击推荐智能体

> 论文：DrunkAgent: Stealthy Memory Corruption in LLM-Powered Recommender Agents，WWW 2026。[https://doi.org/10.1145/3774904.3792688](https://doi.org/10.1145/3774904.3792688)
> 代码：无
> 更新记录：2026-10-22 统一笔记顶部信息格式；保留原有阅读与复现记录，论文结果与本地结果分开。

---
# 一、文献阅读

## 1. 研究背景

![原文图 1：推荐智能体的记忆污染示意](assets/DrunkAgent/fig1.png)

**图 1：推荐智能体的记忆污染示意**（英文原文 PDF 第 1 页）。 原文通过用户、物品及智能体记忆之间的关系说明污染如何影响后续推荐。

大语言模型（LLM）驱动的智能体正越来越多地用于推荐系统（RS），实现个性化行为建模。记忆机制在其中扮演关键角色，使智能体能自主探索、学习并从真实交互中自我演化。然而，这一作为上下文仓库的记忆机制，天然暴露了对抗性操纵的攻击面。尽管其核心地位重要，智能体 RS 面对此类威胁的鲁棒性仍鲜有探索。

现有工作存在以下问题：
- 传统协同过滤投毒攻击假设静态嵌入，关注数值扰动，与文本驱动系统存在语义鸿沟；
- 针对 LLM 推荐器的对抗文本攻击多关注预定义提示，难以适用于动态记忆更新的智能体 RS；
- 智能体 RS 研究忽视安全性分析，关键漏洞未被探索。

此外，商业推荐系统的黑盒特性加剧了挑战。与现实平台滥用（如 Amazon、eBay、Sony）一致的威胁向量是**物品推广攻击**。投毒攻击依赖虚假用户注入，成本高且间接；文本攻击通常关注降低整体性能或导致误分类，忽视了实际推广驱动的威胁。

## 2. 研究目的

首次系统研究 LLM 驱动推荐智能体中的基于记忆的漏洞，揭示其安全限制，并指导增强系统韧性与可信度。具体提出一种新颖的黑盒攻击框架 **DrunkAgent**，为目标物品推广精心构造语义有意义的对抗文本触发器，并引入一系列策略，通过破坏交互过程中的记忆更新来最大化触发器效果。触发器和策略在代理模型上优化，使 DrunkAgent 具有可迁移性和隐蔽性。

## 3. 研究问题

论文通过实验回答：
- DrunkAgent 能否在不同黑盒智能体 RS 上有效推广目标物品？（攻击迁移性）
- DrunkAgent 生成的对抗描述能否跨不同目标物品通用？（攻击通用性）
- DrunkAgent 是否隐蔽，不易被检测？（攻击隐蔽性）
- DrunkAgent 能否抵抗现有防御机制？（防御鲁棒性）

## 4. 研究方法

DrunkAgent 由三个模块组成：

1. **代理模块（Surrogate Module）**：构建一个智能体 RS 作为代理，模拟真实目标系统行为。包含用户智能体、物品智能体和推荐智能体，均基于 LLM 骨干并增强记忆组件。使用任务对齐的提示模板作为智能体间的行为接口。
2. **生成模块（Generation Module）**：生成有效的对抗描述触发器以推广目标物品。
   - **推荐上下文构建**：构造推荐风格交互上下文，包括基本提示模板（风格定义、任务目标、推荐指令、格式约束）、通用用户记忆、流行物品记忆。
   - **贪婪搜索算法**：从攻击目标对齐的种子开始，迭代选择、重组和润色候选，逐步精炼搜索空间。包括质量估计阶段、特征整合阶段、语言丰富阶段。
3. **策略模块（Strategy Module）**：阻碍目标物品智能体从环境中演化，即目标智能体的记忆在智能体-环境交互中无法有效更新。设计一系列策略使目标智能体“喝醉”：
   - 虚假任务响应（Fake Task Response）
   - 上下文文本切换（Contextual Text Switching）
   - 分段信号增强（Segmentation Signal Augment）
   - 恶意任务注入（Malicious Task Injection）
   - 特殊字符使用（Special Character Usage）

## 5. 研究内容（公式与算法分析）

### 5.1 问题定义

用户-物品交互矩阵 $\mathcal{Y} = \{y_{u,v}: u \in \mathcal{U}, v \in \mathcal{V}\}$。用户 $u$ 的画像 $\mathcal{V}_u = \{v \in \mathcal{V}: y_{u,v} \neq 0\}$。LLM 推荐器 $\text{LLM}_\Theta$ 的偏好函数：

$$
\mathcal{R}_u = f_{\text{LLM}_\Theta}(\mathcal{X}_{u,v}) \tag{1}
$$

其中 $\mathcal{X}_{u,v} = \mathcal{P} \oplus m_u \oplus \mathcal{M}_v^u \oplus \mathcal{C}_v^u$。$\mathcal{P}$ 为提示模板，$m_u$ 为用户特征，$\mathcal{M}_v^u$ 为已交互物品特征集，$\mathcal{C}_v^u$ 为候选物品特征集。

### 5.2 威胁模型

- **攻击者知识**：黑盒设置，无法获得模型参数、架构、智能体记忆。仅可访问公开推荐数据（物品标题、用户评论）。
- **攻击者目标**：最大化目标物品在尽可能多正常用户中的曝光。需要强迁移性，并保持隐蔽性。
- **攻击者能力**：可修改目标物品的描述（如初始记忆 $\mathcal{T}_0$）。描述通常冗长且富含上下文，易于隐藏扰动。商家可通过 API 频繁更新描述。

### 5.3 代理模块

代理框架 $f_{\text{Agent}_\Phi}(\cdot)$ 模拟真实目标系统，包含用户智能体、物品智能体、推荐智能体，均基于 LLM 和记忆组件。

### 5.4 生成模块

**对抗提示构建**：目标物品的对抗提示

$$
\widetilde{\mathcal{X}}_t = \widetilde{\mathcal{P}} \oplus \widetilde{m_u} \oplus C_p \oplus m_t \tag{2}
$$

其中 $\widetilde{\mathcal{P}}$ 为对抗提示模板，$\widetilde{m_u}$ 为通用用户记忆，$C_p$ 为流行物品特征序列，$m_t$ 为目标物品元数据。

**触发器初始化**：

$$
\widetilde{\mathcal{M}}_c = f_{\text{LLM}_\Psi}(\widetilde{\mathcal{X}}_c \oplus m_t) \tag{3}
$$

$\widetilde{\mathcal{M}}_c = \{\widetilde{m}_{t_i}\}$ 为触发器候选序列。

**质量估计**：计算每个候选的性能分数，分数为负损失：

$$
s_i^{(k-1)} = -\mathcal{L}_{\widetilde{m}_{t_i}^{(k-1)}} \tag{4}
$$

损失为自回归语言生成损失（负对数似然）：

$$
\mathcal{L}_{\widetilde{m}_{t_i}^{(k-1)}} = -\log \mathbb{P}\left(f_{\text{Agent}_\Psi}(\widetilde{\mathcal{R}}_t | \widetilde{\mathcal{X}}_t \oplus \widetilde{m}_{t_i}^{(k-1)})\right)
$$

其中 $\widetilde{\mathcal{R}}_t$ 为目标物品排名第一的 token 序列。

**特征整合**：基于 softmax 概率随机选择候选，随机交换文本切片。

**语言丰富**：使用 LLM 润色组合，保持语义并提高流畅性。

**最优触发器**：

$$
\widetilde{m}_{t_i}^* = \arg\max_{\widetilde{m}_{t_i}^*} (\{s_i^E\}) \tag{5}
$$

### 图 2：DrunkAgent 整体框架——结果描述与分析

![原文图 2：DrunkAgent 总体框架](assets/DrunkAgent/fig2.png)

**图 2：DrunkAgent 总体框架**（英文原文 PDF 第 4 页）。

论文 Figure 2 用一个端到端流程展示 DrunkAgent 如何作用于 LLM-powered recommender agent。图中可以看到，正常推荐链路以 **User Agent → Item Agent → Recommendation Agent** 为核心，Item Agent 会维护并更新自身的长期/动态 memory；攻击者则从目标物品的 **Description** 入手，在不需要访问受害模型参数的情况下构造扰动。

**图中各部分的含义：**

- **目标物品（Target Item）**：攻击不是随机改变所有物品，而是围绕一个希望被提升曝光/排序的目标 item 进行。
- **Item Agent / Memory Update**：目标 item 的智能体会在交互过程中产生并更新 memory，这正是 DrunkAgent 与传统一次性 prompt attack 的关键区别——攻击影响可以被后续交互再次检索和利用。
- **Attacker**：攻击者修改目标 item 的描述，并通过 surrogate 环境提前搜索攻击文本和策略。
- **Recommendation Agent**：最终推荐由 agentic RS 的用户建模、检索/序列建模以及 item memory 共同影响，因此污染 memory 后，攻击效果可以持续作用于后续推荐。

**结果与分析：**

这张图最重要的信息不是“攻击者把一段恶意 prompt 塞进输入”，而是揭示了 DrunkAgent 的**持久化攻击路径**：

```text
目标物品描述
    ↓
触发 Item Agent 的异常行为
    ↓
污染 / 偏移 Item Agent Memory
    ↓
后续交互继续检索被污染的 Memory
    ↓
用户偏好表示逐渐被目标物品影响
    ↓
目标物品获得更高推荐概率
```

因此，论文把攻击面从传统的“当前输入 → 当前输出”扩展到了“**当前输入 → memory update → 后续多轮推荐**”。这也是 DrunkAgent 所谓 **stealthy memory corruption** 的核心：单次文本扰动可能并不显眼，但一旦进入动态 memory，就可能形成持续影响。

### 5.5 策略模块

**扰动策略定义**：

$$
\widetilde{q} = \widetilde{q}_f \oplus \widetilde{q}_c \oplus \widetilde{q}_g \oplus \widetilde{q}_n \oplus \widetilde{q}_s \tag{6}
$$

**策略优化**：最大化期望概率，使智能体在对抗修改下执行恶意动作：

$$
\mathbb{E}_{\widetilde{q} \sim \pi_{\widetilde{q}}} \left[ \mathbb{I}(f_{\text{Agent}_\Phi}(\widetilde{\mathcal{X}}_s \oplus \widetilde{q} \oplus \widetilde{m}_t^*) = \widetilde{a}_t) \right] \tag{7}
$$

其中 $\widetilde{a}_t$ 为期望的恶意动作（目标物品智能体无法正常执行角色，无法从交互环境中感知、学习和自演化）。

最终对抗描述为 $\widetilde{m}_t^* \oplus \widetilde{q}^*$。

### 5.6 算法 1：DrunkAgent 整体优化过程

论文的 **Algorithm 1: Optimization Procedure of DrunkAgent** 给出了从公开的用户-物品交互数据与目标物品描述出发，在 surrogate module 上联合求解 **最优对抗描述触发器**与**最优对抗扰动策略**的完整流程。原论文算法输入是带有基本文本元数据（如标题、类别）的用户-物品交互矩阵 $\mathcal{Y}$，输出是目标物品 $t$ 的对抗描述。

#### Algorithm 1 逐行解释

**输入与输出**

- **Input**：用户-物品交互矩阵 $\mathcal{Y}$，以及用户、物品的基础文本元数据，例如 item title、category。
- **Output**：目标物品 $t$ 的最终对抗描述，即 $\widetilde{m}_t^* \oplus \widetilde{q}^*$。

**第一阶段：在 surrogate module 上优化 adversarial description trigger**

1. **构造对抗推荐提示**：利用论文式（2）得到目标物品对应的对抗提示 $\widetilde{\mathcal{X}}_t$。该提示把推荐任务模板、通用用户记忆、热门物品记忆以及目标物品元数据组合起来。
2. **初始化候选触发器集合**：利用式（3）让 LLM 根据目标物品描述和攻击目标生成候选集合 $\widetilde{\mathcal{M}}_c=\{\widetilde m_{t_i}\}$。这一步不是直接确定一个 trigger，而是保留多个候选进入后续迭代。
3. **进入迭代优化**：在最多 $\mathcal{E}$ 个 epoch 内持续优化；如果贪婪搜索已经稳定达到攻击目标，则可以提前结束。每个 epoch 包含三个阶段。
4. **Quality Estimation Stage（质量评估）**：对当前每个候选 trigger 计算分数 $s_i^{(k-1)}=-\mathcal{L}_{\widetilde m_{t_i}^{(k-1)}}$。论文使用自回归语言生成的负对数似然损失衡量 surrogate agent 对“目标物品排在第一位”这一目标输出的匹配程度；损失越小，score 越高。
5. **保留 Top-$n$**：按照 score 从高到低直接保留 $n$ 个最优候选，使高质量候选不会在后续随机重组中全部丢失。
6. **Feature Integration Stage（特征/文本组合）**：从剩余的 $|\widetilde{\mathcal{M}}_c|-n$ 个候选中按 Softmax 概率采样。score 越高的候选被选中的概率越大。
7. **随机切片交换**：把抽到的候选文本在随机位置切分为文本片段（例如在标点处），再在成对文本之间随机交换片段，从不同候选中组合新的词、短语和句子，以扩大搜索空间。
8. **Linguistic Enrichment Stage（语言增强）**：把上一阶段得到的组合交给 LLM 进行优化和润色，修复直接拼接可能产生的语法不自然、语义断裂等问题，同时扩展候选的语言表达空间。
9. **形成下一轮候选集**：将前面保留下来的 $n$ 个高分候选与语言增强后的 $|\widetilde{\mathcal{M}}_c|-n$ 个候选合并，重新形成 $|\widetilde{\mathcal{M}}_c|$ 个 trigger，进入下一 epoch。
10. **结束 trigger 搜索**：达到迭代上限或贪婪搜索稳定收敛后，根据式（5）从最终候选中选择 score 最大的 $\widetilde m_t^*$ 作为最优对抗描述触发器。

**第二阶段：在 surrogate module 上优化 adversarial perturbation strategy**

11. **定义对抗策略集合**：根据式（6）构造 $\widetilde q$，由五类策略组合而成：Fake Task Response、Contextual Text Switching、Segmentation Signal Augment、Malicious Task Injection、Special Character Usage。这里优化的是策略的组织与组合方式，而不是重新搜索 trigger 文本。
12. **检查目标恶意动作是否发生**：如果目标 item agent 仍能正常完成环境交互、感知并更新 memory，则继续优化策略。
13. **优化策略排列**：根据式（7）最大化在策略扰动下目标 agent 执行期望恶意动作 $\widetilde a_t$ 的概率/期望指标。
14. **获得最优策略**：当式（7）的目标达到最大值并使目标 agent 达到预期的“drunk”状态后，得到 $\widetilde q^*$。
15. **输出最终攻击描述**：Algorithm 1 最终返回 $\widetilde m_t^* \oplus \widetilde q^*$，即“最优对抗描述触发器 + 最优对抗扰动策略”。

### Algorithm 1 图示结果与分析

论文 Algorithm 1 可以看作 Figure 2 的**算法化展开**。它把整体攻击过程明确拆成两个连续优化阶段：第一阶段寻找最有效的 adversarial description trigger；第二阶段寻找能够让目标 Item Agent 进入异常状态、从而产生 memory corruption 的 perturbation strategy。

从算法结构上看，可以得到以下流程关系：

```text
输入：用户-物品交互矩阵 Y + 基础文本元数据
                    │
                    ▼
        构造目标物品推荐上下文 X~t
                    │
                    ▼
        LLM 初始化多个候选 trigger
                    │
                    ▼
       ┌── Quality Estimation ──┐
       │ 计算 NLL / score       │
       └──────────┬─────────────┘
                  ▼
             保留 Top-n
                  │
                  ▼
       Softmax 采样剩余候选
                  │
                  ▼
          Random Slice Exchange
                  │
                  ▼
          LLM Linguistic Enrichment
                  │
                  └──────→ 下一轮搜索
                              │
                              ▼
                    得到最优 trigger m~*
                              │
                              ▼
                    优化 perturbation q~
                              │
                              ▼
                  输出 m~* ⊕ q~*
```

**从结果上如何理解 Algorithm 1：**

1. **不是一次生成，而是迭代搜索。** 初始化阶段生成多个候选 trigger，之后通过质量评分、保留精英候选、随机重组和语言润色不断迭代，因此最终 trigger 是搜索得到的，而不是单次 LLM 输出。
2. **Quality Estimation 决定“哪些候选值得保留”。** NLL 越低，说明 surrogate agent 对目标推荐结果的生成概率越高，因此对应候选具有更高的攻击潜力。
3. **Feature Integration 负责扩大搜索空间。** 直接只保留 Top-n 容易过早收敛；softmax 采样 + slice exchange 让低排名但可能包含有用片段的候选仍有机会参与下一轮组合。
4. **Linguistic Enrichment 负责恢复文本质量。** 随机切片交换会破坏语法和语义连续性，因此论文再利用 LLM 进行语言优化，使攻击文本在保持攻击效果的同时更加自然。
5. **第二阶段改变优化目标。** 第一阶段解决的是“什么文本最容易让目标 item 获得推荐优势”，第二阶段解决的是“什么策略最容易让 Item Agent 的交互/记忆更新进入异常状态”。
6. **最终攻击效果来自两者组合。** $\widetilde m_t^*$ 提供内容层面的触发，$\widetilde q^*$ 提供交互策略层面的扰动，两者组合后才形成论文最终使用的对抗描述。

> **关键理解**：Algorithm 1 不是单纯的“生成一段恶意文本”。它实际上包含两个相互衔接的优化问题：第一部分寻找能够提高目标物品排序表现的文本 trigger；第二部分寻找能够干扰目标 item agent 记忆更新的 strategy。两者都先在 surrogate agentic RS 上优化，再用于黑盒目标系统，从而减少直接查询真实受害系统的需求，并提高迁移性与隐蔽性。上述流程与论文 Algorithm 1 的 20 行结构一致。它实际上包含两个相互衔接的优化问题：第一部分寻找能够提高目标物品排序表现的文本 trigger；第二部分寻找能够干扰目标 item agent 记忆更新的 strategy。两者都先在 surrogate agentic RS 上优化，再用于黑盒目标系统，从而减少直接查询真实受害系统的需求，并提高迁移性与隐蔽性。上述流程与论文 Algorithm 1 的 20 行结构一致。

## 5.7 方法流程补充：从触发器到记忆污染

论文的核心不是一次性的 prompt 注入，而是让**目标物品智能体的动态记忆发生持久性偏移**。完整流程可概括为：

```text
公开物品/用户文本
      ↓
Surrogate Agentic RS
      ↓
生成模块：构造 adversarial trigger
      ↓
贪婪搜索：质量评估 → 特征重组 → LLM 语言润色
      ↓
策略模块：优化“让目标智能体喝醉”的交互策略
      ↓
trigger ⊕ strategy
      ↓
目标物品描述被修改
      ↓
目标 item agent 在交互过程中错误更新 memory
      ↓
后续推荐持续读取被污染 memory
      ↓
目标物品曝光率上升
```

这里有两个需要区分的优化对象：**trigger** 负责让目标物品在推荐排序中具有较强的目标导向语义；**strategy** 负责破坏目标 item agent 的正常记忆更新，使 trigger 更容易进入并长期留在动态 memory 中。论文明确将二者共同放在 surrogate module 上优化，而不是直接频繁查询真实黑盒受害系统。

## 5.8 生成模块的三个阶段

### 5.8.1 推荐上下文构建

对抗提示模板由四部分组成：

1. **Style Definition**：定义推荐角色与领域风格；
2. **Task Goal**：明确“让目标物品排名靠前”的任务目标；
3. **Recommendation Instruction**：要求比较用户偏好与候选物品；
4. **Format Constraint**：限定输出格式，减少无关生成。

为了适应黑盒设置，论文没有假设攻击者能够获得真实用户/物品 agent 的优化后 memory，而是使用**通用用户 memory + 流行物品 memory**近似真实上下文。流行物品被作为候选参照，是因为它们通常更容易出现在推荐列表前部；如果目标物品能够超过这些高频候选，推广成功的可能性更高。

### 5.8.2 贪婪搜索不是简单的 beam search

每轮搜索包含三个阶段：

- **Quality Estimation**：计算每个 trigger candidate 使目标输出出现的负对数似然，保留 top-$n$；
- **Feature Integration**：根据 softmax 后的候选分数随机采样其余候选，并在标点等位置切片、交换文本片段，组合不同候选中的有效语义特征；
- **Linguistic Enrichment**：利用 LLM 对重组后的文本进行润色，恢复连贯性、自然性和语义完整性。

因此它兼顾了 exploitation（保留高分候选）与 exploration（随机重组其他候选），最后得到最优 trigger $\widetilde m_t^*$。论文算法 1 明确把这一过程迭代至攻击目标稳定达到或搜索终止。

## 5.9 策略模块的具体含义

论文设计五类“让目标 agent 喝醉”的扰动策略：

| 策略 | 作用 |
|---|---|
| Fake Task Response | 伪造任务已经完成的响应，使 agent 误以为原有记忆协同任务已经结束 |
| Contextual Text Switching | 引导 agent 忽略其他上下文，转而执行攻击者注入的当前上下文 |
| Segmentation Signal Augment | 加入分段/边界信号，让模型将恶意内容解释为一个独立任务或高优先级片段 |
| Malicious Task Injection | 直接插入与原任务不一致的恶意任务目标 |
| Special Character Usage | 利用特殊字符影响上下文边界、注意力或任务解析 |

策略优化的目标不是让模型直接输出某个推荐，而是让目标 item agent **无法正常完成 memory update / collaborative optimization**，从而增加 adversarial trigger 被写入或保留下来的机会。论文把五类策略的组合记为 $\widetilde q=\widetilde q_f\oplus\widetilde q_c\oplus\widetilde q_g\oplus\widetilde q_n\oplus\widetilde q_s$，并搜索其排列/组合，使目标恶意动作的发生概率最大。

## 5.10 威胁模型与现实性补充

DrunkAgent 的黑盒假设需要特别记录：

- 攻击者**不知道**受害模型参数、内部 memory、具体 agent architecture 或底层 LLM；
- 攻击者只需要公开可获得的推荐数据，例如物品标题、用户评论等；
- 攻击者可以修改目标物品描述，因此攻击面主要位于 item-side textual description；
- 攻击目标是**targeted item promotion**，而不是单纯降低整体推荐性能；
- 攻击同时追求 transferability 与 stealthiness。

这一点使 DrunkAgent 与传统 shilling/data-poisoning 以及仅针对固定 prompt 的文本攻击有明显区别：传统方法往往污染静态 embedding，固定 prompt 攻击则难以适应持续变化的 agent memory。

## 5.11 实验设置补充

论文为了控制 LLM API 成本，在三个真实数据集上进行采样：

| 数据集 | 原始用户 | 原始物品 | 原始交互 | 稀疏度 | 实验采样 |
|---|---:|---:|---:|---:|---|
| CDs & Vinyl | 112,395 | 73,713 | 1,443,755 | 99.98% | 100 用户 / 800 交互 / 777 物品 |
| Office Products | 101,501 | 27,965 | 800,357 | 99.97% | 99 用户 / 693 交互 / 619 物品 |
| Musical Instruments | 27,530 | 10,620 | 231,392 | 99.92% | 98 用户 / 588 交互 / 483 物品 |

使用 leave-one-out 思路进行 9:1 train/test 划分。论文明确说明，这种采样主要是考虑昂贵 API 调用，同时用于考察稀疏与冷启动场景；扩展到完整大规模数据集被留作未来工作。

## 5.12 攻击迁移性结果：当前笔记缺失的定量信息

论文 Table 1 的完整结果非常庞大，按 AgentCF、AgentRAG、AgentSEQ 分别报告 HR@1/2/3 与 NDCG@1/2/3。一个值得单独记录的结果是 AgentSEQ 上，DrunkAgent 在三个数据集均明显超过 benign 与其他 baseline。例如：

- CDs & Vinyl：HR@1 = **0.4949**；
- Office Products：HR@1 = **0.1429**；
- Musical Instruments：HR@1 = **0.1443**。

在同一表中，其他 character/word/sentence-level attacks 在多数设置下明显更低。这说明 DrunkAgent 的优势并不只是“改变文本”，而是通过动态 memory corruption 使触发器持续发挥作用。

## 5.13 攻击成本与查询次数

论文 Table 3 给出了不同 baseline 修改目标描述所需的查询次数示例：DeepwordBug 34、PuncAttack 45、TextFooler 55、BertAttack 23、TrivialInsertion 100、ChatGPTAttack 100；Benign 不需要攻击查询。DrunkAgent 的核心设计则是在 surrogate 上优化，从而减少直接对真实黑盒系统反复试探的需求。

## 5.14 局限与复现注意事项

1. **数据规模受 API 成本限制**：论文只在三个数据集的采样子集上做主要实验，完整规模扩展仍未验证。
2. **攻击面依赖可修改的 item description**：如果平台对商品描述实行强校验、版本锁定或人工审核，攻击能力会下降。
3. **surrogate-target gap 仍然存在**：方法依赖代理模型与真实黑盒系统之间的可迁移性，因此不能保证每个商业系统都具有相同效果。
4. **评价主要关注 targeted promotion**：不能直接把结果解释为所有形式的 agent security 都已解决。
5. **防御仍是开放问题**：论文展示了现有释义防御的不足，因此后续应重点研究 memory provenance、memory write validation、异常交互检测与持久化污染清除。

## 5.15 与现有笔记相比需要更新的事实

- 当前公开版本为 **arXiv:2503.23804v3（2025-10-21）**；同时论文已经作为 WWW 2026 论文发表。
- 论文正式版本页码为 WWW 2026, pp. 6853–6864；会议原计划日期后来发生调整，因此不建议继续把旧的 April 13–17 日期当作唯一会议日期。
- 当前没有检索到作者明确公开的官方复现仓库，因此“代码复现”部分仍应保持谨慎表述；第三方项目对 DrunkAgent 的适配不能等同于作者官方代码。

## 6. 实验与环境、数据集、参数配置

### 6.1 数据集

使用三个文本密集型真实数据集（Amazon Review Data）：**CDs & Vinyl**、**Office Products**、**Musical Instruments**。额外在 **Yelp** 上评估跨域泛化性。目标物品从每个数据集的物品池中随机采样。

数据集统计（Table 2）：

| 数据集 | 用户数 | 物品数 | 交互数 | 稀疏度 |
|---|---:|---:|---:|---:|
| CDs & Vinyl | 112,395 | 73,713 | 1,443,755 | 99.98% |
| Office Products | 101,501 | 27,965 | 800,357 | 99.97% |
| Musical Instruments | 27,530 | 10,620 | 231,392 | 99.92% |

实验子集：分别随机采样 100 用户/800 交互/777 物品、99 用户/693 交互/619 物品、98 用户/588 交互/483 物品。留一法划分训练/测试 9:1。

### 6.2 基线攻击方法

- 字符级：DeepwordBug、PuncAttack
- 词级：TextFooler、BertAttack
- 句子级：TrivialInsertion、ChatGPTAttack

所有基线扰动目标物品描述，并在相同代理模型上优化。

### 6.3 目标推荐系统（受害模型）

- **AgentCF**：标准协同过滤范式
- **AgentRAG**：检索增强
- **AgentSEQ**：序列推荐

### 6.4 评估指标

- 攻击迁移性：HR@K、NDCG@K（K=1,2,3）
- 攻击隐蔽性：整体性能变化（HR@3）、文本困惑度（GPT-Neo）
- 防御策略：使用 OpenAI GPT-o1 进行释义防御

## 7. 完整实验结果表格

### 表 1：攻击迁移性（论文 Table 1）

不同攻击针对多种黑盒受害 LLM 智能体 RS 在真实数据集上的 HR@K 和 NDCG@K。加粗为最佳。DrunkAgent 取得最高值。

### 表 5：消融研究（论文 Table 5）

在 AgentCF 上，移除贪婪优化算法（$\mathcal{G}_r$）或策略模块（$\mathcal{S}_t$）的性能变化：

| 数据集                 | $\mathcal{G}_r$ | $\mathcal{S}_t$ | HR@1       | HR@2       | HR@3       | NDCG@1     | NDCG@2     | NDCG@3     |
| ------------------- | --------------- | --------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| CDs & Vinyl         | ×               | ✓               | 0.0808     | 0.1313     | 0.1919     | 0.0808     | 0.1127     | 0.1430     |
|                     | ✓               | ×               | 0.0707     | 0.1212     | 0.2222     | 0.0707     | 0.1026     | 0.1531     |
|                     | ✓               | ✓               | **0.4040** | **0.4141** | **0.4343** | **0.4040** | **0.4104** | **0.4205** |
| Office Products     | ×               | ✓               | 0.1327     | 0.1429     | 0.1531     | 0.1327     | 0.1391     | 0.1442     |
|                     | ✓               | ×               | 0.0612     | 0.0714     | 0.1122     | 0.0612     | 0.0677     | 0.0881     |
|                     | ✓               | ✓               | **0.2449** | **0.2449** | **0.2551** | **0.2449** | **0.2449** | **0.2500** |
| Musical Instruments | ×               | ✓               | 0.1546     | 0.1649     | 0.1753     | 0.1546     | 0.1611     | 0.1663     |
|                     | ✓               | ×               | 0.0309     | 0.0619     | 0.1237     | 0.0309     | 0.0504     | 0.0814     |
|                     | ✓               | ✓               | **0.2268** | **0.2268** | **0.2371** | **0.2268** | **0.2268** | **0.2320** |

### 表 6：成功使目标智能体“喝醉”的成功率（论文 Table 6）

| 数据集 | 成功率 |
|---|---:|
| CDs & Vinyl | 100% |
| Office Products | 100% |
| Musical Instruments | 100% |

### 表 3：基线对抗描述示例（论文 Table 3）

| 攻击方法 | 目标物品对抗描述 | 查询次数 |
|---|---|---:|
| Benign (Default) | The CD is called “Counterparts”. The category of this CD is: “Rock; Progressive; Progressive Metal”. | - |
| DeepwordBug | Te hCD is called “Counterparts”. The category of this CD is: “Rock; Progressive; Progressive Metal”. | 34 |
| PuncAttack | The CD i-s called “Counterparts”. The category of this CD is: “Rock; Progressive; Progressive Metal-1”. | 34 |
| TextFooler | The CD is titled “Counterparts”. The category of this CD is: “Rock; Gradually; Progressive Metals”. | 55 |
| BertAttack | The CD is called “Counterparts”. The category of each CD is: “Rock; Progressive; Progressive Metal”. | 55 |
| TrivialInsertion | The CD is called “Counterparts”. The category of this CD is: “Rock; Progressive; Progressive Metal”. Amazing!!! | 100 |
| ChatGPTAttack | “Counterparts” CD: Perfect blend of rock, progressive, and metal. Experience the wonderful sound that will elevate your music collection. | 100 |

### 表 4：受害模型提示模板示例（论文 Table 4）

| 受害模型 | 推荐提示模板示例 |
|---|---|
| AgentCF | “role”: “system”, “content”: “You are a CD recommender system. Here is a user's self-introduction, expressing his/her preferences and dislikes: user_agent_memory. Now, you are considering to sort ten candidate CDs that are listed as follows: "CD title: candidate_item_title, where its features: candidate_item_memory"” |
| AgentRAG | “role”: “system”, “content”: “You are a CD recommender system. Here is a user's self-introduction, expressing his/her preferences and dislikes: retrieval_user_agent_memory\nuser_agent_memory. Now, you are considering to sort ten candidate CDs that are listed as follows: "CD title: candidate_item_title, where its features: candidate_item_memory"” |
| AgentSEQ | “role”: “system”, “content”: “You are a CD recommender system. Here is a user's self-introduction, expressing his/her preferences and dislikes: user_agent_memory\nuser_agent_memory. Now, you are considering to sort ten candidate CDs that are listed as follows: "CD title: candidate_item_title, where its features: candidate_item_memory"” |

### 图 3：攻击通用性（论文 Figure 3）

![原文图 3：不同推荐智能体与数据集上的攻击通用性](assets/DrunkAgent/fig3.png)

**图 3：不同推荐智能体与数据集上的攻击通用性**（英文原文 PDF 第 7 页）。

展示 DrunkAgent 在不同模型和目标物品上的 HR@1 (NDCG@1)，表明其跨样本可迁移性和触发器通用性。

### 图 4：攻击隐蔽性（论文 Figure 4）

![原文图 4：攻击前后整体推荐性能的变化](assets/DrunkAgent/fig4.png)

**图 4：攻击前后整体推荐性能的变化**（英文原文 PDF 第 8 页）。

所有受害模型在所有真实数据集上攻击前后推荐性能差异的总体分布。DrunkAgent 不引起整体性能剧烈变化，表明隐蔽。

### 图 5：攻击不可感知性（论文 Figure 5）

![原文图 5：扰动文本的困惑度比较](assets/DrunkAgent/fig5.png)

**图 5：扰动文本的困惑度比较**（英文原文 PDF 第 8 页）。

扰动文本在真实数据集上的困惑度。DrunkAgent 困惑度最低，表明文本更自然、更隐蔽。

### 图 6：对防御机制的鲁棒性（论文 Figure 6）

![原文图 6：面对改写防御的攻击鲁棒性](assets/DrunkAgent/fig6.png)

**图 6：面对改写防御的攻击鲁棒性**（英文原文 PDF 第 8 页）。

使用 GPT-o1 释义防御后，DrunkAgent 仍可迁移，甚至在某些情况下增强。

### 图 7：跨域泛化（论文 Figure 7）

![原文图 7：跨域攻击泛化](assets/DrunkAgent/fig7.png)

**图 7：跨域攻击泛化**（英文原文 PDF 第 12 页）。

在 Yelp 数据集上，DrunkAgent 仍能有效推广目标物品。

## 8. 结果分析

- **攻击迁移性（RQ1）**：DrunkAgent 在不同受害模型和数据集上均取得最高 HR 和 NDCG，表明优异迁移性。所有基线性能相近，但不能保证始终达成攻击目标，例如在 AgentCF 上对 Musical Instruments 的表现均差于良性状态。这说明策略模块和触发器迁移性的重要性。
- **攻击通用性**：DrunkAgent 生成的对抗描述可植入完全不同的目标物品，HR@1 和 NDCG@1 表明其跨目标物品的通用性。
- **攻击隐蔽性**：DrunkAgent 不引起整体性能剧烈变化，不易被用户和平台检测。困惑度最低，文本连贯流畅、语义有意义且隐蔽。
- **防御鲁棒性**：使用释义防御后，DrunkAgent 仍可迁移且优于基线。有趣的是，释义在某些情况下增强攻击，可能因为新词/短语/句子提高了对抗文本的语义意义。
- **消融研究**：移除策略模块导致性能大幅下降，说明记忆更新对非定制攻击有一定鲁棒性。移除贪婪优化算法性能适度下降，得益于初始化有效性。
- **参数敏感性**：策略模块贡献大于优化算法。增加优化迭代次数，成功使目标智能体“喝醉”的成功率保持 100%。DrunkAgent 对其他攻击参数不敏感。
- **跨域泛化**：在 Yelp 上仍高度有效，确认攻击的迁移性和记忆扰动的跨域有效性。

## 9. 结论

论文提出 DrunkAgent，一种针对 LLM 驱动智能体 RS 的新型攻击框架，通过有效扰动目标智能体的记忆来推广目标物品。在真实数据集上的大量实验表明，其在黑盒设置下具有最先进的迁移性和隐蔽性。更重要的是，研究发现揭示了当前智能体 RS 中未知的安全漏洞，强调了记忆机制可被真实世界输入影响。这项工作既提供了有效的攻击方法，也提供了更好理解智能体漏洞的视角，为构建更鲁棒和可信的推荐智能体提供了指导。

# 二、代码复现

**无代码**。论文声明源代码将通过官方仓库发布，但当前尚未公开。实验基于 PyTorch 和 LLM API（如 OpenAI GPT-o1、GPT-Neo 等）实现，但未开源。因此无环境配置、源代码分析、bug 修复、结果复现等内容。

# 三、总结

DrunkAgent 是首个针对 LLM 驱动推荐智能体的黑盒记忆污染攻击框架。其核心创新在于将攻击面从静态输入转移到动态记忆更新过程，通过生成模块构造语义隐蔽的对抗触发器，通过策略模块使目标智能体“喝醉”从而无法正常更新记忆，最终实现目标物品的推广。实验在三个 Amazon 数据集和 Yelp 上，针对 AgentCF、AgentRAG、AgentSEQ 三种受害系统，验证了攻击的迁移性、通用性、隐蔽性和防御鲁棒性。消融研究证实了贪婪优化和策略模块的有效性。该工作揭示了智能体推荐系统中记忆机制的安全漏洞，为未来防御研究提供了方向。局限在于当前仅在采样子集上评估，扩展到更大数据集留作未来工作。
