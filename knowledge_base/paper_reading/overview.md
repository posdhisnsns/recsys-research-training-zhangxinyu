# 推荐系统

## 基于扩散模型的序列推荐

---

### 2026/07/21 - TDRec: Beyond Static Diffusion - 显式建模时间模式(SIGIR 2026)

#### 一、研究问题:扩散模型中的时间依赖未被显式建模

现有基于扩散模型的序列推荐方法,将时间上下文仅作为外部条件信号注入,未将时间转换真正融入扩散动态过程。这导致模型难以捕捉用户兴趣随时间的精细演化。

#### 二、研究目的

将**时间进程(Temporal Progression)** 显式融入扩散的前向和反向过程,在每个扩散步骤中同时完成:1 位置隐表示的噪声注入;2 与前序隐表示的时间混合。

#### 三、研究内容

1. **TDRec(Temporally-aware Diffusion for sequential Recommendation)**:提出将时间进展同时融入前向扩散和反向去噪过程。
2. **时间混合的闭式解推导**:为时间混合过程推导出闭式(closed-form)解,使训练复杂度相对于序列长度达到 **O(1)**,支持高效的并行训练。
3. **DDPM-like 反向过程**:证明存在对应的 DDPM-like 反向过程和重参数化目标,保证高效优化和采样,不引入额外计算开销。
4. 在 3 个公开数据集上验证。

#### 四、研究结果

实验表明 TDRec 在准确性指标上优于现有扩散推荐基线,证明了将时间转换显式融入扩散动态的有效性。

> **出处**:Wu, Yao; Liu, Chengyi; Fan, Wenqi; Zhang, Rui; Moffat, Alistair; Scholer, Falk; Bast, Hannah; Najork, Marc; Zhang, Min
> **会议**:SIGIR 2026(Proceedings of the 49th International ACM SIGIR Conference)
> **DOI**:10.1145/3805712.3809536
> **代码**:https://github.com/wuyaoericyy/TDRec

---

### 2026/07/21 - M3BSR:多模态多行为条件扩散特征去噪(SIGIR 2025)

#### 一、研究问题:多模态多行为序列推荐中的三重挑战

1. 用户对不同行为下不同物品模态的关注存在差异,难以有效刻画跨行为的模态偏好;
2. 用户行为中隐含噪声(如误点)难以有效消除;
3. 多模态表征中存在模态噪声,进一步影响用户偏好的准确建模。

#### 二、研究目的

提出 **M3BSR(Multi-Modal Multi-Behavior Sequential Recommendation)**,利用条件扩散模型对多模态特征进行去噪,有效融合多行为模式,提升序列推荐准确性。

#### 三、研究内容

1. **条件扩散模态去噪层(Conditional Diffusion Modality Denoising Layer)**:对多模态表征中的噪声进行条件去噪;
2. **多模态多行为偏好建模**:根据不同行为(点击、收藏、购买等)建模用户对物品不同模态的差异化关注;
3. 在多行为、多模态场景下进行实验验证。

#### 四、研究结果

M3BSR 在多模态多行为场景下优于现有基线,有效缓解了行为噪声和模态噪声对推荐准确性的影响。

> **出处**:Cui, Xiaoxi; Lu, Weihai; Tong, Yu; Li, Yiheng; Zhao, Zhejun
> **会议**:SIGIR 2025
> **DOI**:10.1145/3726302.3730044
> **代码**:无

---

### 2026/07/21 - InDiRec:意图感知扩散 + 对比学习(SIGIR 2025)

#### 一、研究问题:随机数据增强破坏序列中的潜在意图信息

现有对比学习方法通过随机数据增强生成多个视图,但用户购买商品时通常有特定意图(如买衣服送人、买化妆品美容)。随机增强可能引入噪声,破坏原始交互序列中隐含的意图信息,误导模型关注无关特征、扭曲表征空间。

#### 二、研究目的

提出 **InDiRec(Intent-aware Diffusion with Contrastive Learning for Sequential Recommendation)**,生成与用户购买意图对齐的物品序列,为对比学习提供更可靠的增强视图。

#### 三、研究内容

1. **意图聚类**:对序列表征使用 K-means 进行意图聚类,构建意图引导信号;
2. **条件扩散模型**:检索目标交互序列的意图表征,引导条件扩散模型生成符合用户意图的物品序列;
3. **意图感知的对比学习**:用生成的意图对齐序列作为对比学习视图,提升模型对用户真实行为模式和意图的捕捉能力;
4. 在 SIGIR 2025 基准数据集上验证。

#### 四、研究结果

InDiRec 通过意图感知增强和对比学习,显著提升了序列推荐的准确性,有效解决了随机增强破坏意图信息的问题。

> **出处**:Qu, Yuanpeng; Nobuhara, Hajime
> **会议**:SIGIR 2025
> **DOI**:10.1145/3726302.3730010
> **代码**:https://github.com/qyp9909/InDiRec

---

### 2026/07/21 - 扩散模型的多样化序列推荐(SIGIR 2025)

#### 一、研究问题:DM-based SRS 的两个多样性瓶颈

1. 现有方法依赖非多样化的用户偏好作为引导信号,限制了扩散网络生成多样化物品的能力;
2. 使用同质的扩散推理机制生成下一物品,只能适应用户的主要偏好,忽视了用户对不同类型物品的异质偏好。

#### 二、研究目的

在保证准确性的同时,通过扩散模型提升序列推荐的多样性,生成更符合用户多样化兴趣的推荐列表。

#### 三、研究内容

1. 研究扩散模型在学习-生成范式下的多样性优势;
2. 针对用户异质偏好设计异质的扩散推理机制;
3. 在准确性(Accuracy)和多样性(Diversity)两个维度同时评估。

#### 四、研究结果

所提方法在 SIGIR 2025 多样化推荐任务上同时提升了推荐准确性和多样性,验证了扩散模型在多样化推荐中的潜力。

> **出处**:Cai, Zhuo; Wang, Shoujin; Chu, Victor W.; Naseem, Usman; Wang, Yang; Chen, Fang
> **会议**:SIGIR 2025
> **DOI**:10.1145/3726302.3730109
> **代码**:无

---

### 2026/07/21 - DMMD4SR:多层次多模态去噪扩散(ACM 2025)

#### 一、研究问题:多模态序列推荐中三重噪声被忽视

多模态序列推荐(MMSR)利用丰富的物品特征,但预训练模型(PTM)衍生的表征存在三类噪声:1 PTM 训练数据与推荐场景之间的领域偏移;2 模态内与兴趣无关的噪声(如无关的背景细节);3 交互不确定性使模态融合更加复杂。

#### 二、研究目的

提出 **DMMD4SR(Diffusion Model-based Multi-level Multimodal Denoising for Sequential Recommendation)**,利用扩散模型的去噪能力,以渐进式多层次策略解决上述三重噪声挑战。

#### 三、研究内容

1. **多层次去噪策略**:逐层消除领域偏移噪声和上下文相关兴趣无关噪声;
2. **不确定性引导的模态去噪融合层**:在考虑交互不确定性的同时自适应融合净化后的多模态表征;
3. 扩散模型作为去噪工具,与多模态特征净化有机结合;
4. 在 ACM MM 2025 基准数据集上与 SOTA 基线对比验证。

#### 四、研究结果

DMMD4SR 在多模态序列推荐任务上显著优于 SOTA 基线,验证了多层次去噪策略的有效性。代码开源:https://github.com/luweihai/DMMD4SR

> **出处**:Lu, Weihai; Yin, Li
> **会议**:ACM Multimedia 2025
> **DOI**:10.1145/3746027.3755861
> **代码**:https://github.com/luweihai/DMMD4SR

---

### 2026/07/21 - LSGM4Rec:LLM 增强的引导扩散多模态序列推荐(ACM 2025)

#### 一、研究问题:现有扩散推荐方法的两大不足

1. 未能利用多模态知识构建具有良好分布特性和语义丰富信息的物品表征;
2. 主要依赖离散扩散过程,导致误差累积严重、时间效率低、生成可控性受限。

#### 二、研究目的

提出 **LSGM4Rec**,将大语言模型(LLMs)与高级多模态连续扩散过程结合,提升扩散生成质量,实现高质量、可控的多模态序列推荐。

#### 三、研究内容

1. **LLM 增强的多模态表征**:利用 LLM 补充物品的语义信息和结构化分布特性;
2. **连续扩散过程**:替代离散扩散,降低误差累积,提升时间效率和可控性;
3. 在 ACM MM 2025 基准数据集上验证。

#### 四、研究结果

LSGM4Rec 在多模态序列推荐任务上优于现有扩散推荐方法,有效提升了生成质量和推荐准确性。

> **出处**:Song, Te; Qi, Lianyong; Liu, Weiming; Wang, Fan; Xu, Xiaolong; Zhang, Xuyun; Beheshti, Amin; Hu, Hongsheng; Cao, Yang
> **会议**:ACM Multimedia 2025
> **DOI**:10.1145/3746027.3755544
> **代码**:无

---

### 2026/07/21 - ADRec:解锁扩散模型在序列推荐中的潜力(KDD 2025)

#### 一、研究问题:嵌入坍塌(Embedding Collapse)

现有基于扩散的序列推荐模型存在一个常被忽视的问题:在扩散过程中,所有物品嵌入趋于收敛到同一区域,物品区分度丧失。

#### 二、研究目的

提出 **ADRec(Independent Noise Per-token Diffusion for Sequential Recommendation)**,通过为每个物品 token 独立施加噪声,从根本上缓解嵌入坍塌问题,实现更有效且简洁的扩散推荐。

#### 三、研究内容

1. **独立噪声机制**:对每个物品 token 独立施加噪声,避免传统全局扩散导致的嵌入坍塌;
2. **三阶段训练策略**:分阶段训练,兼顾去噪质量与表征区分度;
3. 基于 RecBole 框架实现,代码完整开源;
4. 在 6 个数据集上验证。

#### 四、研究结果

相比最优扩散基线,ADRec 在 HR@20 提升 **15.45%**,NDCG@20 提升 **13.02%**,训练时间平均减少 **70.98%**。

> **出处**:Chen, Jialei; Xu, Yuanbo; Jiang, Yiheng(吉林大学)
> **会议**:KDD 2025
> **DOI**:10.1145/3711896.3737172
> **代码**:https: //doi.org/10.5281/zenodo.15470542

---

### 2026/07/21 - HorizonRec:Align-for-Fusion - 三重偏好双导向扩散(KDD 2025)

#### 一、研究问题:跨域序列推荐中细粒度偏好融合缺失

现有跨域序列推荐(CDSR)方法遵循"先对齐后融合"范式,仅在表征层面跨多域对齐并机械组合,忽视了对领域特定偏好的细粒度融合。

#### 二、研究目的

提出 **HorizonRec(Align-for-Fusion)**,通过双导向扩散模型(Dual-oriented DMs)协调三重偏好(Triple Preferences),在扩散分布匹配层面实现跨域偏好的精细融合。

#### 三、研究内容

1. **对齐-融合新范式**:替代传统的"对齐-融合"两阶段范式;
2. **三重偏好建模**:显式建模用户在跨域场景下的三种偏好维度;
3. **混合条件分布检索策略**:利用分布检索结合混合条件,解决现有 DM 推荐器中随机噪声导致的不稳定问题;
4. 在跨域序列推荐数据集上验证。

#### 四、研究结果

HorizonRec 在跨域序列推荐任务上优于现有 CDSR 方法,有效解决了细粒度偏好融合和扩散不稳定问题。

> **出处**:Zha, Yongfu; Dong, Xinxin; Ma, Haokai; Yang, Yonghui; Wang, Xiaodong
> **会议**:KDD 2026
> **DOI**:10.1145/3770854.3780258
> **代码**:https://github.com/YongfuZha/HorizonRec

---

### 2026/07/21 - FatsMB:从 agnostic 到 specific - 潜偏好扩散(KDD 2025)

#### 一、研究问题:行为无关偏好难以支撑行为特定的推荐

多行为序列推荐(MBSR)需要学习用户多行为序列中动态且异质的交互关系,从而捕捉目标行为下的用户偏好。但现有方法采用单向建模将辅助行为映射到目标行为,忽略了支撑用户决策的潜在偏好。偏好评分的判别式范式无法捕捉从低熵行为到高熵项目的不对称不确定性。

#### 二、研究目的

提出 **FatsMB**,基于扩散模型引导偏好生成从"行为无关"向"行为特定"转变,在潜空间中实现多样且准确的多行为序列推荐。

#### 三、研究内容

1. **多行为自编码器(MBAE)**:构建统一的用户潜在偏好空间,促进不同行为之间的交互与协作;
2. **RoPE 编码(BaRoPE)**:进行多源信息融合;
3. **行为感知框架**:在行为感知框架下实现等效贡献;
4. **潜空间偏好转移**:在潜空间中进行针对目标行为的偏好转移,结合信息丰富的先验知识;
5. **多条件引导层归一化(MCGLN)**:用于去噪处理;
6. 在多个真实世界数据集上广泛验证。

#### 四、研究结果

FatsMB 在多行为序列推荐任务上显著优于现有方法,证明了从"行为无关"到"行为特定"偏好引导的有效性。

> **出处**:Yang, Ruochen; Li, Xiaodong; Sheng, Jiawei; Cao, Jiangxia; Lin, Xinkui; Wang, Shen; Yang, Shuang; Liu, Zhaojie; Liu, Tingwen
> **会议**:KDD 2026
> **DOI**:10.1145/3770854.3780285
> **代码**:https://github.com/OrchidViolet/FatsMB

---

### 2026/07/21 - AuxDiff:辅助信息挖掘-探索 + 高效采样增强扩散推荐(AAAI 2025)

#### 一、研究问题:引导信号不足 + 推理速度慢

现有基于扩散模型的方法:1 缺乏有效的用户序列表征来引导生成过程,影响推荐效果;2 忽略慢速推理的缺陷,严重限制实际应用价值。

#### 二、研究目的

在获取有效生成引导信号的同时,通过高效采样机制加速推理,使扩散推荐方法兼具高质量和高效率。

#### 三、研究内容

1. **辅助信息挖掘-探索模块**:利用辅助信息构建更有效的用户序列表征作为扩散生成引导;
2. **高效采样机制**:设计快速的推理采样策略,减少多步去噪带来的计算开销;
3. 在 AAAI 2025 基准数据集上验证。

#### 四、研究结果

所提方法在推荐准确性上优于现有扩散推荐基线,同时显著加速了推理过程,提升了方法的实用价值。

> **出处**:Song, Te; Qi, Lianyong; Liu, Weiming; Wang, Fan; Xu, Xiaolong; Zhang, Xuyun; Beheshti, Amin; Zhou, Xiaokang; Dou, Wanchun
> **会议**:AAAI 2025
> **DOI**:10.1609/AAAI.V39I12.33370
> **代码**:无

---

### 2026/07/21 - GlobalDiff:全局扩散增强序列推荐(AAAI 2025)

#### 一、研究问题:局部序列中的不一致物品误导序列模型

现有序列推荐模型大多基于序列模型,容易被局部序列中的不一致物品(stochastic behaviors)误导,忽视了整个物品空间的全局非序列数据结构。

#### 二、研究目的

提出 **GlobalDiff**,利用扩散模型恢复物品空间的全局非序列数据结构,并补偿局部序列上下文,使序列模型既能利用局部序列信息,又能借助全局数据结构增强推荐质量。

#### 三、研究内容

1. **训练构造(Training Construction)**:设计全局扩散的训练方式;
2. **引导反向近似器(Guided Reverse Approximator)**:将全局扩散与序列上下文有机结合;
3. **推理集成(Inference Ensemble)**:在推理阶段无缝集成扩散信号与序列模型输出;
4. 即插即用(Plug-and-Play)框架,可增强多种先进序列模型。

#### 四、研究结果

GlobalDiff 在多个数据集上对先进序列模型带来平均 **9.67%** 的提升,证明了全局数据结构对序列推荐的重要价值。

> **出处**:Luo, Mingxuan; Li, Yang; Lin, Chenxi
> **会议**:AAAI 2025
> **DOI**:10.1609/AAAI.V39I12.33341
> **代码**:https://github.com/XMUDM/GlobalDiff

---

### 2026/07/21 - ADARec:解坍塌用户意图 - 自适应扩散增强 + MoE(AAAI 2026)

#### 一、研究问题:从稀疏数据中重建用户意图的层次表示

序列推荐依赖丰富的用户交互数据,常面临数据稀疏问题。现有数据增强方法要么引入相似数据(使语义聚类区分能力受限),要么引入多样性数据(破坏用户真实意图),导致推荐结果偏离实际需求。

#### 二、研究目的

提出 **ADARec(ADAec)**,利用完整的分步去噪过程,从单个稀疏序列中重建用户意图的层次表示,实现自适应的扩散增强。

#### 三、研究内容

1. **自适应深度控制器(ADC)**:智能确定每个用户序列的最佳扩散深度,避免对所有序列应用相同增强带来的成本问题;
2. **分层扩散增强(HDA)模块**:为每个用户序列生成分层明确的意图层次结构(同时包含粗粒度和细粒度意图);
3. **分层解析专家混合(HP-MoE)模块**:专门处理意图层次结构中不同层级的表征;
4. 在 Beauty, Sports, Toys, Yelp 四个数据集上与其他方法进行比较,平均提升约 3%,面对极度稀疏序列(数据量<5)时能达到约 10%的提升。

#### 四、研究结果

ADARec 在标准基准测试和稀疏序列上均优于最先进的方法,证明了其从稀疏数据中重建分层意图表示的能力。

> **出处**:Cui, Xiaoxi; Zhao, Chao; Cheng, Yurong; Zhou, Xiangmin; Koenig, Sven; Jenkins, Chad; Taylor, Matthew E.
> **会议**:AAAI 2026
> **DOI**:10.1609/AAAI.V40I17.38481
> **代码**:https://github.com/ZhaoChao52/ADARec

---

### 2026/07/21 - DiQDiff:扩散序列推荐的非同质化量化引导(WWW 2025)

#### 一、研究问题:扩散推荐中的两大核心缺陷

1. 用户序列在长度和内容上异质,且存在由随机用户行为带来的噪声,用这样的序列作为引导可能阻碍 DM 准确理解用户兴趣;
2. DM 易受数据偏置影响,倾向于生成主导训练数据集的热门物品,无法满足不同用户的个性化需求。

#### 二、研究目的

提出 **DiQDiff(Distinguished Quantized Guidance for Diffusion-based Sequence Recommendation)**,提取稳健的引导信号来理解用户兴趣,并生成具有区分度的物品以满足用户的个性化偏好。

#### 三、研究内容

1. **语义向量量化(Semantic Vector Quantization, SVQ)**:提取稳健引导,理解用户兴趣;
2. **区分性物品生成**:在扩散模型内为不同用户生成个性化、区分度高的目标物品;
3. 针对序列异质噪声和数据热门偏置两个问题提出系统性解决方案;
4. 在 WWW 2025 基准数据集上验证。

#### 四、研究结果

DiQDiff 在序列推荐任务上有效解决了热门偏置和序列噪声问题,显著提升了推荐的准确性和个性化程度。

> **出处**:Mao, Wenyu; Liu, Shuchang; Liu, Haoyang; Liu, Haozhe; Li, Xiang; Hu, Lantao
> **会议**:The Web Conference(WWW)2025
> **DOI**:10.1145/3696410.3714955
> **代码**:无

---

## 生成式推荐与大推荐模型(Scaling Law / token-based / 推理式)

---

### 2026/09/05 - HSTU:生成式推荐 Scaling Law(Meta,ICML 2024)

#### 一、研究问题:判别式推荐模型不体现 Scaling Law

工业界 DLRM(深度学习推荐模型)用异质特征(数值型 + 类别型 embedding 表 + MLP),但实验发现 DLRM 不体现 Scaling Law--随计算规模扩大性能难提升。目标:让推荐也受益于 Scaling Law,铺路搜广推首个基础模型。

#### 二、研究目的

将推荐任务重定义为生成式框架下的 Seq2Seq 序列转导任务,设计具备有利缩放性质的 Transformer 类架构 HSTU,并完成工业级工程优化。

#### 三、研究内容

1. **生成式框架**:用户行为 / 物品特征 / 上下文统一序列化为按时间排序的 token(action/feature/context token),自回归训练同 NLP 语言建模。
2. **HSTU 架构**:1 去 softmax(pointwise 聚合注意,替代 softmax 归一化);2 Stochastic Length(训练随机采子序列,处理用户历史重尾分布);3 减少线性投影(6 层→2 层);4 M-FALCON(微批缓存 KV cache 摊还推理)。
3. 在合成 + 公开数据集 + 工业线上验证 Scaling Law。

#### 四、研究结果

HSTU 在合成与公开数据集 NDCG 最高超基线 65.8%;1.5 万亿参数线上 A/B ranking 提升 **12.4%**;模型质量随训练 FLOPs 呈幂律,跨三个数量级验证(与 GPT-3/LLaMA-2 行为一致)。

> **出处**:Zhai, Jiaqi; Liao, Lucy; Liu, Xing; 等(Meta)
> **标题**:Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations
> **会议**:ICML 2024(CCF-A)
> **官方地址**:https://proceedings.mlr.press/v235/zhai24a.html
> **代码**:https://github.com/facebookresearch/generative-recommenders

---

### 2026/09/05 - ULTRA-HSTU:弯曲推荐 Scaling Law 曲线(Meta,arXiv 2026)

#### 一、研究问题:HSTU 的二次方自注意力在超长序列下仍昂贵

HSTU 沿用全序列自注意力,复杂度 O(L2),在工业超长序列(16k+)与高 QPS 下仍有算力瓶颈。

#### 二、研究目的

通过模型-系统端到端协同设计,用半局部注意力大幅降复杂度,同时保持并强化 Scaling Law。

#### 三、研究内容

1. **SLA(Semi-Local Attention)**:注意力掩码拆为局部窗口 K1 + 全局窗口 K2,复杂度 O((K1+K2)·L);
2. **输入序列优化**:合并 item 与 action 表示使序列长度减半、FLOPs 降 4×,候选掩码防泄漏;
3. 注意力截断、MoT(混合 transducer)、负载均衡随机长度采样、混合精度(BF16/FP8/INT4)。

#### 四、研究结果

比常规模型训练缩放快 >5×、推理快 21×;18 层、服务 16k 序列;Meta 生产全量部署,消费/互动指标 +4%~8%。结论:工业场景 self-attention 严格优于 cross-attention。

> **出处**:Ding, Qin; Li, Rui; 等(Meta Recommendation Systems)
> **标题**:Bending the Scaling Law Curve in Large-Scale Recommendation Systems
> **会议**:arXiv 预印本(2026-02,非 CCF-A)
> **DOI**:https://doi.org/10.48550/ARXIV.2602.16986

---

### 2026/09/05 - OneRec：端到端生成式推荐替代级联（快手，arXiv 2025）

#### 一、研究问题:级联架构的计算碎片化、目标冲突与技术脱节

传统「召回-粗排-精排-重排」多级级联:1 计算碎片化(精排 SIM GPU MFU 仅 4.6%/11.2%);2 优化目标冲突(数百个目标分散各阶段);3 与 LLM/VLM 有技术代差。

#### 二、研究目的

用单个端到端 Encoder-Decoder 生成式模型替代整条级联链路。

#### 三、研究内容

1. **Tokenizer**:融合多模态 + 用户行为,RQ-Kmeans 分层量化为 3 层语义 ID;
2. **Encoder**:压缩用户全生命周期行为序列(多尺度);**Decoder**:MoE 增强 + NTP 自回归,支持会话级生成;
3. **RL 对齐**:偏好+格式+业务奖励 + 个性化 P-Score,改进 ECPO 算法;DPO + 迭代偏好对齐。

#### 四、研究结果

快手/极速版双端全量上线,观看时长提升(口径 +0.54%/+1.6%),MFU 11%→28.8%,运行成本仅级联架构约 10.6%。

> **出处**:Zhou, Guorui; Deng, Jiaxin; Wang, Shiyao; 等(快手)
> **标题**:OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment
> **会议**:arXiv 预印本(2025,非 CCF-A)
> **DOI**:https://doi.org/10.48550/ARXIV.2502.18965
> **代码**:无(社区实现 Gitee one-rec)

---

### 2026/09/05 - RankMixer：硬件感知 token-based 排序（字节，CIKM 2025）

#### 一、研究问题:工业排序模型 Memory-bound 而非 Compute-bound

工业排序训练/服务受延迟与高 QPS 限制;大量手工特征交叉模块继承自 CPU 时代,GPU 上 MFU 仅个位数百分比(实测 4.5%)。

#### 二、研究目的

保留 Transformer 高并行,用硬件感知的无参 token mixing 替代二次方 self-attention,并扩展容量。

#### 三、研究内容

1. **Multi-head Token Mixing**:无参算子实现跨 token 特征交叉,避免注意力矩阵 Memory-bound;
2. **Per-token FFN(PFFN)**:不同特征子空间独立参数,解决 inter-feature-space domination;
3. 扩展至 10 亿参数 Sparse-MoE 变体。

#### 四、研究结果

MFU 从 4.5% 提升至 45%;参数扩 100 倍而推理延迟几乎不变;抖音全量上线,活跃天数 +0.2%~0.3%、App 时长 +0.5%~1.08%。

> **出处**:Zhu, Jie; Fan, Zhifang; Zhu, Xiaoxie; 等(字节跳动)
> **标题**:RankMixer: Scaling Up Ranking Models in Industrial Recommenders
> **会议**:CIKM 2025(CCF-B)
> **DOI**:https://doi.org/10.1145/3746252.3761507

---

### 2026/09/05 - OneTrans:统一序列与非序列特征建模(字节,WWW 2026)

#### 一、研究问题:级联范式阻断序列建模与特征交互的信息流

传统「encode-then-interaction」把序列建模(LONGER)与特征交互(Wukong/RankMixer)拆为独立模块,阻断双向信息流、妨碍统一优化与缩放、无法复用 LLM 优化。

#### 二、研究目的

提出 OneTrans 统一 Transformer 骨干,将序列建模与特征交互融合进单一架构。

#### 三、研究内容

1. **统一 tokenizer**:序列 + 非序列特征转为单一扁平 token 序列,插入 [SEP] 分隔不同行为;
2. **混合参数化**:S-token 共享 Q/K/V/FFN 参数、NS-token 专属参数;
3. **因果注意力 + 跨请求 KV 缓存 + 金字塔堆叠**。

#### 四、研究结果

随参数增长高效缩放、持续优于强基线;线上 A/B 每用户 GMV +5.68%。

> **出处**:Zhang, Zhaoqi; Pei, Haolei; Guo, Jun; 等(字节跳动 + 南洋理工大学)
> **标题**:OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender
> **会议**:WWW 2026(CCF-A)
> **DOI**:https://doi.org/10.1145/3774904.3792838

---

### 2026/09/05 - TokenMixer-Large:7B/15B token-based 排序扩展(字节,arXiv 2026)

#### 一、研究问题:RankMixer 的残差设计、非纯净模型与深度不足

RankMixer 存在次优残差设计(mixing 前后直接相加致语义错位)、非纯净模型(残留低 MFU 算子)、深度不足、ReLU-MoE 训练不省成本等问题。

#### 二、研究目的

把 TokenMixer 扩展到 7B online / 15B offline 参数,修复 RankMixer 四处缺陷。

#### 三、研究内容

1. **Mixing & Reverting**:混合-还原保证语义一致 + 层间残差 + 辅助损失;
2. **Sparse-Pertoken MoE**:「先扩大后稀疏」,Gate Value Scaling 保证梯度;
3. FP8 E4M3 量化推理、Token Parallel。

#### 四、研究结果

MFU 达 60%;AUC 随参数对数线性提升且斜率更陡;电商订单 +1.66%、人均 GMV +2.98%。

> **出处**:(字节跳动)
> **会议**:arXiv 预印本(2026,非 CCF-A)
> **DOI**:https://doi.org/10.48550/ARXIV.2602.06563

---

### 2026/09/05 - R²ec：大推荐模型内生推理（NeurIPS 2025）

#### 一、研究问题:LLM 作推荐器忽视推理价值,作外部推理模块开销大

LLM 推荐两大范式(编码器嵌入 / 自回归生成 ID)忽视推理价值;现有 reasoning-augmented 做法把 LLM 当外部推理模块,资源开销大 + 联合优化差。

#### 二、研究目的

提出 R²ec,单个 decoder-only LLM 主干内生推理 + 推荐,用无标注 RL 联合优化。

#### 三、研究内容

1. **双头架构**:lm_head(自回归出推理 token)+ rec_head(物品嵌入内积评分),共享语义隐空间;
2. **RecPO 训练框架**:融合奖励 R = β·R_similarity + R_ranking(NDCG),GRPO/RLOO + PPO-clip + KL 正则。

#### 四、研究结果

三数据集 Hit@5 相对提升 68.67%、NDCG@20 提升 45.21%,效率接近传统 LLM-based 推荐器。

> **出处**:You, Runyang; Li, Yongqi; Lin, Xinyu; Zhang, Xin; Wang, Wenjie; Li, Wenjie; Nie, Liqiang
> **标题**:R²ec: Towards Large Recommender Models with Reasoning
> **会议**:NeurIPS 2025(CCF-A)
> **DOI**:https://doi.org/10.48550/ARXIV.2505.16994
> **代码**:https://github.com/YRYangang/RRec

---

### 2026/09/05 - RecZero / RecOne:RL 自主推理增强推荐(NeurIPS 2025)

#### 一、研究问题:蒸馏式 LLM 增强推荐的三重缺陷

主流 LLM 增强评分预测用蒸馏(ChatGPT 当 teacher 生成推理 → SFT 模仿):teacher 缺推荐知识、推理数据收集贵且静态、SFT 只学表面模式。

#### 二、研究目的

受 DeepSeek-R1-Zero 启发,用纯 RL 让 LLM 在推荐任务中自主涌现推理。

#### 三、研究内容

1. **RecZero(纯 RL)**:LLM 生成多条推理轨迹,规则奖励算优势、GRPO 优化;无需 teacher 与标注。
2. **RecOne(SFT+RL 冷启动)**:DeepSeek-R1 当教师生成「推理+评分」,预测错时喂真实评分重写自洽解释,再 RL 优化。
3. **Think-before-Recommendation 提示模板** + 规则奖励(格式 + 准确率)。

#### 四、研究结果

四数据集显著超蒸馏基线(MAE/RMSE);消融证实 GRPO 关键、纯 SFT 不行。

> **出处**:Kong, Xiaoyu; Jiang, Junguang; Liu, Bin; Xu, Ziru; Zhu, Han; Xu, Jian; Zheng, Bo; Wu, Jiancan; Wang, Xiang
> **标题**:Think before Recommendation: Autonomous Reasoning-Enhanced Recommender(RecZero / RecOne)
> **会议**:NeurIPS 2025(CCF-A)
> **DOI**:https://doi.org/10.48550/ARXIV.2510.23077
> **代码**:https://github.com/AkaliKong/RecZero

---
