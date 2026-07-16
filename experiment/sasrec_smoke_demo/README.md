# SASRec Smoke Demo

本目录用于第三堂课的GPU环境与完整训练链路演示。数据是确定性生成的合成数据，只用于smoke test，不代表正式推荐效果或论文复现结果。

## 环境

- Python 3.10
- PyTorch 2.12.1
- GPU环境使用课程已验证的cu126安装命令
- 除PyTorch外不需要第三方Python包

本代码不会安装或捆绑PyTorch。GPU版PyTorch应先按第三课演示步骤安装。

## 文件

```text
sasrec_smoke_demo/
├── sasrec_demo.py
├── generate_smoke_data.py
├── README.md
├── data/
│   └── smoke_interactions.csv
└── tests/
    └── test_sasrec_demo.py
```

`sasrec_demo.py`在一个文件中包含数据读取、ID映射、留一验证/测试切分、逐位置负采样、SASRec模型、训练、全物品排序评估、Checkpoint、日志和显存监控。

## Smoke数据

CSV字段：

```text
user_id,item_id,timestamp
```

固定数据包含：

- 128个用户；
- 120个物品；
- 每个用户24条有序交互；
- 共3072条交互；
- 生成种子为`20260714`。

数据中的大部分交互遵循分组后的循环偏好模式，并包含少量确定性噪声。每个用户最后两条交互分别作为验证目标和测试目标。

重新生成并核对固定数据：

```bash
python generate_smoke_data.py --output /tmp/smoke_interactions_regenerated.csv
cmp data/smoke_interactions.csv /tmp/smoke_interactions_regenerated.csv
```

## 运行

CPU smoke运行：

```bash
python sasrec_demo.py --device cpu --epochs 1
```

GPU环境检查：

```bash
CUDA_VISIBLE_DEVICES=0 python sasrec_demo.py --device cuda --check-only
```

GPU课堂运行：

```bash
CUDA_VISIBLE_DEVICES=0 python sasrec_demo.py \
  --device cuda \
  --epochs 10 \
  --batch-size 32 \
  --max-len 20 \
  --hidden-size 64 \
  --num-heads 2 \
  --num-blocks 2 \
  --allocator-limit-mib 700 \
  --gpu-budget-mib 1024
```

运行前应通过`nvidia-smi`选择空闲物理GPU。设置`CUDA_VISIBLE_DEVICES`后，程序内部选中的可见设备编号仍为`cuda:0`。

## 显存约束

- `--allocator-limit-mib 700`限制PyTorch缓存分配器可使用的显存；
- `--gpu-budget-mib 1024`是`nvidia-smi`实际进程显存上限；
- 程序在模型初始化、每个epoch结束和最终评估后采样显存；
- 如果当前PID的实际显存超过1024 MiB，程序立即报错退出；
- 如果系统不能返回按PID统计的显存，`summary.json`中的实际进程显存字段为`null`，此时必须人工查看`nvidia-smi`。

## 输出

每次运行创建独立的`outputs/run_时间戳/`目录：

```text
config.json
train.log
metrics.jsonl
summary.json
best_model.pt
```

最佳模型只根据验证集NDCG和HR选择。测试集指标在最佳Checkpoint加载后计算。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖固定数据重建、留出切分、负样本、模型形状、因果掩码、有限损失、参数更新和一次CPU端到端运行。

## 开源参考

- Kang与McAuley的原作者实现：[kang205/SASRec](https://github.com/kang205/SASRec)
- PyTorch实现：[pmixer/SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch)，Apache-2.0 License

课程代码保留SASRec的核心建模和训练范式，但使用独立编写的教学实现，并针对合成小数据、单进程运行和低显存课堂演示进行了简化。
