<p align="right"><a href="README.md">English</a> | <b>中文</b></p>

# G7：按小组统一协议运行 LoRA-GA

负责人：Liyouran，分支 `liyouran`。本目录包含 G7 的代码、Slurm 执行脚本与结果。

实验遵循 main 的公共约定，并复用 `yinchenyu_g3g4:G3G4/src/` 的数据划分和训练流程。[参数核对表](docs/protocol-audit.zh-CN.md) 记录了采用的设置及来源。

## 方法

先预热分类头一个 epoch，再用训练集的 1,024 条带标签样本，估计 72 个注意力与 FFN 权重矩阵的均值损失梯度。校准采用 FP32、eval 模式并冻结预热后的分类头；各样本权重相同。

对梯度 `G = U S Vᵀ`，作者版 `ArB2r` 初始化为 `A = c V[:, :r]ᵀ`、`B = c U[:, r:2r]`，其中 `c = d_out**0.25 / sqrt(16)`。两侧均使用奇异向量，不把奇异值乘入 B。随后执行 `W = W₀ − (α/r)B₀A₀`，保持有效初始权重等于 W₀。实现记录补偿误差，并把初始 A/B 与最后的 adapter 保存在本地缓存，供模型重建。

初始化依据 [LoRA-GA 作者源码](https://github.com/Outsider565/LoRA-GA/tree/c4cd5372c75b290924214b348008891f744512ef) 的 `peft/src/peft/tuners/lora/layer.py::lora_ga_init`。G7 的 B 非零；G5/G6 的 GW-EVA 则只初始化 A、保持 B 为零。

## 统一参数

| 设置 | G7 |
|---|---|
| 模型与任务 | RoBERTa-base；SST-2、MRPC、RTE |
| LoRA | 全部 72 个注意力与 FFN 线性层；r=8、α=8、dropout=0 |
| 可训练参数 | LoRA 1,327,104，分类头 592,130 |
| 学习率网格 | 2e-4、5e-4、1e-3；调参用 seed 0 |
| 正式种子 | SST-2：0–2；MRPC/RTE：0–4 |
| 训练轮数 | SST-2：3；MRPC/RTE：10 |
| batch / 最大长度 | 32 / 128；动态补齐 |
| 优化器 | AdamW；微调 weight decay 0；线性调度与 6% warm-up；不裁剪梯度 |
| 精度 | FP32 权重，BF16 混合精度训练 |
| 分类头预热 | 1 epoch、LR 1e-3、冻结编码器；与参考代码相同，整个模型置为 train |
| 数据划分 | NumPy seed 2026 的随机排列；10% dev，其余 90% 训练 |
| 校准 | 训练集 1,024 条、batch 16、NumPy 抽样种子等于 run seed；G7 保留 labels |
| 选参 | 最后一轮 dev 指标；MRPC F1，其余准确率；并列取较小 LR |
| 正式评估 | 最后一轮模型，在官方 validation 上评估 |

分类头预热保留参考实现 AdamW 默认的 weight decay 0.01，正式微调则显式设为 0。预热和正式训练连续复用同一个带种子的 train loader。逐 epoch 诊断使用 dev，每次 main run 仅在训练结束后评估一次官方 validation；结果 JSON 的 `per_epoch_eval_split` 标明诊断集。

总共 9 次调参、13 次独立正式训练。提案的低资源 SST-2 属于 G1/G3/G5，本轮 G7 不包含该任务。

## 运行

使用项目已有虚拟环境和本目录依赖列表。`GWEVA_DATA` 指向包含 `roberta-base/` 和 Hugging Face datasets 格式的 `glue/{sst2,mrpc,rte}/` 的本地目录。

```bash
cd G7
export GWEVA_PYTHON=/path/to/project/.venv/bin/python
export GWEVA_DATA=/path/to/gweva-data
"$GWEVA_PYTHON" jobs/prepare_data.py
mkdir -p logs
sbatch jobs/run_g7.sbatch
"$GWEVA_PYTHON" jobs/summarize.py
```

默认准备脚本复用当前工作区已下载的 parquet 与模型；在新机器上可给准备命令增加 `--download`，下载固定版本的公开数据和模型。作业先完成调参、选择各任务学习率，再运行 main 阶段；同一作业的两个进程共享它申请的一张 GPU。

每次 run 的公共格式 JSON 写入 `results/tune/` 或 `results/main/`；再次提交会跳过已完成 run。缓存、权重、原始数据和日志不进入 Git。

## 结果与比较

[结果汇总](results/summary.md) 包含完成数量、学习率选择、均值与样本标准差、耗时、loss 曲线，以及同种子的 G7−G3/G4 配对差值和 95% t 区间。[参照指标快照](results/reference_metrics.json) 保存其他成员的必要标量和来源提交，下载本分支后即可重新统计。

G7 使用 Python 3.9、torch 2.4.0、Transformers 4.52.4、PEFT 0.17.1 和 L40S；已有 G3/G4 结果来自 A40、torch 2.11.0、Transformers 5.21.0、PEFT 0.21.2。每次 run 的 JSON 保存实际版本和 GPU。配对比较统一了种子与训练设置，同时保留这些环境差异；G3 的 rank 重分配还会改变实际 LoRA 参数量。

## 目录

- `src/data.py`：共享的数据划分、分词与 loader。
- `src/train.py`：共享训练流程、G7 初始化和官方集最终评估。
- `src/init_g7.py`：带标签均值梯度 SVD 与权重补偿。
- `jobs/`：数据准备、Slurm 执行与汇总。
- `results/`：run JSON、CSV、汇总和图。
- `docs/protocol-audit.zh-CN.md`：旧协议与小组协议的逐项核对。
