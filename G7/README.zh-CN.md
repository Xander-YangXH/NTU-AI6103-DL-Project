<p align="right"><a href="README.md">English</a> | <b>中文</b></p>

# G7：按小组统一协议运行 LoRA-GA

负责人：Liyouran，分支 `liyouran`。本目录包含 G7 的代码、Slurm 执行脚本与结果。

实验遵循 main 的公共约定，数据与训练管线源自共享提交 `26616379899f0abce7d693e227ad4e03138312ab`。[实验协议](docs/experiment-protocol.zh-CN.md) 记录 G7 的方法、数据与执行设置。

## 方法

先预热分类头一个 epoch，再用训练集的 1,024 条带标签样本，估计 72 个注意力与 FFN 权重矩阵的均值损失梯度。校准采用 FP32、eval 模式并冻结预热后的分类头；各样本权重相同。

对梯度 `G = U S Vᵀ`，作者版 `ArB2r` 初始化为 `A = c V[:, :r]ᵀ`、`B = c U[:, r:2r]`，其中 `c = d_out**0.25 / sqrt(16)`。两侧均使用奇异向量，不把奇异值乘入 B。随后执行 `W = W₀ − (α/r)B₀A₀`，保持有效初始权重等于 W₀。实现记录补偿误差，并把初始 A/B 与最后的 adapter 保存在本地缓存，供模型重建。

初始化依据 [LoRA-GA 作者源码](https://github.com/Outsider565/LoRA-GA/tree/c4cd5372c75b290924214b348008891f744512ef) 的 `peft/src/peft/tuners/lora/layer.py::lora_ga_init`。A、B 从第一次更新开始共同训练。

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

三个完整数据任务共执行 9 次调参、13 次独立正式训练。

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

## 结果与分析

| 任务 | 种子数 | 准确率 (%) | F1 (%) |
|---|---:|---:|---:|
| SST-2 | 3 | 93.96 ± 0.46 | — |
| MRPC | 5 | 88.53 ± 0.74 | 91.83 ± 0.46 |
| RTE | 5 | 56.68 ± 8.85 | — |

三个任务均选择学习率 2e-4。提高到 1e-3 后，seed 0 的 dev 主指标分别下降 2.44、8.63、18.07 个百分点。RTE 的正式准确率范围为 47.29–66.43%；seed 1 在 277 条官方验证样本上全部预测 label 1。已记录 loss 均为有限值，反映分类退化现象。

[完整 G7 分析](results/summary.md) 仅从本组 JSON 生成，包含任务解读、均值置信区间、precision/recall、学习率搜索、逐种子预测、初始化误差、开销和前 500 步曲线。[中文结果解读](docs/analysis.zh-CN.md) 提供完整叙述。

G7 使用 Python 3.9、torch 2.4.0、Transformers 4.52.4、PEFT 0.17.1 和 L40S；每次 run 的 JSON 保存实际版本和 GPU。Slurm 作业 48482 用两进程共享一张卡，在 1,179 秒内完成 22 次训练。

## 目录

- `src/data.py`：共享的数据划分、分词与 loader。
- `src/train.py`：共享训练流程、G7 初始化和官方集最终评估。
- `src/init_g7.py`：带标签均值梯度 SVD 与权重补偿。
- `jobs/`：数据准备、Slurm 执行与汇总。
- `results/`：run JSON、CSV、汇总和图。
- `docs/experiment-protocol.zh-CN.md`：方法、数据与执行协议。
- `docs/analysis.zh-CN.md`：G7 实验结果与解读。
