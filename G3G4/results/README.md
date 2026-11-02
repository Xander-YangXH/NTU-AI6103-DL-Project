# results/ — 全部实验结果

这里包含集群上 53 个运行的原始结果 JSON，以及由它们汇总得到的表格和图。

## 子文件夹（每个运行一个 JSON）

| 文件夹 | 运行数 | 评估集 | 内容 |
|---|---|---|---|
| `pilot/` | 4 | dev | lr 5e-4、seed 0 的试跑：G3 SST-2 / MRPC / RTE、G4 RTE |
| `tune/` | 18 | dev | 2 组 × 3 任务 × 3 个 lr，seed 0，用于选择学习率 |
| `main/` | 26 | 官方 validation | 按选定 lr 的主实验：SST-2 3 个种子，MRPC / RTE 各 5 个种子，G3 / G4 都有 |
| `lowres/` | 5 | 官方 validation | G3，SST-2 1k 子集，20 epoch，种子 0–4 |

文件名为 `<stage>_<group>_<task>_lr<lr>_s<seed>.json`。JSON 字段见下文。

## 汇总文件（由 `jobs/summarize.py` 和 `jobs/plot_ranks.py` 生成）

| 文件 | 作用 |
|---|---|
| `chosen_lr.json` | main 阶段为每个 (组, 任务) 选定的学习率。G3：SST-2 1e-3，MRPC / RTE 2e-4；G4：SST-2 5e-4，MRPC / RTE 2e-4。由 `make_queue.py main` 写入 |
| `g34_runs.csv` | 每个运行一行。列：run_id、stage、group、task、lr、seed、eval_split、acc、f1、primary、step0_grad_B、lora_params、budget_dev、init_s、total_s、peak_mem_gb、slurm_job。用于配对统计和报告表格 |
| `g34_ranks.csv` | 每个运行 × 每个 LoRA 层一行。列：run_id、group、task、seed、layer、kind、rank。rank 为 0 的层不出现。`plot_ranks.py` 读取这个文件 |
| `g34_summary.txt` | 汇总表的文本版（UTF-8），见下方说明 |
| `g34_rank_heatmap.png` / `.pdf` | 2 × 3 热力图：行是 G3 / G4，列是 SST-2 / MRPC / RTE。每张图为 12 层 × 6 种（Q、K、V、Attn-out、FFN-in、FFN-out），格子里是 main 种子平均的秩，颜色上限 16。标题给出注意力 / FFN 的秩总数 |
| `g34_rank_heatmap_diff.png` / `.pdf` | 1 × 3 差值图，每格为 G4 − G3。红色表示 G4 更多，蓝色表示更少，色标对称。标题给出注意力 / FFN 的净变化（+202 到 +246 / −65 到 −78） |

`g34_summary.txt` 包含以下内容：

- tune 网格；
- main 的 mean ± sd 与参数量；
- G4 − G3 配对差的 95% CI；
- 按层类型的平均秩；
- G3 / G4 方向一致性核对（子空间重合度、L1 距离）。

PNG 用于查看和插入报告，PDF 是矢量版。

## 单个结果 JSON 的字段

| 字段 | 含义 |
|---|---|
| `run_id`、`stage`、`group`、`task`、`lr`、`seed` | 运行标识 |
| `eval_split` | `dev`（pilot / tune）或 `val`（main / lowres，即官方 validation） |
| `final` | 最后一个 epoch 的指标：`acc`、`primary`，MRPC 另有 `f1`。报告用的就是 `final.primary` |
| `per_epoch` | 每个 epoch 的评估结果列表 |
| `step0_grad_B` | 第 0 步、尚未更新时 LoRA B 的梯度范数（初始化信号强度） |
| `loss_first_steps` | 前 500 步的训练 loss |
| `steps` | 总训练步数 |
| `lora_params`、`head_params` | LoRA 参数量、分类头参数量 |
| `budget_dev` | `lora_params` 相对预算 1,327,104 的偏差 |
| `ranks` | 每个 LoRA 层的实际秩 |
| `time_s` | `load`、`head_warmup`、`init`、`train_eval`、`total`（秒） |
| `peak_mem_gb` | GPU 峰值显存 |
| `cfg` | 本次运行的完整 `CFG` |
| `host`、`slurm_job`、`gpu`、`versions` | 运行环境：节点、作业号、GPU 型号，以及 torch / transformers / peft 版本 |
| `g3_rule_on_cache` | 仅 G4 有：在同一缓存上按 G3 规则重算的分配，用于核对 |

## 重新生成汇总

在 `G3G4/` 中运行以下命令（需要 `cache/` 才能输出一致性核对部分）：

```powershell
..\code\.venv-prep\Scripts\python.exe jobs\summarize.py
..\code\.venv-prep\Scripts\python.exe jobs\plot_ranks.py
```
