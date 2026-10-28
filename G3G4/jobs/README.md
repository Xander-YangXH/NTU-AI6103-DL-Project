# jobs/ — Slurm 作业、队列、数据下载、汇总与画图

本文件夹里的脚本都在 `G3G4/`（集群上是 `~/myproject/gweva-g34`）根目录下执行。

## Slurm 作业脚本（TC2，分区 MGPU-TC2，QoS normal）

| 文件 | 资源 | 作用 |
|---|---|---|
| `smoke.sbatch` | 1 GPU、4 CPU、16 GB、30 min | 一次性检查：`nvidia-smi`、torch / CUDA / 各库版本、`tests/test_tiny.py`、`gweva-data` 中模型存在且三个任务样本数正确 |
| `g34.sbatch` | 1 GPU、8 CPU、24 GB、5 h 50 min | 训练作业：用 `STAGE` 指定阶段，在计算节点上运行 `make_queue.py` 生成队列，再用 `run_queue.sh` 以 `NPAR` 个进程并行执行 |
| `summarize.sbatch` | 无 GPU、2 CPU、4 GB、20 min | 运行 `summarize.py`，可以和训练作业同时排队 |

三个脚本的共同处理：

- 激活 conda 时临时关闭 `set -u`。conda 的 activate.d 脚本会读取未定义变量，第一次 smoke（作业 48242）就是因此失败的。
- 设置 `HF_HUB_OFFLINE=1`，因为计算节点不能联网。
- 把 `GWEVA_DATA` 设为代码目录旁边的 `gweva-data`。

提交示例：

```bash
sbatch jobs/smoke.sbatch
sbatch --export=ALL,STAGE=tune,NPAR=2 jobs/g34.sbatch     # STAGE = pilot | tune | main | lowres
sbatch jobs/summarize.sbatch
```

作业输出写在 `logs/slurm_<作业名>_<作业号>.{out,err}`。

## 队列

| 文件 | 作用 |
|---|---|
| `make_queue.py` | 生成 `jobs/queues/<stage>.txt`，每行一条 `python src/train.py ... > logs/<run_id>.log` 命令 |
| `run_queue.sh` | 用 `xargs -P NPAR` 在当前 GPU 上并行执行队列，结束时报告“多少个运行已有结果”。已完成的运行由 `train.py` 跳过，超时后重新提交即可续跑 |

`make_queue.py` 中的常量：

- 学习率网格 `LR_GRID = [2e-4, 5e-4, 1e-3]`。
- 调参种子 `TUNE_SEED = 0`。
- 种子 `SEEDS`：SST-2 0–2，MRPC / RTE 0–4；低资源 0–4。
- 选择规则 `pick_best`：dev primary 最高者胜，平局取较小学习率。

各阶段队列内容：

| 阶段 | 内容 |
|---|---|
| pilot | lr 5e-4、seed 0：G3 三个任务 + G4 RTE，共 4 个运行 |
| tune | 2 组 × 3 任务 × 3 个 lr，共 18 个运行 |
| main | 读取 tune 结果选 lr，并写入 `results/chosen_lr.json`；共 26 个运行 |
| lowres | G3、SST-2，用 G3 SST-2 选定的 lr，5 个种子 |

`jobs/queues/` 是在集群上自动生成的，没有下载到本机（`.gitignore` 也排除了它）。需要时重新运行 `make_queue.py` 即可得到。

## 数据下载

| 文件 | 作用 |
|---|---|
| `prefetch.py` | **在能联网的本机上运行一次**：用 `snapshot_download` 原样下载 `FacebookAI/roberta-base`（不含分类头，所以每个种子都会新建随机分类头），用 `load_dataset("nyu-mll/glue", task)` 下载 SST-2 / MRPC / RTE，核对样本数后 `save_to_disk` 到 `$GWEVA_DATA` |

## 汇总与画图（集群或本机均可）

| 文件 | 输入 | 输出 |
|---|---|---|
| `summarize.py` | `results/*/*.json`、`cache/*.pt` | `results/g34_runs.csv`、`results/g34_ranks.csv`；打印汇总表，并以 UTF-8 同时写入 `results/g34_summary.txt` |
| `plot_ranks.py` | `results/g34_ranks.csv` 中 main 阶段的行 | `results/g34_rank_heatmap.{png,pdf}`、`results/g34_rank_heatmap_diff.{png,pdf}` |

`summarize.py` 打印的汇总表包括：

- tune 网格；
- main 的 mean ± sd 与参数量；
- G4 − G3 配对差的 95% CI；
- 按层类型的平均秩；
- G3 / G4 方向一致性核对（需要 `cache/`）。

`plot_ranks.py` 生成两张图：

- `g34_rank_heatmap`：2 × 3 张热力图（G3 / G4 × 三个任务），每张 12 层 × 6 种，颜色上限 16。
- `g34_rank_heatmap_diff`：G4 − G3 差值图，红色表示 G4 给的秩更多，蓝色表示更少。

本机运行：

```powershell
..\code\.venv-prep\Scripts\python.exe jobs\summarize.py
..\code\.venv-prep\Scripts\python.exe jobs\plot_ranks.py
```
