<p align="right"><a href="RUN.md">English</a> | <b>中文</b></p>

# 代码运行说明

本文件说明环境、数据、检查环境的冒烟测试，以及各组的运行命令。项目概览和结果见 [`README.zh-CN.md`](README.zh-CN.md)。

## 1. 包里有什么

| 文件夹 | 内容 | 入口 |
|---|---|---|
| `G1G2/` | G1 LoRA、G2 幅值对照；`jobs/` 里有各项对照和 H4 分析 | `src/train.py` |
| `G3G4/` | G3 EVA、G4 按参数成本分配；共享的数据划分和训练循环 | `src/train.py` |
| `G5G6/` | G5 / G6 GW-EVA（Python 包 `gw_eva`） | `python -m gw_eva.study` |
| `G7/` | G7 LoRA-GA | `src/train.py` |
| `G8/` | G8：用 wikitext-103 校准的 EVA | `src/train.py` |
| `G9/` | G9：用别的任务校准的 EVA | `src/train.py` |
| `H1 Diagnosis/` | H1 子空间诊断（GW-EVA 对比 EVA） | `scripts/run_h1.py` |
| `summary/` | 跨组表格、图、报告初稿、结果页面 | `collect.py`、`figures.py`、`site/build_data.py` |
| `tools/` | 覆盖全部组的冒烟测试（`smoke_test.py`、`smoke_test.sbatch`） | `smoke_test.py` |

各组的 `results/` 文件夹里已经有报告用到的全部运行 JSON，所以不需要 GPU 就能重新生成表格、图和结果页面（第 6 节）。

## 2. 硬件

- **训练需要支持 bf16 的 NVIDIA GPU**（Ampere 及以后）。报告中的每次运行都只用一张卡，显存不超过 24 GB；共享实验分别在 L40S、A40、RTX 4090 D 和 RTX 3050 Ti 笔记本显卡上运行。
- G5/G6、G7 和 H1 直接调用 CUDA，只能在 GPU 上运行。G1–G4、G8、G9 也能在 CPU 上运行，但在 CPU 上只适合跑冒烟测试。
- 单张 L40S 上每次运行耗时：MRPC / RTE 1–4 分钟，SST-2 3–8 分钟。一张卡可以同时跑两个。
- 汇总脚本和结果页面不需要 GPU。

## 3. 环境

所有组都可以在同一个环境里运行。这就是报告中大多数组使用的环境（torch 2.11.0、transformers 5.21.0、peft 0.21.2、datasets 5.1.0）。

```bash
python3.11 -m venv .venv            # Python 3.11 或 3.12；也可以用 conda
source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # 按显卡驱动选择对应 CUDA 版本的 wheel
pip install -r requirements.txt
```

只有 CPU 的机器或 macOS 上，直接从 PyPI 安装 `torch==2.11.0`。[`requirements.txt`](requirements.txt) 列出了代码用到的全部第三方包。G9 的 notebook `src/overlap_analysis.ipynb` 另外需要 `pandas` 和 Jupyter。

**2026-11-22 用本包的冒烟测试（第 5 节）验证过：**

| 机器 | Python / torch / transformers / peft / datasets | 结果 |
|---|---|---|
| NTU CCDS TC2，NVIDIA L40S（Slurm） | 3.11.17 / 2.11.0+cu128 / 5.21.0 / 0.21.2 / 5.1.0 | 冒烟测试：G1–G4、G5/G6、G7、G8、G9 和 H1 **全部通过**（H1 约 20 分钟，其余每组 9–70 秒） |
| 同一节点，两次完整运行 | 同上 | G1、RTE、种子 0、lr 5e-4：**与已发表结果逐位一致**（74.01，第 0 步梯度和第一步 loss 完全相同）。G7、RTE、种子 0：能完整跑完，得到 70.76（已发表为 66.43，原因见下） |
| macOS，仅 CPU | 3.12.13 / 2.11.0 / 5.21.0 / 0.21.2 / 5.1.0 | 冒烟测试：G1–G4、G8、G9 **通过**；G5/G6 在替换 CUDA 调用后通过；G7 和 H1 跳过（只能用 GPU）。各组自带的离线测试（`G3G4/tests/test_tiny.py`、`G5G6/tests/`）也都通过 |

也就是说，这一个环境能运行所有组。发表时用的就是这个环境的组（G1–G4、G8 和所有对照），在同型号显卡上能逐位复现。G7、G5/G6 和 H1 发表时用的是别的库版本（见下表），在这个环境里得到的数字相近但不完全相同：G7 见上；H1 在 RTE 种子 0 上得到 EVA–GW 0.421、GW–GW 0.783，已发表的是 0.479 和 0.853，结论相同。如果要与这几组的发表数字完全一致，请使用它们记录的版本。

**发表结果所用的版本。** 换库版本或换显卡，结果可能有细微差别（见 README 的“已知局限”）。如果要逐位复现某一组，请使用该组运行 JSON 里记录的版本（`versions`、`gpu` 字段）：

| 组 | Python | torch | transformers | peft | datasets | 记录位置 |
|---|---|---|---|---|---|---|
| G1–G4、G8、所有对照 | 3.11 / 3.12 | 2.11.0+cu128 | 5.21.0 | 0.21.2 | 5.1.0 | `G3G4/requirements.txt`、`G3G4/env_gweva.yml`、`G3G4/pip_freeze.txt` |
| G9 | – | 2.11.0+cu128 | 5.20.2 | 0.21.2 | – | 运行 JSON |
| G5 / G6、H1 | ≥ 3.11 | 2.8.0 | 4.45.1 | 0.12.0 | 2.20.0 | `G5G6/requirements.txt` |
| G7 | 3.9 | 2.4.0 | 4.52.4 | 0.17.1 | 2.20.0 | `G7/requirements.txt` |

## 4. 数据

多数集群的计算节点不能上网，所以要在能上网的机器（登录节点或自己的电脑）上下载一次，必要时再把文件夹拷过去。之后训练作业设置 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` 离线运行。

**4.1 共享数据文件夹（G1–G4、G7、G8、G9）。** 包括 RoBERTa-base、GLUE SST-2 / MRPC / RTE，以及 G8 用的 wikitext-103，约 1 GB：

```bash
export GWEVA_DATA=$HOME/gweva-data     # 任意文件夹；各组都读这个变量（默认 ~/gweva-data）
python G8/jobs/prefetch.py
```

这会生成 `$GWEVA_DATA/roberta-base/`、`$GWEVA_DATA/glue/{sst2,mrpc,rte}/` 和 `$GWEVA_DATA/wikitext103/`。之后每个训练用的终端和作业里都要设置同样的 `GWEVA_DATA`。

**4.2 G5/G6 和 H1。** 它们会把同一模型和 GLUE parquet 文件的固定版本下载到 `G5G6/models/` 和 `G5G6/data/`，并用 SHA-256 校验（约 500 MB）：

```bash
cd G5G6
PYTHONPATH=src python jobs/prefetch.py --tasks rte mrpc sst2
cd ..
```

把 `GWEVA_ROOT` 指向 `G5G6/` 后，H1 会复用这些文件（第 7.6 节）。

## 5. 检查环境：冒烟测试

冒烟测试用几十条样本、只跑 1 个 epoch，执行每组真正的入口代码，然后检查它写出的结果 JSON：最终指标、第 0 步 B 的梯度、loss 曲线、LoRA 参数量。整个测试只需几分钟，用来证明代码、环境和数据能配合运行；测出来的数字没有意义。

```bash
python tools/smoke_test.py                    # G1–G4、G8、G9、G7、G5G6；在 CPU 上会跳过 G7，G5G6 的 CUDA 调用会被替换掉
python tools/smoke_test.py --groups G3 G8     # 只测部分组
python tools/smoke_test.py --groups H1        # H1，需要 GPU，约 10–30 分钟
```

在 Slurm 集群上，从仓库根目录提交。分区、QoS 和节点参数因集群而异，下面是 NTU CCDS TC2 的写法：

```bash
sbatch -p MGPU-TC2 --qos=normal --nodelist=TC2N08 --export=ALL,VENV=$PWD/.venv,GWEVA_DATA=$HOME/gweva-data tools/smoke_test.sbatch
```

每组会输出 `[PASSED]`；或者输出 `[SKIPPED]` 并给出原因；或者输出 `[FAILED]` 并附上报错的最后几行。

## 6. 重新生成表格、图和结果页面（不需要 GPU）

```bash
python summary/collect.py           # summary/tables.md、summary/all_runs.csv
python summary/figures.py           # summary/figures/*.png、*.pdf
python summary/site/build_data.py   # summary/site/data.js
python -m http.server 8000 --directory summary/site   # 然后打开 http://localhost:8000
```

这些脚本只读取 `*/results/` 和 `H1 Diagnosis/results/` 下的 JSON。换 matplotlib 版本后，图的渲染细节可能不同，但数字不变。

## 7. 运行实验

**重跑之前注意：** 如果某次运行的结果 JSON 已经存在，`train.py` 会跳过它；而各组的 `results/` 里已经有发表用的结果。单次运行请用 `--out` 指定一个新文件夹；完整重跑请先把原文件夹改名，例如 `mv results results_published`。`main` 和 `lowres` 的队列会从 `results/tune/` 读取学习率，所以要保留调参结果，或者先重跑调参阶段。

在 dev 划分上选出的学习率（低资源沿用 SST-2 的选择；G8、G9 沿用 G3 的）：

| 组 | SST-2 | MRPC | RTE |
|---|---|---|---|
| G1 | 5e-4 | 2e-4 | 5e-4 |
| G2 | 5e-4 | 1e-3 | 2e-4 |
| G3、G8、G9 | 1e-3 | 2e-4 | 2e-4 |
| G4 | 5e-4 | 2e-4 | 2e-4 |
| G5 | 1e-3 | 2e-4 | 2e-4 |
| G6 | 5e-4 | 2e-4 | 2e-4 |
| G7 | 2e-4 | 2e-4 | 2e-4 |

种子：SST-2 用 0–2，MRPC、RTE 用 0–4，低资源用 0–4（G5/G6 为 0–2）。各阶段：
- `tune`：种子 0，学习率网格 2e-4 / 5e-4 / 1e-3，在 10% 的 dev 划分上评估；
- `main`：在官方 validation 上评估；
- `lowres`：1,000 条的 SST-2 子集。

### 7.1 G1 / G2（`G1G2/`）和 G3 / G4（`G3G4/`）

```bash
cd G1G2                                        # G3G4 同理，用 --group G3 / G4
python src/train.py --stage main --task rte --group G1 --lr 5e-4 --seed 0 --out results_new
```

跑完整的一个阶段，一张卡同时跑两个（Linux）：

```bash
python jobs/make_queue.py tune                 # 生成 jobs/queues/tune.txt；之后依次是 main、lowres
bash jobs/run_queue.sh jobs/queues/tune.txt 2
python jobs/summarize.py --g3 ../G3G4/results  # G1G2：汇总表，并按种子与 G3 配对
```

用 Slurm 时：`sbatch --export=ALL,STAGE=tune,NPAR=2 jobs/g12.sbatch`（G3G4 用 `jobs/g34.sbatch`）。模板里写的是 TC2 的分区和路径，换集群时需要修改。

报告中的低资源对照、噪声基线和逐层 H4 分析，来自 `G1G2/jobs/make_queue.py` 的几个额外阶段（`lowres_ctl`、`seeds10`、`g3_lowres_ctl`、`quick4`、`quick_ad`，见脚本开头的说明），以及 `jobs/g3_control.py`、`jobs/eva_noise_floor.py`、`jobs/h4_layers.py` 三个脚本。它们会导入 `../G3G4/src` 和 `../G8/src`，本包的目录结构已经满足这一点。

### 7.2 G5 / G6（`G5G6/`）

```bash
cd G5G6
export PYTHONPATH=$PWD/src
python -m gw_eva.study --tasks rte             # 先用种子 0 调参，再对 G5、G6 跑全部种子
python -m gw_eva.study                         # 全部任务：rte mrpc sst2 sst2_lowres（sst2_lowres 要在 sst2 之后跑）
python jobs/summarize.py                       # 根据 results/main 生成结果表
```

输出写到 `G5G6/outputs/gw_eva/shared_study/`，其中也保存了 H1 需要的每个种子的分类头检查点。`jobs/parallel_worker.py` 可以在同一个 study 文件夹里并行跑两个任务队列，与原始重跑的做法相同。

### 7.3 G7（`G7/`）

```bash
cd G7
python src/train.py --stage main --task rte --lr 2e-4 --seed 0 --out results_new
python jobs/run_experiments.py                 # 对所有任务先调参再跑主实验，一张卡上两个进程
python jobs/summarize.py
```

`jobs/run_g7.sbatch` 是对应的 Slurm 版本。

### 7.4 G8（`G8/`）

```bash
cd G8
python src/train.py --stage main --task rte --lr 2e-4 --seed 0 --out results_new
python jobs/make_queue.py main && bash jobs/run_queue.sh jobs/queues/main.txt 2
```

G8 没有调参阶段，直接用 G3 的学习率。

### 7.5 G9（`G9/`）

```bash
cd G9
python src/train.py --stage main --task rte --lr 2e-4 --seed 0 --out results_new
```

13 次主实验列在 `run_main.bat`（Windows）中。Linux / macOS 上可以这样跑：

```bash
for s in 0 1 2; do python src/train.py --stage main --task sst2 --lr 1e-3 --seed $s; done
for t in mrpc rte; do for s in 0 1 2 3 4; do python src/train.py --stage main --task $t --lr 2e-4 --seed $s; done; done
```

重合度分析在 `src/overlap_analysis.ipynb` 里；`G1G2/jobs/h4_layers.py` 可以直接从仓库按层重建同样的重合度。

### 7.6 H1（`H1 Diagnosis/`）

H1 要用 G5/G6 study 保存的分类头检查点，所以要先对该任务运行 G5/G6（第 7.2 节），至少完成初始化这一步。

```bash
cd "H1 Diagnosis"
export GWEVA_ROOT=$PWD/../G5G6                 # 复用 G5G6 的模型和数据
export PYTHONPATH=$PWD/src:$PWD/scripts
python scripts/run_h1.py --task rte --seed 0 --checkpoint-root ../G5G6/outputs/gw_eva/shared_study --output results_new
python scripts/summarize_h1.py results_new    # 需要 4 个任务 × 3 个种子都跑完
```

## 8. 常见问题

| 现象 | 解决办法 |
|---|---|
| 输出 `skip <run>: done`，没有开始训练 | 该结果 JSON 已存在；用 `--out` 指定新文件夹（第 7 节） |
| 报错 `FileNotFoundError`，路径在 `gweva-data` 下 | `GWEVA_DATA` 没设置或指向了别处；重新执行第 4.1 节，或导出正确的路径 |
| 计算节点上出现网络错误 | 在能上网的机器上下载（第 4 节），并设置 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` |
| 提示 `G5G6/models/roberta-base is missing` | 执行第 4.2 节 |
| G7 / G5G6 / H1 在 CPU 机器上停止 | 它们的设计就要求 CUDA |
| 结果与报告略有差别 | 不同显卡或库版本会改变 bf16 数值；所用版本见第 3 节 |
| `run_queue.sh` 报 `xargs: invalid option -- 'd'` | 这是 macOS 的 xargs；请在 Linux 上运行，或逐条执行 `jobs/queues/*.txt` 里的命令 |

## 9. 打包时做的改动

实验代码与仓库一致。为了让各文件夹能在同一个环境里配合运行，只做了以下改动：

- 新增 `requirements.txt`、`RUN.md` / `RUN.zh-CN.md` 和 `tools/`（冒烟测试）。
- `G7/src/train.py` 和 `G7/jobs/run_experiments.py` 不再要求必须在 Slurm 作业中运行，但仍要求 CUDA GPU。在 Slurm 之外运行时，JSON 里的 `slurm_job` 为 `null`。训练过程本身没有改动。
