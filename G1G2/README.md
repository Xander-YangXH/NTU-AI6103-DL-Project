# G1 / G2：LoRA 基线与幅值对照（RoBERTa-base × GLUE）

AI6103 小组项目“改进 EVA”中我负责的两组实验。

- **G1 LoRA**：A 为 Kaiming 均匀随机初始化（PEFT 默认），B = 0，均匀 r = 8。
- **G2 幅值对照**：A 为随机正交行，行范数与 EVA 相同；B = 0，均匀 r = 8。用来回答“谱初始化的收益来自方向还是幅值”。

## 1. 与 G3/G4 的关系

为了让 G1/G2 能和 G3/G4 按种子配对比较，本文件夹**直接复用**组员的管线（分支 `yinchenyu_g3g4`，提交 `2661637`，文件夹 `G3G4/`）。

| 文件 | 来源 |
|---|---|
| `src/data.py`、`src/alloc.py`、`jobs/prefetch.py`、`jobs/run_queue.sh`、`requirements.txt`、`.gitignore`、`.gitattributes` | 原样复制，未改动 |
| `src/train.py` | 复制后只改了初始化一步：`--group` 改为 G1/G2，不再构建 EVA 校准数据；其余代码（预热、训练循环、评估、JSON 字段）不变 |
| `jobs/make_queue.py` | 复制后改了组名、pilot 内容，以及低资源只跑 G1 |
| `src/init_g12.py`、`jobs/g12.sbatch`、`jobs/summarize.py` | 新写 |

因此以下设置与 G3/G4 完全相同：dev 划分（`SPLIT_SEED=2026`，numpy 置换）、低资源 1k 子集（种子 2027）、分类头预热、AdamW + 线性调度、bf16、学习率网格与选择规则、种子、结果 JSON 格式。

软件版本也与 G3/G4 一致：torch 2.11.0+cu128、transformers 5.21.0、peft 0.21.2、datasets 5.1.0。

## 2. 初始化细节

| | G1 | G2 |
|---|---|---|
| A 的方向 | Kaiming 均匀 U(±1/√d_in) | 随机高斯矩阵 QR 分解得到的 r 个正交行 |
| A 的行范数 | ≈ 1/√3 ≈ 0.577 | 1.0 |
| B | 0 | 0 |
| rank / α / scaling | 8 / 8 / 1 | 8 / 8 / 1 |
| LoRA 参数量 | 1,327,104 | 1,327,104 |

**为什么 G2 的行范数取 1。** PEFT 0.21.2 的 EVA（`whiten=False`，默认）把增量 PCA 的前 r 个主成分直接写入 `lora_A`，这些主成分是单位正交向量，所以 EVA 的每一行范数都是 1。G2 保持这一幅值，只把方向换成随机的，从而把“方向”和“幅值”分开。

**配对。** G2 先按 G1 的方式构建 LoRA（消耗相同的全局随机数），再用独立的 `torch.Generator(seed)` 覆盖 A。同一种子下，G1 和 G2 的分类头、预热、数据顺序和 dropout 都相同，唯一的差别就是 A。

## 3. 运行方式（CCDS TC2）

集群目录结构：

```
~/gweva/
├── G1G2/        本文件夹
├── gweva-data/  jobs/prefetch.py 下载的 RoBERTa-base 与 GLUE（488 MB）
└── venv/        基于 conda 环境 ecomq（torch 2.11）的 venv，装 transformers / peft / datasets 的上述版本
```

只使用 TC2N07 / TC2N08 两个节点（L40S）：`g12.sbatch` 排除了 N01–N06；如果分到的不是 L40S，脚本直接退出。

```bash
cd ~/gweva/G1G2
sbatch --export=ALL,STAGE=pilot,NPAR=2  jobs/g12.sbatch
sbatch --export=ALL,STAGE=tune,NPAR=2   jobs/g12.sbatch
sbatch --export=ALL,STAGE=main,NPAR=2   jobs/g12.sbatch     # 读 tune 结果选学习率，写 results/chosen_lr.json
sbatch --export=ALL,STAGE=lowres,NPAR=2 jobs/g12.sbatch
sbatch --export=ALL,STAGE=lowres_ctl,NPAR=2 jobs/g12.sbatch # 学习率对照
sbatch --export=ALL,STAGE=seeds10,NPAR=2 jobs/g12.sbatch    # G1 种子 5–9
sbatch --export=ALL,STAGE=g3_lowres_ctl,NPAR=2 jobs/g12.sbatch  # G3 低资源对照，需要同级的 ../G3G4/src
python jobs/summarize.py                                    # results/g12_summary.txt、results/g12_runs.csv
```

| 阶段 | 内容 | 运行数 |
|---|---|---|
| pilot | G1、G2 × MRPC，lr 5e-4，seed 0 | 2 |
| tune | 2 组 × 3 任务 × 3 个 lr，seed 0，dev 评估 | 18 |
| main | SST-2 3 个种子，MRPC / RTE 各 5 个，官方 validation | 26 |
| lowres | G1，SST-2 1k，5 个种子，lr 沿用 G1 在 SST-2 上的选择 | 5 |
| lowres_ctl | 对照（不属于共享协议）：同上，但 lr 用 G3 的 1e-3，写入 `results/control/` | 5 |
| seeds10 | G1 的 MRPC、RTE、低资源各加种子 5–9，沿用已选学习率，写入 `results/seeds10/` | 15 |
| g3_lowres_ctl | G3 低资源：L40S 复现（lr 1e-3），lr 5e-4、2e-4，以及均匀秩 ρ = 1；用 G3G4 未改动的代码，写入 `results/control*/` | 20 |

## 4. 结果摘要

全部 51 个运行已完成（pilot 2、tune 18、main 26、lowres 5），另有 5 个低资源学习率对照。官方 validation，mean ± sd，%：

| 任务 | G1 | G2 | G3（参照） |
|---|---|---|---|
| SST-2 (acc) | 93.92 ± 0.00 | 94.07 ± 0.24 | 94.19 ± 0.43 |
| MRPC (F1) | 91.34 ± 0.43 | 91.16 ± 1.11 | 92.02 ± 0.79 |
| RTE (acc) | 73.94 ± 2.10 | 73.86 ± 1.04 | 74.80 ± 2.49 |
| 低资源 SST-2 1k | 91.44 ± 0.28 | – | 88.81 ± 0.83 |
| 低资源，lr 1e-3（对照） | 91.47 ± 0.66 | – | 88.81 ± 0.83 |

- G2 与 G1 没有差别：幅值本身不带来收益。
- G3 在主实验上比两者略高，但不显著。
- 低资源下 G1 显著高于 G3。G1 改用与 G3 相同的 1e-3 后结果不变，G3 − G1 = −2.66 [−3.22, −2.11]，所以差距不是学习率造成的。

完整的表格、逐种子结果、第 0 步梯度和结论见 [`PROGRESS.md`](PROGRESS.md)。
