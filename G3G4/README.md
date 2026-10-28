# G3 / G4：EVA 复现与按参数成本分配秩（RoBERTa-base × GLUE）

AI6103 小组项目“改进 EVA”中由我负责的两组实验。分工按实验划分。

- **G3**：原版 EVA，使用 PEFT 0.21.2 内置实现，按“秩个数”重分配。
- **G4**：使用与 G3 相同的 EVA 方向，但按“每参数解释方差”在固定参数预算下分配秩。

本文件夹是整理后的完整、可复现副本，包含代码、环境、全部结果、缓存和日志。

> 一句话结论：在相同 LoRA 参数预算下，G4 在 SST-2、MRPC、RTE 上都略低于 G3（−0.5 到 −1.2 分），
> 但三个任务的 95% 置信区间都包含 0，差异不显著。按参数成本分配会把秩从 FFN 大量转移到注意力层，
> 却没有带来收益。

---

## 1. 目录结构

```
G3G4/
├── README.md            本文件：配置、架构、流程、结果、结论
├── requirements.txt     pip 依赖（固定 transformers / peft / datasets 版本）
├── env_gweva.yml        集群 conda 环境 gweva 的完整导出（conda env export）
├── pip_freeze.txt       集群环境的 pip freeze（含 torch 2.11.0+cu128）
├── .gitattributes       强制 .py/.sh/.sbatch/.txt/.md 使用 LF 换行（Slurm 不认 CRLF）
├── .gitignore           推 GitHub 时忽略 logs/、cache/、jobs/queues/、gweva-data/ 等
├── src/                 核心代码：数据、秩分配、G3/G4 初始化、单次训练
├── jobs/                Slurm 脚本、队列生成、数据下载、汇总、画图
├── tests/               离线小模型端到端测试（CPU，约 1 分钟）
├── results/             全部 53 个运行的 JSON、汇总 CSV/TXT、两张热力图
│   ├── pilot/  tune/  main/  lowres/
├── cache/               EVA 分量缓存（G4 用）与 G3 初始 A 矩阵（核对用），共 107 MB
└── logs/                每个运行的训练日志与 Slurm 作业输出
```

每个子文件夹里都有自己的 README，说明里面每个文件的用途。

以下内容**没有放进**本文件夹：

| 位置 | 内容 | 未复制原因与获取方式 |
|---|---|---|
| `code\gweva-data\` | RoBERTa-base 原始权重与 GLUE 三个任务（487 MB） | 属于公开数据，体积大。可用 `jobs/prefetch.py` 重新下载 |
| `code\upload\` | 上传集群用的 tar 包与 SHA256SUMS | 由 g34 与 gweva-data 打包而来，属于中间产物 |
| `code\.venv-prep\` | 本机 Windows Python 3.11 环境 | 与机器绑定，按 `requirements.txt` 重建即可 |

---

## 2. 实验配置

### 2.1 模型、任务与数据划分

| 项目 | 设置 |
|---|---|
| 预训练模型 | `FacebookAI/roberta-base`，分类头在每个种子下随机初始化，`num_labels=2` |
| 任务 | GLUE SST-2、MRPC、RTE，外加 SST-2 1k 低资源子集 |
| 评价指标（primary） | SST-2 / RTE 用 accuracy，MRPC 用 F1（同时记录 acc） |
| dev 划分 | 官方 train 的 10% 作为 dev，用于选学习率；`SPLIT_SEED=2026`，全组统一 |
| 训练集 | 官方 train 剩余的 90%：SST-2 60,614、MRPC 3,301、RTE 2,241 条 |
| dev 集 | SST-2 6,735、MRPC 367、RTE 249 条 |
| 测试集 | 官方 validation：SST-2 872、MRPC 408、RTE 277 条。只在 main 和 lowres 阶段使用 |
| 低资源子集 | 从 90% 训练部分抽 1,000 条 SST-2（种子 2027） |
| 最大长度 | 128，动态 padding |

### 2.2 LoRA 与 EVA

| 项目 | 设置 |
|---|---|
| 目标层 | 12 层 × 6 种 = 72 个线性层：Q、K、V、attention.output.dense、intermediate.dense（FFN-in）、output.dense（FFN-out） |
| 不含 | `classifier.dense`（分类头本身全量训练） |
| r / alpha / dropout | 8 / 8 / 0 |
| ρ | 2.0，即每层秩上限为 r·ρ = 16 |
| 参数预算 | 均匀 r=8 的 LoRA 参数量。注意力 48 层 × 8 × (768+768) + FFN 24 层 × 8 × (768+3072) = **1,327,104** |
| EVA 校准数据 | 训练集中随机 1,024 条，batch 16，去掉标签（`use_label_mask=False`） |
| B 矩阵 | 初始化为 0，A 矩阵为 EVA 的主成分方向 |

### 2.3 训练

| 项目 | 设置 |
|---|---|
| 分类头预热 | 冻结编码器，只训练分类头 1 个 epoch，lr 1e-3（G3/G4 相同） |
| 优化器 | AdamW，weight decay 0，线性 warmup 6%，然后线性衰减 |
| batch size | 32 |
| epoch | SST-2 3、MRPC 10、RTE 10、低资源 20 |
| 精度 | bf16 autocast |
| 报告指标 | 最后一个 epoch 的结果（不做 early stopping，每个 epoch 的结果都记录在 JSON 里） |
| 学习率网格 | {2e-4, 5e-4, 1e-3}，seed 0 在 dev 上调参 |
| 选择规则 | dev primary 最高者胜，平局取较小学习率 |
| 种子 | SST-2 0–2，MRPC / RTE 0–4，低资源 0–4 |

---

## 3. 实验架构

### 3.1 单次运行的流水线（`src/train.py`）

```
加载 tokenizer、数据划分（data.py）
  → 加载 RoBERTa-base，核对 72 个目标层和参数预算
  → 分类头预热 1 epoch（编码器冻结）
  → LoRA 初始化
       G3：init_g34.init_g3   PEFT 内置 EVA（增量 SVD + 按秩个数重分配）
       G4：init_g34.init_g4   读取或构建 EVA r=16 缓存 → alloc.param_cost 分配 → eva_state_dict 写入 PEFT
  → LoRA 微调（AdamW + 线性调度，bf16）
  → 每个 epoch 评估一次
  → 原子写入 results/<stage>/<run_id>.json
```

`run_id` 的格式是 `{stage}_{group}_{task}_lr{lr}_s{seed}`。
已存在结果的运行会被跳过，所以作业超时后重新提交同一队列即可续跑。

### 3.2 G3 与 G4 的唯一区别：秩分配规则

两组使用同一批 EVA 主成分方向，即每层校准激活协方差的前 16 个主成分。

**G3（原版 EVA）**
- 规则：把所有层的分量按解释方差比统一排序，取全局前 r×72 = 576 个，每层最多 16 个。
- 预算单位是“秩的个数”，因此实际参数量随分配结果浮动。

**G4（按参数成本）**
- 每个分量的得分是“解释方差比 / (d_in + d_out)”，也就是每个参数换来的解释方差。
- 按得分贪心装入，直到参数预算 1,327,104 用完：装不下的分量跳过，继续考虑更便宜的分量。
- 每层上限同样是 16。
- 每增加一个秩的成本：注意力层 1,536 参数，FFN 层 3,840 参数（贵 2.5 倍）。

**为什么说两组只差在分配**
- G4 的解释方差比由 `init_g34.explained_variance_ratios` 计算，口径与 PEFT 的 IncrementalPCA 一致：centred 协方差、非 padding token。
- 汇总时做了两项核对：
  - G3 实际 A 矩阵与缓存方向的子空间重合度；
  - “在缓存上重算 EVA 规则”与 PEFT G3 分配之间的 L1 距离。

  两项都证明两组的方向相同，见 5.4 节。

### 3.3 代码模块

| 文件 | 作用 |
|---|---|
| `src/data.py` | GLUE 加载、固定 dev 划分、低资源子集、DataLoader、EVA 校准 loader |
| `src/alloc.py` | `eva_rank_count`（G3 规则重实现，用于核对）、`param_cost`（G4 规则）、参数计数、子空间重合度 |
| `src/init_g34.py` | 目标层正则、LoRA/EVA 配置、`init_g3`、`init_g4`、EVA 缓存与解释方差比 |
| `src/train.py` | 单次运行入口，`CFG` 是全部超参数 |
| `jobs/make_queue.py` | 生成各阶段命令队列；main / lowres 阶段读取 tune 结果选学习率 |
| `jobs/run_queue.sh` | 在一张 GPU 上以 NPAR 个进程并行执行队列 |
| `jobs/*.sbatch` | smoke 检查、训练、汇总三种 Slurm 作业 |
| `jobs/summarize.py` / `jobs/plot_ranks.py` | 汇总统计表与秩热力图 |
| `tests/test_tiny.py` | 小模型 + 合成数据的离线端到端测试 |

---

## 4. 实验流程

| 步骤 | 内容 | 运行数 |
|---|---|---|
| 0 本机准备 | `prefetch.py` 下载模型与数据；`test_tiny.py` 通过；打包上传 | – |
| 1 smoke | 检查 GPU / CUDA / 版本 / 离线测试 / 数据样本数 | – |
| 2 pilot | lr 5e-4、seed 0：G3 三个任务 + G4 RTE，检查能否跑通 | 4 |
| 3 tune | 2 组 × 3 任务 × 3 个 lr，seed 0，在 dev 上评估 | 18 |
| 4 汇总 | tune 结束后汇总查看 dev 结果；正式选 lr 由 main 作业内的 `make_queue.py` 完成 | – |
| 5 main | 按选定 lr 在官方 validation 上跑全部种子：SST-2 3 个，MRPC / RTE 各 5 个 | 26 |
| 6 lowres | G3、SST-2 1k、lr 取 G3 SST-2 的 1e-3，5 个种子 | 5 |
| 7 汇总 | `summarize.py` 输出 CSV 与 summary | – |
| 8 本机画图 | 下载 results/cache/logs，本机运行 `summarize.py` 与 `plot_ranks.py` | – |

共 53 个运行，全部成功，没有超时或重跑。

### 4.1 复现命令

**集群上**（在 `~/myproject/gweva-g34` 中提交；数据放在 `~/myproject/gweva-data`）：

```bash
sbatch jobs/smoke.sbatch
sbatch --export=ALL,STAGE=pilot,NPAR=2 jobs/g34.sbatch
sbatch --export=ALL,STAGE=tune,NPAR=2  jobs/g34.sbatch
sbatch --export=ALL,STAGE=main,NPAR=2  jobs/g34.sbatch     # 依赖 tune 结果，自动写 results/chosen_lr.json
sbatch --export=ALL,STAGE=lowres,NPAR=2 jobs/g34.sbatch
sbatch jobs/summarize.sbatch
```

**本机上**（Windows PowerShell，在本文件夹中执行）：

```powershell
..\code\.venv-prep\Scripts\python.exe tests\test_tiny.py        # 离线测试
..\code\.venv-prep\Scripts\python.exe jobs\summarize.py         # 重新生成 CSV 与 g34_summary.txt（需要 results/ 和 cache/）
..\code\.venv-prep\Scripts\python.exe jobs\plot_ranks.py        # 重新生成两张热力图
```

---

## 5. 实验结果

### 5.1 学习率选择（dev，seed 0，primary %）

| 组 | 任务 | 2e-4 | 5e-4 | 1e-3 | 选定 |
|---|---|---|---|---|---|
| G3 | SST-2 | 94.8 | 95.0 | **95.2** | 1e-3 |
| G3 | MRPC (F1) | **91.3** | 91.2 | 91.1 | 2e-4 |
| G3 | RTE | **79.1** | 48.2 | 48.2 | 2e-4 |
| G4 | SST-2 | 94.6 | **95.1** | 94.9 | 5e-4 |
| G4 | MRPC (F1) | **91.4** | 90.7 | 90.6 | 2e-4 |
| G4 | RTE | **77.5** | 65.9 | 48.2 | 2e-4 |

### 5.2 主结果（官方 validation，mean ± sd，%）

| 任务 | 种子数 | G3 | G4 | G4 − G3 配对差（95% CI） |
|---|---|---|---|---|
| SST-2 (acc) | 3 | 94.19 ± 0.43 | 93.65 ± 0.24 | −0.54 [−1.69, +0.62] |
| MRPC (F1) | 5 | 92.02 ± 0.79 | 91.50 ± 0.68 | −0.52 [−1.56, +0.52] |
| RTE (acc) | 5 | 74.80 ± 2.49 | 73.57 ± 2.06 | −1.23 [−4.31, +1.85] |

置信区间使用同种子配对差的 t 分布（df = n−1）。

**LoRA 实际参数量**（预算 1,327,104）：

| 组 | SST-2 | MRPC | RTE |
|---|---|---|---|
| G3 | 1,267,200（−4.5%） | 1,258,906（−5.1%） | 1,246,925（−6.0%） |
| G4 | 1,326,592（−0.0%） | 1,326,797（−0.0%） | 1,326,950（−0.0%） |

G4 与预算的偏差都在 ±0.1% 以内。

**低资源**：SST-2 1k，仅 G3，5 个种子，结果 **88.81 ± 0.83**（87.73 / 90.02 / 88.99 / 88.65 / 88.65）。供其他组对照。

### 5.3 秩分配（所有运行，按层类型平均）

| 组 | Q | K | V | Attn-out | FFN-in | FFN-out |
|---|---|---|---|---|---|---|
| G3 | 8.8 | 8.8 | 8.8 | 8.0 | 8.8 | 4.7 |
| G4 | 13.7 | 13.6 | 13.6 | 13.0 | 4.8 | 2.4 |

与 G3 相比，G4 在三个任务上的总变化是：注意力层 +202 到 +246 个秩，FFN 层 −65 到 −78 个秩。

逐层分布见以下两张图（12 层 × 6 种，主阶段种子平均）：
- `results/g34_rank_heatmap.png`
- `results/g34_rank_heatmap_diff.png`（G4 − G3）

### 5.4 G3 / G4 方向一致性核对（13 个 main 的 (task, seed)）

- G3 实际 A 矩阵与 EVA 缓存方向的平均子空间重合度为 0.974–0.990。
- 在缓存上重算 EVA 规则，与 PEFT 的 G3 分配相比，L1 距离为 10–26（总共 576 个秩）。

也就是说，两组用的是同一批方向，差别几乎全部来自分配规则。

### 5.5 资源开销（main 阶段，每个运行的平均值）

| | SST-2 | MRPC | RTE | lowres |
|---|---|---|---|---|
| G3 总用时 | 436 s | 139 s | 143 s | 79 s |
| G4 总用时 | 451 s | 210 s | 223 s | – |

- EVA 初始化：G3 17–32 s；G4 34–103 s（多一次解释方差计算）。如果复用 tune 阶段已有的缓存，G4 不到 1 s。
- 峰值显存 1.3–2.4 GB。

---

## 6. 实验说明与注意事项

1. **dev 与 test 严格分开。** 学习率只在 dev（train 的 10%）上选；官方 validation 只在 main 和 lowres 阶段使用，没有用来做任何选择。
2. **G3 用的参数比预算少约 5%。** 原因是 EVA 的预算是“576 个秩”：G3 把一部分秩从 FFN-out 挪到了更便宜的注意力层。G4 精确用满预算，所以 G4 实际多用约 5% 的参数，但仍然没有更好。
3. **pilot 中 G3 RTE 停在多数类（48.2%）。** 当时 lr 是 5e-4。tune 显示 RTE 在 5e-4 / 1e-3 下会塌缩，2e-4 正常，于是 main 用了 2e-4，main 中没有塌缩的运行。
4. **EVA 缓存复用。** `cache/eva16_{task}_False_s{seed}.pt` 由 tune 阶段生成，main 阶段的同种子 G4 直接复用，所以 G4 seed 0 的初始化时间很短。不同阶段的方向完全相同。
5. **非确定性。** GPU 上的 bf16 训练不是逐位可复现的，重跑结果会有小幅波动，但结论不变。pilot 与 tune 中配置相同的运行结果一致到 0.1 分。
6. **分类头。** `prefetch.py` 用 `snapshot_download` 原样保存 hub 权重（不含分类头），所以每个种子都会得到不同的随机分类头。
7. **与合并提案的一处出入。** 提案要求“参数量与预算偏差在 1% 以内”，但 G3 按定义只控制秩的个数。建议把这条改为只约束 G4 及其他按参数分配的组，G3 则如实报告偏差。
8. **推 GitHub 时。** `.gitignore` 会排除 `cache/`（107 MB）和 `logs/`，只提交代码与 `results/`。本文件夹保留了它们作为完整记录。

---

## 7. 实验结论

1. **G3 复现跑通。** RoBERTa-base + EVA（r=8、rho=2）在本设定下于 SST-2、MRPC、RTE 上分别达到 94.2 / 92.0（F1）/ 74.8，作为 G4 和其他组的同设定基线。
2. **G4 没有带来提升。** 在相同参数预算下，G4 三个任务的均值都比 G3 低 0.5–1.2 分，但配对 95% CI 都跨过 0，统计上与 G3 无显著差异。即使 G4 多用了约 5% 的参数，也没有超过 G3。
3. **分配行为差异很大，性能差异却很小。** 按“每参数解释方差”打分时，注意力层每个秩只花 1,536 参数，FFN 要 3,840，所以 G4 把注意力层几乎填到上限 16（Q/K/V 平均约 13.7），FFN-out 压到约 2.4。热力图显示这种转移在三个任务上高度一致。
4. **解读。** 解释方差衡量的是输入激活被保留的程度，并不直接对应下游任务的收益。按参数归一化后，这个指标会系统性地偏向便宜的注意力层，而削弱 FFN 的适配容量，在这三个任务上没有转化为精度提升。因此，“把 EVA 的预算单位从秩个数改成参数量”这一改进思路在本设定下不成立。原版 EVA 按秩个数分配反而更稳。
5. **成本。** G4 需要额外一次前向来计算解释方差，初始化时间约为 G3 的 2–3 倍；但相对于整次训练，开销很小。
6. **可供后续使用。** 后续各组需要从 G3/G4 取用的数据，直接列在下面的 7.1 节。

### 7.1 后续各组需要的数据

**G3 / G4 选定的学习率**（G8 / G9 必须沿用 G3 的学习率；文件为 `results/chosen_lr.json`）

| 组 | SST-2 | MRPC | RTE |
|---|---|---|---|
| **G3** | **1e-3** | **2e-4** | **2e-4** |
| G4 | 5e-4 | 2e-4 | 2e-4 |

低资源实验沿用各组在 SST-2 上选出的学习率，G3 为 1e-3。

**G3 main 每个种子的结果**（官方 validation，%；用于 G5、G8 / G9 的同种子配对比较，以及 G1 的复现对照）

| 任务 | lr | s0 | s1 | s2 | s3 | s4 | mean ± sd |
|---|---|---|---|---|---|---|---|
| SST-2 (acc) | 1e-3 | 93.69 | 94.38 | 94.50 | – | – | 94.19 ± 0.43 |
| MRPC (F1) | 2e-4 | 92.06 | 92.63 | 91.41 | 91.07 | 92.93 | 92.02 ± 0.79 |
| RTE (acc) | 2e-4 | 76.53 | 75.81 | 75.45 | 70.40 | 75.81 | 74.80 ± 2.49 |

**G3 低资源结果**（SST-2 1k，20 epoch，lr 1e-3，官方 validation acc %；供 G1 和低资源对比使用）

| s0 | s1 | s2 | s3 | s4 | mean ± sd |
|---|---|---|---|---|---|
| 87.73 | 90.02 | 88.99 | 88.65 | 88.65 | 88.81 ± 0.83 |

**G3 第 0 步 B 的梯度范数 `step0_grad_B`**（供 G5 检验 H2 的初始化信号）

| 任务 | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|
| SST-2 | 12.457 | 27.907 | 34.809 | – | – |
| MRPC | 1.072 | 1.706 | 1.492 | 1.850 | 2.010 |
| RTE | 2.785 | 1.149 | 0.658 | 1.532 | 0.637 |
| SST-2 低资源 | 3.026 | 1.388 | 1.501 | 1.985 | 3.033 |

**按使用方列出的文件**

| 使用方 | 需要的内容 | 对应文件 |
|---|---|---|
| G8 / G9（H4） | G3 的学习率（见上表） | `results/chosen_lr.json` |
| | 用于 G3 / G8 / G9 子空间重合度的 EVA 方向（计算函数见 `src/alloc.py` 的 `subspace_overlap`） | G3 实际 A 矩阵 `cache/g3A_{task}_False_s{seed}.pt`；EVA 前 16 个分量与解释方差比 `cache/eva16_{task}_False_s{seed}.pt` |
| | G3 main 结果，用于配对比较 | `results/main/main_G3_*.json`，汇总见上表 |
| G6 | G4 的分配函数，G6 必须使用同一个实现：`param_cost(ratios, costs, param_budget, cap)`，得分为解释方差比 / (d_in+d_out)，贪心装入预算 1,327,104，每层上限 16 | `src/alloc.py` |
| | G4 的逐层秩，供对照 | `results/g34_ranks.csv`（只看 group = G4 的行） |
| G5（H2） | G3 同种子的指标、`step0_grad_B`、前 500 步 loss，用于配对比较 | `results/main/main_G3_*.json` 中的 `final`、`step0_grad_B`、`loss_first_steps` 字段；汇总见 `results/g34_runs.csv` |
| G1 / 低资源对比 | G3 低资源的 5 个结果，以及 G3 main 结果 | `results/lowres/*.json`、`results/main/main_G3_*.json` |

注意：`cache/` 不进 git（107 MB）。需要时通过组里的共享盘发给 G8 / G9 的同学，其余文件都会随 `results/` 和 `src/` 推送到个人分支。
