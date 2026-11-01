<p align="right"><a href="README.md">English</a> | <b>中文</b></p>

# GW-EVA：去掉独立性假设的梯度加权 LoRA 初始化

AI6103 Deep Learning 小组项目（Option 2：在已有代码库上做改进），六人一组。

- **基础论文**：EVA（Paischer 等，NeurIPS 2025）。它用激活的主成分初始化 LoRA 的 A 矩阵，并按解释方差重新分配各层 rank。
- **代码基础**：作者的官方仓库（MIT 许可），以及 Hugging Face PEFT 中的 EVA 实现。
- **我们的改动**：去掉 EVA 最优性证明中的一个独立性假设，推导出梯度加权的初始化 **GW-EVA**，并在 PEFT 上实现。

## 1. 研究问题

EVA 的最优性结论（定理 3.2）假设层输出梯度 $g = \partial L / \partial y$ 与输入激活 $x$ 统计独立。

如果不做这个假设，B 在初始化时（B = 0）收到的梯度信号精确等于：

$$
\mathbb{E}\left\Vert \frac{\partial L}{\partial B}\right\Vert _F^2 = \mathrm{Tr}\left(A\,\Sigma_g\,A^\top\right),\qquad
\Sigma_g = \mathbb{E}\left[\Vert g\Vert^2\, x x^\top\right]
$$

所以最优的 A 是 $\Sigma_g$ 的前 r 个特征向量，我们称之为 **GW-EVA**。当 $\Vert g\Vert^2$ 与 $x$ 无关时，它就退化为 EVA。

对大小为 m 的小批量：

$$
\mathbb{E}\left\Vert \bar G_m A^\top\right\Vert _F^2 = \tfrac{1}{m}\mathrm{Tr}\left(A\Sigma_g A^\top\right) + \left(1-\tfrac{1}{m}\right)\left\Vert \mathbb{E}[g x^\top]A^\top\right\Vert _F^2
$$

第二项的最优方向正是 LoRA-GA 用的均值梯度方向。因此 EVA、GW-EVA、LoRA-GA 是同一目标在不同假设下的解。

另一条主线是“LoRA 初始化需要多少任务信息”。按使用的任务信息从少到多，四种初始化排成一条线：

通用语料（G8）→ 别的任务输入（G9）→ 本任务输入（G3，EVA）→ 本任务输入 + 标签梯度（G5，GW-EVA）

## 2. 假设

| | 假设 | 若不成立 |
|---|---|---|
| **H1** | RoBERTa-base 中 $\Vert g\Vert^2$ 与 $x$ 相关，$\Sigma_g$ 与激活协方差的前 r 子空间有可测的差异（第 1 周，不训练即可检验） | 是对 EVA 假设的实证支持，GW-EVA 退化为 EVA，报告负结果 |
| **H2** | 参数量相同时，GW-EVA 第 0 步梯度更大、前 500 步 loss 更低、精度高于 EVA，至少在低资源设置下配对 95% CI 不含 0 | 说明多一次反向传播不值得 |
| **H3** | 按“每参数解释方差”而不是“每个 rank”分配，在固定参数预算下会改变结果（受控因素，不作为创新主张） | 原版按 rank 个数分配已经足够 |
| **H4** | EVA 的激活子空间带有任务特异的信息：把校准数据换成通用语料或别的任务，子空间和精度都会变 | 激活统计主要由模型决定，任务信息只能靠标签梯度引入，反而加强 GW-EVA 的动机 |

## 3. 实验组

核心是“方向 × 分配”的 2×2 因子设计（G3–G6），另加参照组。

| 组 | A 的方向 | rank 分配 | 回答的问题 | 分支 | 状态 |
|---|---|---|---|---|---|
| G1 LoRA | Kaiming 随机 | 均匀 r = 8 | 基线 | [`yxh`](https://github.com/Xander-YangXH/NTU-AI6103-DL-Project/tree/yxh/G1G2) | 进行中 |
| G2 幅值对照 | 随机正交，行范数与 EVA 相同 | 均匀 r = 8 | 收益来自方向还是幅值 | [`yxh`](https://github.com/Xander-YangXH/NTU-AI6103-DL-Project/tree/yxh/G1G2) | 进行中 |
| G3 EVA | 激活协方差 | 按 rank 个数（原版，ρ = 2） | 复现原方法 | [`yinchenyu_g3g4`](https://github.com/Xander-YangXH/NTU-AI6103-DL-Project/tree/yinchenyu_g3g4/G3G4) | 已完成 |
| G4 | 激活协方差 | 按参数成本 | 只换分配（H3） | [`yinchenyu_g3g4`](https://github.com/Xander-YangXH/NTU-AI6103-DL-Project/tree/yinchenyu_g3g4/G3G4) | 已完成 |
| G5 GW-EVA | $\Sigma_g$ | 按 rank 个数 | 只换方向（H2） | [`hanyanze`](https://github.com/Xander-YangXH/NTU-AI6103-DL-Project/tree/hanyanze/G5G6) | 13组完成（独立协议，非共享协议复跑） |
| G6 GW-EVA | $\Sigma_g$ | 按参数成本 | 方向与分配的组合 | [`hanyanze`](https://github.com/Xander-YangXH/NTU-AI6103-DL-Project/tree/hanyanze/G5G6) | 13组完成（独立协议，非共享协议复跑） |
| G7 LoRA-GA | 均值梯度（B ≠ 0，改写 W） | 均匀 r = 8 | 目标第二项的代表 | – | 未开始 |
| G8 EVA-generic | 通用语料（wikitext-103）的激活协方差 | 按 rank 个数 | 任务无关的一端（H4） | – | 未开始 |
| G9 EVA-cross | 别的任务的激活协方差（SST-2←MRPC，MRPC←RTE，RTE←MRPC） | 按 rank 个数 | 任务输入有多特异（H4） | – | 未开始 |

**模型与任务**：RoBERTa-base；GLUE SST-2、MRPC、RTE，另加 1,000 条的低资源 SST-2 子集（G1、G3、G5）。

**LoRA 位置与预算**：全部 72 个注意力与 FFN 线性层，预算固定为 **1,327,104** 个可训练参数（即均匀 r = 8 时的参数量）。

## 4. 共享实验设置

所有组必须使用以下设置，以便按种子做配对比较。实现以 `G3G4/src/` 中的 `data.py` 与 `train.py` 为准（G1/G2 已直接复用）。

| 设置 | 取值 |
|---|---|
| 学习率网格 | 2×10⁻⁴、5×10⁻⁴、1×10⁻³ |
| 种子 | SST-2 为 0–2，MRPC / RTE 为 0–4；调参用 0 |
| 训练轮数 | SST-2 3，MRPC 10，RTE 10，低资源 20 |
| 批大小 / 最大长度 / 预热比例 / 权重衰减 | 32 / 128 / 6% / 0；bf16 混合精度；不裁剪梯度 |
| LoRA α / dropout | α = 8（α/r = 1）/ 0；EVA 重分配后 PEFT 按 rank 等比例调整 α，使 α/r 不变 |
| 分类头预热 | 冻结编码器，训练 1 个 epoch，学习率 1×10⁻³ |
| dev 划分 | 训练集的 10%，`SPLIT_SEED = 2026`；低资源 1k 子集用种子 2027 |
| 最终训练数据 | 划掉 dev 后的 90%，与调参一致 |
| EVA 校准 | 训练集中随机 1,024 条，batch 16，去掉 labels；τ = 0.99，ρ = 2 |
| 选参与报告 | 取最后一个 epoch；在 dev 上选参，官方 validation 只在最终评估时使用 |
| 低资源学习率 | 沿用各组在 SST-2 上选出的学习率 |
| 结果格式 | 每次运行一个 JSON，字段见 `G3G4/src/train.py` 末尾 |
| 统计 | 同种子配对差，报告 95% 置信区间（t 分布，df = n − 1） |
| 指标 | SST-2 / RTE 准确率，MRPC F1 与准确率；第 0 步 ‖∂L/∂B‖、前 500 步 loss；初始化耗时、总耗时、峰值显存 |

已选定的学习率：

| 组 | SST-2 | MRPC | RTE |
|---|---|---|---|
| G3 | 1e-3 | 2e-4 | 2e-4 |
| G4 | 5e-4 | 2e-4 | 2e-4 |

G8 / G9 沿用 G3 的学习率。

## 5. 目前的结果

**G3 / G4**（官方 validation，mean ± sd，%）：

| 任务 | 种子数 | G3 | G4 | G4 − G3（95% CI） |
|---|---|---|---|---|
| SST-2 (acc) | 3 | 94.19 ± 0.43 | 93.65 ± 0.24 | −0.54 [−1.69, +0.62] |
| MRPC (F1) | 5 | 92.02 ± 0.79 | 91.50 ± 0.68 | −0.52 [−1.56, +0.52] |
| RTE (acc) | 5 | 74.80 ± 2.49 | 73.57 ± 2.06 | −1.23 [−4.31, +1.85] |

按参数成本分配（G4）没有带来提升，三个任务的差异都不显著。G3 低资源 SST-2 1k：88.81 ± 0.83。详见 [`G3G4/README.md`](https://github.com/Xander-YangXH/NTU-AI6103-DL-Project/tree/yinchenyu_g3g4/G3G4)。

**G1 / G2**：pilot 已通过，调参与主实验正在 CCDS TC2 上运行。进度见 [`G1G2/PROGRESS.md`](https://github.com/Xander-YangXH/NTU-AI6103-DL-Project/blob/yxh/G1G2/PROGRESS.md)。

## 6. 仓库组织

- 每人在自己的分支上工作，每组实验一个文件夹（如 `G1G2/`、`G3G4/`）。文件夹内包含代码、Slurm 脚本和结果 JSON。
- `main` 只放项目简介与公共约定。
- 大文件（模型权重、数据、EVA 缓存、日志）不进 git，用 `.gitignore` 排除。

| 分支 | 内容 |
|---|---|
| `main` | 本简介 |
| `yinchenyu_g3g4` | G3 EVA 复现、G4 按参数成本分配；包含共享的数据划分与训练管线 |
| `yxh` | G1 LoRA 基线、G2 幅值对照 |
| `hanyanze` | G5/G6：梯度加权方向与成本分配；独立协议，26组正式结果 |

## 7. 计算资源

CCDS GPU Cluster TC2，Slurm 调度。每人的 QoS 为 1 张 GPU、最多 2 个作业、30 GB 内存，单个作业不超过 6 小时。每个作业把多次训练打包，在同一张卡上并行 2 个进程；每次训练结束立刻写出 JSON，重新提交即可续跑。

## 8. 课程要求摘要

- Option 2：选 2019 年及以后、有公开代码的论文，在其基础上实现改进，允许复用原代码。
- 不得与组员个人项目的“模型 + 数据集”组合重复。
- 不得刻意压低基线结果。
- 评分共 30 分：新颖性与难度 10、技术方案 10、写作 10。
