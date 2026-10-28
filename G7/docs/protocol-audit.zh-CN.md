# G7 与小组共同实验设置的对照

核对日期：2026-10-28。共同约定来自 main（`6546cb1`）的 [README](../../README.md) 第 4 节；训练与数据划分以 `origin/yinchenyu_g3g4`（`2661637`）中的 `G3G4/src/data.py`、`G3G4/src/train.py` 为准，学习率选择以 `G3G4/jobs/make_queue.py` 为准。G1/G2 使用同一份参考管线。

此前独立执行的 G7 使用合并提案与 EVA 论文补充设定。小组 main 已给出不同的统一配置，本分支按共同配置重新执行全部学习率搜索与正式种子。此前的 94.69/88.58/75.31 准确率不参与本分支的配对比较。

## 影响可比性的参数

| 项目 | 此前独立 G7 | 本分支小组协议 |
|---|---|---|
| 模型/任务 | RoBERTa-base；SST-2、MRPC、RTE | 相同 |
| LoRA 位置 | 72 个 attention/FFN 线性层 | 相同 |
| rank / alpha / scaling | 8 / 1 / 1/8 | 8 / 8 / 1 |
| LoRA dropout | 0 | 0 |
| 学习率网格 | 1e-4、4e-4、1e-3 | 2e-4、5e-4、1e-3 |
| SST-2 / MRPC / RTE epochs | 10 / 20 / 20 | 3 / 10 / 10 |
| batch | 64 / 32 / 64 | 全部 32；无梯度累积 |
| 最大长度 / padding | 512；补到 8 的倍数 | 128；按 batch 最长样本动态 padding |
| 正式训练 AdamW | weight decay 0、eps 1e-8 | 相同 |
| 调度 | linear；6% warmup 向上取整 | linear；`int(0.06*steps)`，向下取整 |
| 梯度裁剪 | norm 1 | 不裁剪 |
| 数据划分 | seed42 分层，保留原序 | NumPy seed2026 permutation，不分层；保留 permutation 顺序 |
| head 预热 | encoder eval；batch64；wd0；clip1 | 全模型 train、encoder 参数冻结；batch32；wd0.01；不裁剪 |
| 校准数量/抽样 | 256；Python random.sample | 1,024；NumPy default_rng(seed).permutation |
| 校准 loader | batch16；不 shuffle | batch16；shuffle=True，generator seed与实验seed相同 |
| epoch 报告 | 内部最佳 epoch | 最后一个 epoch |
| MRPC 选参指标 | (accuracy+F1)/2 | F1；同时报告 accuracy |
| LR 平局规则 | 更低内部交叉熵 | 更小学习率 |
| eval batch | 64 | 128 |
| seeds | SST-2 0–2；MRPC/RTE 0–4；调参0 | 相同 |
| 精度 | FP32 权重、BF16 autocast | 相同；G7 校准/权重补偿采用 FP32 |

参考代码 `warm_up_head` 调用 `AdamW(..., lr=1e-3)`，未传 weight decay，因此预热阶段实际使用默认 **0.01**。正式 LoRA 训练显式传入共同 CFG 的 weight decay **0**。编码器冻结不关闭 dropout，预热中整个模型保持 train 模式；这与此前独立实现的 encoder eval 不同。

同一个训练 DataLoader 及其 seeded generator 在 head 预热和所有训练 epoch 中连续使用，不按 epoch 重新播种。初始化后的 LoRA 微调也不重置全局 Torch RNG。模型原生 dropout 为 0.1，LoRA dropout 为 0。

## 完全共享的数据划分

`G7/src/data.py` 原样取自 `G3G4/src/data.py`。官方 train 的原始顺序经 `np.random.default_rng(2026).permutation(n)` 打乱，前 `round(0.1*n)` 为 dev，剩余为训练集。低资源索引用 2027 生成，但 G7 主实验不包括低资源任务。

| 任务 | 训练 / dev / 官方 validation | 训练 label0 / label1 | dev label0 / label1 |
|---|---|---|---|
| SST-2 | 60,614 / 6,735 / 872 | 26,832 / 33,782 | 2,948 / 3,787 |
| MRPC | 3,301 / 367 / 408 | 1,088 / 2,213 | 106 / 261 |
| RTE | 2,241 / 249 / 277 | 1,129 / 1,112 | 120 / 129 |

新旧 dev 的重叠分别只有 664/6,735、39/367、26/249 条，因此重新划分数据后重新调参。分类头与初始化缓存也重新生成。

`jobs/prepare_data.py` 将公开原始 parquet 写成共享管线需要的 DatasetDict 磁盘格式，不做额外清洗、去重或重排。模型原始文件不含随机分类头，分类头按每个实验 seed 新建。数据与模型使用固定公开来源：

- `FacebookAI/roberta-base`，revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b`。
- `nyu-mll/glue`，revision `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`。

本机复用 `/home/msai/youran005/DeepLearning/Project/data/gweva/raw/` 与 `models/roberta-base/`，输出至 `data/gweva-team/`。模型使用符号链接。外部机器可用 `--download --root PATH` 下载相同版本；训练设置 `GWEVA_DATA=PATH`。模型、数据、校准缓存和日志均在 git 之外。

## G7 保留标签的校准与初始化

参考 EVA 校准去掉 labels，因为 EVA 只使用输入激活。G7 的目标是均值监督梯度，因此在同样的 1,024 条训练样本和 batch16 上保留 labels。校准模型为 eval，分类头冻结，不做优化器更新；各 batch loss 乘样本数累积后除总样本数，得到逐样本平均交叉熵梯度。

采用作者版 LoRA-GA：`ArB2r`、`stable`、gamma16；`A=c*V[:,:r]^T`，`B=c*U[:,r:2r]`，`c=d_out**0.25/sqrt(16)`。近似 SVD 的 `q=4r`、`niter=4`。初始化后 `W -= (alpha/r)*B0@A0`，记录补偿前后 logits 与有效权重的误差。LoRA 参数为 1,327,104，分类头为 592,130，总训练参数为 1,919,234。

## 阶段、验证与报告

共执行 **22 次完整训练**：9 次 tune（3任务×3学习率，seed0）与13次 main（3+5+5种子）。学习率按 tune 最后一个 epoch 的 dev primary 选择；SST-2/RTE 为 accuracy，MRPC 为 F1；平局取较小学习率。

训练过程中 `per_epoch` 记录 dev，所有结果取最后 epoch，不选择最佳 epoch。main 训练结束后才在官方 validation 上评估一次，官方指标不参与选参。参考 G3/G4 的 main JSON 每 epoch 记录官方 validation；这里保留每 epoch 的评估调用次数而将其数据改为 dev，以维持 DataLoader 对 Torch RNG 的消耗节奏，同时把官方 validation 限制在最终评估。

第0步 B 梯度与前500步 loss 在真正的第一个训练 batch、train/BF16 模式下记录，与参考训练代码的口径一致。单组 mean±sd 与 G7−G3 同 seed 差值均按任务种子数计算；配对95%区间使用 Student t，df=n−1。输出 JSON 沿用共享训练字段，并增加 G7 初始化说明与来源。

## 环境版本

本项目既有环境是 Python3.9，使用 uv 管理。G7 当前兼容版本为 torch2.4.0+cu121、transformers4.52.4、peft0.17.1、datasets2.20.0，BF16训练；G7 移植作者版初始化函数。G3/G4 记录的环境为 transformers5.21.0、peft0.21.2、datasets5.1.0，Torch2.11.0+cu128。共同数据划分与训练参数按参考源码执行，软件版本属于跨成员结果的实验边界，均写入每次运行的 `versions`。SVD和BF16数值差异可能带来额外的跨环境波动。
