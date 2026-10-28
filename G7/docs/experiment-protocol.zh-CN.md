# G7 LoRA-GA 实验协议

实验采用 main 提交 `6546cb1` 的公共参数。共享数据与训练管线的来源提交为 `26616379899f0abce7d693e227ad4e03138312ab`，G7 的初始化依据作者仓库 `Outsider565/LoRA-GA` 提交 `c4cd5372c75b290924214b348008891f744512ef`。

## 数据与划分

模型为 `FacebookAI/roberta-base`，revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b`；数据为 `nyu-mll/glue`，revision `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`。原始 parquet 转为 Hugging Face DatasetDict，保留样本、标签和原始顺序。分词最大长度 128，按 batch 最长序列动态补齐。

训练集按 `np.random.default_rng(2026).permutation(n)` 排列，前 `round(0.1*n)` 条作为 dev，其余 90% 用于预热、校准和训练。

| 任务 | 训练 / dev / 官方 validation | 训练 label0 / label1 | dev label0 / label1 |
|---|---|---|---|
| SST-2 | 60,614 / 6,735 / 872 | 26,832 / 33,782 | 2,948 / 3,787 |
| MRPC | 3,301 / 367 / 408 | 1,088 / 2,213 | 106 / 261 |
| RTE | 2,241 / 249 / 277 | 1,129 / 1,112 | 120 / 129 |

## 分类头预热与梯度校准

各次运行先按实验 seed 创建模型和二分类头，冻结编码器参数，整个模型保持 train 模式，原生 dropout 为 0.1。使用 batch32、AdamW、学习率 1e-3、默认 weight decay 0.01，训练分类头一个 epoch。

随后以 `np.random.default_rng(seed).permutation(trainN)[:1024]` 选取带标签校准样本，校准 loader 采用 batch16、shuffle=True、generator seed 与实验 seed 相同。模型置为 eval，固定预热后的分类头，以 FP32 估计 72 个目标权重的平均交叉熵梯度。每批 loss 乘样本数再反向传播，最后除总样本数，得到逐样本等权平均梯度；校准阶段不更新参数。

## LoRA-GA 初始化

全部 72 个 attention/FFN 线性层使用均匀 rank8、alpha8、LoRA dropout0，缩放 `s=alpha/r=1`。LoRA 参数量为 1,327,104，分类头为 592,130，总训练参数为 1,919,234。

对均值梯度 `G=U S Vᵀ`，作者版 `ArB2r`、`stable`、gamma16 设置为：

```
c = d_out**0.25 / sqrt(16)
A0 = c * V[:, :r].T
B0 = c * U[:, r:2*r]
W_residual = W0 - s * B0 @ A0
```

近似 SVD 使用 `q=4r`、`niter=4`。两侧均为缩放后的奇异向量。初始有效权重满足 `W_residual+s*B0@A0=W0`；实现记录 FP32 有效权重误差及相同校准 batch 的 logits 误差。

## 训练与选参

| 项目 | 设置 |
|---|---|
| 学习率网格 | 2e-4、5e-4、1e-3；调参 seed0 |
| 正式种子 | SST-2：0–2；MRPC/RTE：0–4 |
| epochs | SST-2：3；MRPC/RTE：10 |
| batch / 梯度累积 | 32 / 1 |
| 优化器 | AdamW；weight decay0；不裁剪梯度 |
| 调度 | linear；warmup steps=`int(0.06*total_steps)` |
| 精度 | FP32 参数，BF16 autocast |
| 训练 loader | 预热与正式训练连续使用同一个 seeded generator |
| 选参指标 | 最后 epoch 的 dev 指标；MRPC F1，其余 accuracy；并列选较小 LR |
| 正式评估 | 最后 epoch，官方 validation 一次 |

逐 epoch 指标记录在 dev 上。官方 validation 只在 main 训练结束后评估，预测和标签保存在对应 JSON。每次运行记录真正第一批训练的 B 梯度范数，以及前 500 次 optimizer update 的 loss。

共执行 9 次 tune 和 13 次 main，22 次完整训练。三个任务均选择学习率 2e-4。结果格式包括 `final`、`per_epoch`、`step0_grad_B`、`loss_first_steps`、`time_s`、参数量、GPU、版本和初始化诊断。

## 执行、重建与汇总

使用项目既有 Python3.9 虚拟环境，通过 uv 管理 `requirements.txt`。实测版本为 torch2.4.0、transformers4.52.4、peft0.17.1、datasets2.20.0。Slurm 作业申请 1 张 L40S、8 CPU、28 GiB 内存，两个进程共享 GPU。

数据准备使用 `jobs/prepare_data.py`；训练通过 `jobs/run_g7.sbatch` 提交；`jobs/summarize.py` 从本目录的运行 JSON 生成统计、分析和曲线。完成的 run 在重新提交时跳过。

本地 `cache/<run_id>/` 保存初始 A0/B0、分类头及最终可训练参数。重建时先载入原 RoBERTa-base 和初始参数，执行权重补偿，再载入最终训练参数。模型权重、原始数据、缓存和日志保留在 Git 之外。

结果统计采用种子间均值、样本标准差和 Student t 均值 95% 区间。固定官方验证集上的种子变化是区间刻画的随机性。完整解读见 [中文分析](analysis.zh-CN.md) 与 [自动汇总](../results/summary.md)。
