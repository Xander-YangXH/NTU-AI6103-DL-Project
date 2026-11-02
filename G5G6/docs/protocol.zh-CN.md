# G5/G6 协议、数学范围与可比性

本目录是2026-10-30服务器运行的整理版。仓库格式参考`yinchenyu_g3g4`的`2661637`和`liyouran`的`0c1b768`；共享设置以`main`的`6546cb1`为参照。代码整理只调整目录定位、revision固定及入口，不改变已导出结果。

| 项目 | 本次已完成G5/G6 | 仓库当前共享设置 |
|---|---|---|
| epochs：RTE/MRPC/SST-2 | 80/30/10 | 10/10/3 |
| 最大长度 | 512，动态padding | 128，动态padding |
| batch：RTE/MRPC/SST-2 | 32/16/32 | 均32 |
| 训练内划分 | 分层抽样seed0，重复文本整组归属 | NumPy seed2026 |
| 正式训练集 | 全量官方train | 保留90% train |
| 最终LR网格 | 1e-4、4e-4、1e-3 | 2e-4、5e-4、1e-3 |
| MRPC选参指标 | mean(accuracy,F1)，正式报告两项 | F1 |
| 校准样本/batch | 256/2，保留标签产生梯度 | 普通EVA1024/16 |
| gradient clipping | norm1 | 无 |
| 预热 | 原MLP分类头；encoder.eval，head.train；batch32，lr1e-3，wd0 | 共享实现有不同train模式及默认预热wd |
| 依赖 | torch2.8.0、transformers4.45.1、peft0.12.0 | 其他分支使用更新版本 |

因此本目录结果只能描述本次G5/G6内部比较，不能直接支持与共享协议G3/G4/G7的H2因果比较。没有运行低资源SST-2 1k实验；也没有声称其已完成。

## 数学及实现

对同一个token，`z=||g||₂x`使`zᵀz`的统计对应未中心化`Σ_g=E[||g||²xxᵀ]`。分类loss按样本取mean时，采集的输出梯度乘实际batch大小，按每样本loss定义加权。真实batch2与逐样本对照的72层最大梯度范数误差约1.19e-7，详见`results/diagnostics/gradient_hook_check.json`。

PEFT的IncrementalPCA会减均值，目标变成`Cov(z)`；本实现GW路径不减均值。序列级B梯度是跨token外积之和，其平方含交叉项，所以逐tokenΣ_g只是指定代理目标，不能宣称它精确最大化任意句子级总B梯度。

每次将`diag(s_old)V_old`与新行合并做精确skinny SVD，再截断到16行。全历史SVD因此是近似，不能宣称精确总体特征分解。保存总原始能量作谱分母，而非只对保留16方向再次归一化；保存主子空间收敛记录。固定256样本停止不代表全部层已收敛，逐层标记在`results/initialization/`。

G5按照PEFT EVA的数量预算语义对归一化谱分数全局选576项，有序分数保证前缀；G5根据加权谱重新分配，未复制G3 pattern。G6使用每rank成本`d_in+d_out`，以768为成本单位的多选背包DP选每层前r个方向；只对给定近似候选谱分数保证最优。二分类头592,130参数另计，G5参数量可能偏离均匀r8预算。

零梯度、空行、少样本、重复特征值及NaN拒绝均有测试；不足16方向用零收益正交补全，不能伪造标签信息。每层A行单位范数、B=0、初始输出与预热模型完全一致。Encoder在校准与LoRA训练期间不更新。

## 选参及记录

最终三档LR对两个方法及三个任务一致，seed0在训练内holdout选参；选参的头预热/校准只访问fit，正式阶段重新在全train预热/校准。平局选择较低LR，官方validation只在最终epoch评估。

早期较高LR探索因fit loss不下降而调整网格；早期microbatch8×累积4也在资源测试后改为完整batch。探索记录完整保存在本地备份，不混入这里18个最终网格tune或26个main结果。此为训练内开发调整，不冒充从未试过其他配置的预注册网格。

所有main保存最终Adapter并重载验证，记录前500步真实loss和clip前B-only梯度。训练用BF16、FP32参数/梯度；不是额外eval模式的step0梯度测量。未进行逐epoch官方dev评估，不填写虚构的per_epoch验证曲线。
