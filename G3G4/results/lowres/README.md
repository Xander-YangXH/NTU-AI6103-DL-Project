# results/lowres/ — 低资源实验结果（5 个运行，官方 validation）

低资源设定用来给其他组提供 G3（原版 EVA）的基线。

- 训练数据：只用 SST-2 训练部分中的 1,000 条（种子 2027 抽取，所有组相同）。
- 训练设置：20 个 epoch，lr 取 G3 SST-2 在 tune 中选定的 1e-3。
- 只跑 G3，种子 0–4。
- EVA 校准数据也从这 1,000 条中抽取。
- 评估集：官方 SST-2 validation（872 条）。
- Slurm 作业 48271，用时 4 分钟；日志见 `logs/lowres_*.log`。

| 文件 | seed | acc (%) | LoRA 参数 | 用时 |
|---|---|---|---|---|
| `lowres_G3_sst2_lr0.001_s0.json` | 0 | 87.73 | 1,267,200 | 81 s |
| `lowres_G3_sst2_lr0.001_s1.json` | 1 | 90.02 | 1,276,416 | 82 s |
| `lowres_G3_sst2_lr0.001_s2.json` | 2 | 88.99 | 1,276,416 | 84 s |
| `lowres_G3_sst2_lr0.001_s3.json` | 3 | 88.65 | 1,271,808 | 84 s |
| `lowres_G3_sst2_lr0.001_s4.json` | 4 | 88.65 | 1,274,112 | 66 s |
| **平均** | | **88.81 ± 0.83** | | 79 s |

## 说明

- 对应的 G3 初始 A 矩阵保存在 `cache/g3A_sst2_True_s{0..4}.pt`，文件名中的 `True` 表示 lowres。
- 这组实验没有对应的 G4 缓存，所以 `summarize.py` 的一致性核对不包含 lowres。
