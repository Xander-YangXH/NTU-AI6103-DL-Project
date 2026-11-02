# results/tune/ — 学习率调参结果（18 个运行，dev 集）

- 设置：G3 / G4 × SST-2 / MRPC / RTE × lr {2e-4, 5e-4, 1e-3}，seed 0。
- 评估集：只在 dev（官方 train 的 10%，`SPLIT_SEED=2026`）上评估，不使用官方 validation。
- Slurm 作业 48253，用时 39 分钟。
- 文件名为 `tune_<组>_<任务>_lr<lr>_s0.json`；日志见 `logs/tune_*.log`。

## 结果（dev primary %，SST-2 / RTE 为 acc，MRPC 为 F1）

| 组 | 任务 | 2e-4 | 5e-4 | 1e-3 | 选定 lr |
|---|---|---|---|---|---|
| G3 | SST-2 | 94.8 | 95.0 | **95.2** | 1e-3 |
| G3 | MRPC | **91.3** | 91.2 | 91.1 | 2e-4 |
| G3 | RTE | **79.1** | 48.2 | 48.2 | 2e-4 |
| G4 | SST-2 | 94.6 | **95.1** | 94.9 | 5e-4 |
| G4 | MRPC | **91.4** | 90.7 | 90.6 | 2e-4 |
| G4 | RTE | **77.5** | 65.9 | 48.2 | 2e-4 |

## 说明

- **选择规则**：dev primary 最高者胜，平局取较小学习率（`jobs/make_queue.py` 中的 `pick_best`）。选出的学习率写在 `results/chosen_lr.json`。
- **两组分别调参**：G3 和 G4 各自选择最适合自己的学习率，避免因为共用学习率而偏袒其中一组。
- **RTE 对学习率很敏感**：较大的学习率会塌缩到多数类（48.2%）。
- **EVA 缓存**：tune 阶段的 G4 运行生成了 `cache/eva16_{task}_False_s0.pt`，main 阶段 seed 0 的 G4 直接复用它。
