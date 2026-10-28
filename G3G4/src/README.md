# src/ — 核心代码

本文件夹包含 G3/G4 每次运行用到的全部 Python 模块。

`jobs/` 里的脚本和 `tests/test_tiny.py` 都从这里导入。运行时以本文件夹的上一级（`G3G4/`）作为工作目录，例如：

```bash
python src/train.py --stage main --task rte --group G4 --lr 2e-4 --seed 0
```

| 文件 | 作用 | 关键内容 |
|---|---|---|
| `train.py` | **单次运行入口**：分类头预热 → EVA 初始化（G3/G4）→ LoRA 微调 → 每个 epoch 评估 → 写结果 JSON | `CFG` 字典里是全部超参数（r=8、alpha=8、rho=2、预算 1,327,104、batch 32、epoch 等）；`metrics`（acc / MRPC F1）；`warm_up_head`；结果原子写入 `results/<stage>/<run_id>.json`，已完成的运行自动跳过 |
| `init_g34.py` | **G3 与 G4 的初始化** | `TARGET` 正则匹配 72 个目标层；`init_g3` 调用 PEFT 内置 EVA；`eva_cache` 生成每层前 16 个 EVA 分量和解释方差比，并缓存到 `cache/eva16_*.pt`；`init_g4` 用 `alloc.param_cost` 分配秩，再通过 `eva_state_dict` 写入 PEFT |
| `alloc.py` | **秩分配规则与计数** | `eva_rank_count`：G3 规则的重实现，只用于核对；`param_cost`：G4 的贪心背包，得分为解释方差比 / (d_in+d_out)；`lora_ranks`、`lora_param_count`、`head_param_count`；`uniform_budget`：均匀 r 的参数预算；`subspace_overlap`：子空间重合度 |
| `data.py` | **数据** | 从 `$GWEVA_DATA`（默认 `~/gweva-data`）读取本地 GLUE；固定划分（`SPLIT_SEED=2026`，10% 作 dev）；低资源 1k 子集；`DictCollator`（PEFT 的 EVA hook 不接受 BatchEncoding）；`calib_loader`（去掉标签的 1,024 条校准数据） |

## 运行 `train.py` 的参数

| 参数 | 取值 | 说明 |
|---|---|---|
| `--stage` | `pilot` / `tune` / `main` / `lowres` | pilot、tune 在 dev 上评估；main、lowres 在官方 validation 上评估；lowres 用 1k 子集训练 20 epoch |
| `--task` | `sst2` / `mrpc` / `rte` | |
| `--group` | `G3` / `G4` | |
| `--lr`、`--seed` | 浮点数、整数 | |
| `--out`、`--cache` | 默认 `results`、`cache` | 结果和缓存的目录 |

## 依赖与注意事项

- 依赖：transformers 5.21.0、peft 0.21.2、datasets 5.1.0、torch（集群上是 2.11.0+cu128）。
- 训练前需要设置环境变量 `GWEVA_DATA`，指向含 `roberta-base/` 和 `glue/` 的数据目录。sbatch 脚本会自动设为代码目录旁边的 `gweva-data`。
- 修改 `CFG` 或 `data.py` 中的划分常量，会改变实验设定，需要与其他组保持一致。
