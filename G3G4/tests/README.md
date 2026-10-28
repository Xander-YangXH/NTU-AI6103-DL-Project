# tests/ — 离线端到端测试

| 文件 | 作用 |
|---|---|
| `test_tiny.py` | 用一个微型 RoBERTa 和合成数据，在 CPU 上把 G3、G4 的完整流程各跑一遍。约 1 分钟，不需要网络和 GPU |

测试会替换 `train.py` 中加载模型和数据的调用。微型模型有 2 层 × 6 个目标层、hidden 64、FFN 256、词表 64。合成数据为 train 1,200 条、validation 200 条，token 分布依赖标签，所以任务可学。

## 检查内容

- **G3**：PEFT 重分配的秩总数等于 r × 12，每层不超过 r·rho = 16。
- **G4**：
  - LoRA 参数量与预算 18,432 的差距小于一层的最小成本；
  - 每层秩不超过 16；
  - 每层 A 等于缓存中的前 k 个分量，B 全为 0。
- **缓存**：解释方差比在 (0, 1] 区间内、总和不超过 1，且大体递减。
- **G3 规则核对**：用 `alloc.eva_rank_count` 在缓存上重算的分配，与 PEFT 的 G3 分配接近。
- **断点续跑**：重复运行同一个配置时会被跳过。
- **训练信号**：step 0 时 B 的梯度大于 0，loss 曲线已记录。

最后打印 `all checks passed`。

## 运行

```bash
python tests/test_tiny.py                                   # Linux / 集群
..\code\.venv-prep\Scripts\python.exe tests\test_tiny.py    # 本机 Windows
```

在集群上，这个测试也包含在 `jobs/smoke.sbatch` 中。
