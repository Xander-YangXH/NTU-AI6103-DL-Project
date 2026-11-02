<p align="right"><a href="README.md">English</a> | <b>中文</b></p>

# G5 / G6：梯度加权 EVA

负责人：韩彦泽（Yanze Han），分支 `hanyanze`。完成RTE/MRPC各5种子、SST-2共3种子，两组共26次正式训练和18次最终网格选参。

**这是独立协议实验，不是仓库共享协议的复跑。** 真实运行采用长度512、epochs80/30/10、全量train正式拟合；仓库当前规范为长度128、epochs10/10/3、保留90% train。划分种子、学习率网格、校准规模等也不同。仅按其他成员的目录和JSON格式整理，未改写历史配置和分数，不能直接和G3/G4/G7配对比较。[完整差异表](docs/protocol.zh-CN.md)。

## 方法

先冻结encoder预热原分类头1 epoch，再冻结全部权重，通过embedding输出requires_grad建立校准计算图。72个Linear输出逐token捕获梯度，剔除padding，将mean-loss梯度乘回实际batch大小，形成`z=||g||₂x`。目标为未中心化二阶矩，不能把普通中心化PCA冒充GW-EVA。

CPU增量SVD每次保留最多16方向；单次合并精确、累计截断为近似。G5按各层归一化谱分数选择576个rank；G6用整数前缀动态规划优化真实参数成本，预算1,327,104。两组共享同任务/种子的预热头及加权谱；A每行单位范数，B=0，alpha/r固定为1，分类头参数另计。

## 最终结果

官方validation，均值±样本标准差（百分数）：

| 指标 | 种子 | G5 | G6 |
|---|---:|---:|---:|
| RTE accuracy | 5 | 78.84 ± 2.00 | 78.84 ± 0.91 |
| MRPC accuracy | 5 | 88.04 ± 1.31 | 88.38 ± 1.72 |
| MRPC F1 | 5 | 91.37 ± 0.95 | 91.68 ± 1.17 |
| SST-2 accuracy | 3 | 93.92 ± 0.23 | 93.43 ± 0.73 |

三个任务的配对准确率差95% t区间均包含0，目前无明确统计优势，不等于证明两组等效。G5参数量1,281,024–1,352,448，G6固定1,327,104，未完全参数匹配；本轮没有EVA/LoRA基线，不能推出GW-EVA优于它们。[简报](docs/analysis.zh-CN.md)。

## 文件与复现

- `src/gw_eva/`：梯度钩子、流式SVD、两种分配、共享预热和训练。
- `jobs/`：公开数据下载、队列、可选Slurm模板、汇总和画图。
- `tests/`：6项离线数学测试及44份运行结果的完整性检查。
- `configs/`：模型/数据固定revision和SHA256、精确fit/selection索引。
- `results/main/`、`tune/`：分别26和18个JSON；字段参考G3G4，补充真实配置、预测、前500步曲线及协议标记。
- `results/initialization/`：13份共享初始化的逐层谱、梯度统计和G5/G6 rank，无二进制权重。

在`G5G6/`目录使用独立环境。依赖和完整命令见[英文README](README.md#run)。核心命令：

```bash
pip install -r requirements.txt
pip install -e . --no-deps
python tests/test_tiny.py
python tests/test_results.py
python jobs/prefetch.py
bash jobs/run_queue.sh --tasks rte
python jobs/summarize.py
python jobs/plot_diagnostics.py
```

服务器实测为Python3.12、torch2.8.0+cu128、RTX4090。校准FP32无TF32；训练BF16。Slurm模板未实际在TC2运行，不冒充已验证集群作业。每epoch保存完整恢复状态，已完成任务跳过。

Git不包含原始模型/数据、缓存、长日志、优化器状态、虚拟环境、个人连接信息或与G5/G6无关的实验。完整二进制备份已保留本地。[结果口径及缺失字段](results/README.md)。
