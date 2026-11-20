# G5/G6 共享配置复跑

2026 年 11 月 18 日完成 18 组选参、26 组常规正式训练和 6 组低资源 SST-2 正式训练。

查看[结果汇总](results/summary.md)、[学习率](results/chosen_lr.json)和[实验协议](docs/protocol.zh-CN.md)。执行方法见[英文说明](README.md)。

普通 EVA 校准可以移除标签。G5/G6 校准保留标签计算梯度，其余训练设置见协议说明。官方验证集只在训练结束时评估。
