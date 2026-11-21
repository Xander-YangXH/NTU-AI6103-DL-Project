# report/ — 项目报告（AAAI-24 模板）

当前修订稿 `main.tex` 是第三版（6 页版）：在第二版基础上压缩到课程要求的篇幅（AAAI 格式，正文最多 6 页，不含参考文献），内容和数字与第二版一致。使用 AAAI-24 CameraReady 模板。

主结果来自各组的逐次运行记录，汇总入口为 `summary/all_runs.csv` 和 `summary/tables.md`。H1 子空间诊断见 `H1 Diagnosis/results/`，G5/G6 硬件记录见 `G5G6/results/provenance.json`。

## 文件

| 文件 | 作用 |
|---|---|
| `main.pdf` | 编译好的报告：双栏，共 7 页，正文第 1–6 页，参考文献在第 7 页另起一页 |
| `main.tex` | LaTeX 源文件，单文件（AAAI 不允许用 `\input` 拆分正文） |
| `refs.bib` | 参考文献，14 条，作者、标题和会议已逐条核对 |
| `aaai24.sty`、`aaai24.bst` | AAAI-24 模板的样式文件和参考文献格式，原样取自 AuthorKit24/CameraReady，没有改动 |
| `figures/` | 图片，均为矢量 PDF（报告用到其中 4 张） |

`figures/` 中每张图对应报告中的编号和生成脚本（脚本路径均相对于 main 分支的仓库根目录）：

| 文件 | 报告中的位置 | 生成脚本 |
|---|---|---|
| `h1_overlap.pdf` | 图 1：H1，EVA 与 GW-EVA 的子空间重合度 | `summary/figures.py` |
| `step0_vs_accuracy.pdf` | 图 2：第 0 步梯度与精度 | `summary/figures.py` |
| `lowres_lr.pdf` | 图 3：低资源下精度与学习率的关系 | `summary/figures.py` |
| `g34_rank_heatmap_diff.pdf` | 图 4：H3，G4 − G3 逐层秩差 | `G3G4/jobs/plot_ranks.py` |
| `h4_layers.pdf` | 第三版未使用（受篇幅限制，H4 的逐层结果改为正文叙述） | `summary/figures.py` |

在仓库根目录运行 `python summary/figures.py --report` 可生成图 1–3 和 `h4_layers.pdf` 的报告版 PDF。图 4 在 `G3G4/` 中运行 `python jobs/plot_ranks.py` 后，将 `results/g34_rank_heatmap_diff.pdf` 复制到 `report/figures/`。

## 编译

**Overleaf**：把本文件夹的全部文件（含 `figures/`）上传到一个新项目，编译器选 pdfLaTeX，点 Recompile 即可。

**本机**（需要 TeX Live，包含 times、helvet、courier、natbib、caption、amsmath、amssymb、array、booktabs）：

```bash
cd report
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

编译会生成 `.aux`、`.log`、`.bbl`、`.blg` 等中间文件，不要提交。

## 报告结构

| 节 | 内容 |
|---|---|
| Abstract | 问题、方法、主要结果和结论 |
| 1 Introduction | 动机：EVA 的最优性依赖“输出梯度与输入独立”的假设；本文的四点贡献 |
| 2 Background and Related Work | LoRA；EVA 的具体做法（增量 SVD、收敛判据、解释方差比、秩重分配）；其他数据驱动初始化与秩分配方法 |
| 3 Method | 第 0 步梯度恒等式；小批量视角下统一 EVA、GW-EVA 和 LoRA-GA；GW-EVA 算法；按参数成本分配（G4/G6）；对照组 G2、G7、G8、G9 的实现 |
| 4 Experimental Design | 9 个实验组（表 1 含各组选定的学习率）、假设 H1–H4、共享协议、统计方法、运行次数和算力 |
| 5 Results | 主结果与配对差；H1/H2；第 0 步梯度分析；低资源；H4；H3；稳健性（RTE 塌缩、GPU 影响） |
| 6 Discussion and Limitations | 四点讨论与假设结论，局限性 |
| 7 Conclusion | 结论 |
| Contributions、References | 成员贡献，参考文献 |

## 模板使用说明

- **CameraReady 版而不是匿名投稿版**：课程要求在首页列出所有成员的姓名、学号和 NTU 邮箱。
- **`\nocopyright`**：关闭首页底部的 AAAI 版权声明。这是课程报告，不是 AAAI 正式发表的论文。
- **`\setcounter{secnumdepth}{2}`**：开启章节编号，正文多处按节号互相引用。模板注释允许这样修改。
- **新增宏包**：只加了 `amsmath`、`amssymb`、`array`、`booktabs`，都不在模板列出的禁用清单里。没有使用任何禁用的宏包或命令。
- **篇幅**：课程要求 AAAI 格式、正文最多 6 页（不含参考文献）。本版正文占第 1–6 页，参考文献用 `\clearpage` 另起一页，在第 7 页。

## 相对初稿（`summary/REPORT_DRAFT.md`）的修改

**补充的内容**

- 首页的成员信息和代码仓库链接。
- §2.2 精确描述 EVA 的做法（课程要求精确描述被改进论文的方法）。
- §3.5 按参数成本分配的优化公式；§3.6 对照组 G2、G7、G8、G9 的实现细节。
- §4.5 运行次数和算力。
- 图 5（G4 − G3 秩差热力图）及其在 H3 一节的分析。
- 假设结论汇总表；原 TODO 改写为局限性。

**改正的内容**

1. GW-EVA 相对 EVA 的第 0 步梯度倍数：初稿的摘要和正文口径不一致，现统一用同种子比值 1.5–2.3×。
2. 运行次数：初稿写“约 230 次”。按结果文件实际计数，共享协议下 201 次，对照和补充种子 95 次，合计约 300 次。
3. 低资源差距：从“消失”改为“缩小到 0.6 分且不显著”。
4. 明确 EVA 实际使用中心化协方差；补充小批量恒等式成立需要 token 相互独立。
5. 参考文献：EVA 使用 NeurIPS 2025 的正式标题；AdaLoRA 补上漏掉的作者 Karampatziakis；新增 Ky Fan 定理、增量 SVD、Kaiming 初始化、Adam、AdamW。
6. G1 在 SST-2 上的标准差为 0.00：已核实三个种子的训练曲线不同，只是最终都对了 819/872，加脚注说明。
7. H4 的结论：从“已回答”改为“部分成立”（子空间依赖任务，但精度不受影响）。

## 第三版（6 页版）相对第二版的修改

第二版正文排到第 8 页。第三版只删减和改排，不改实验数字和结论：

- **表格并入正文或其他表**：数据驱动初始化对比表改为一段文字；学习率表并入实验组表（新增 lr 列）；共享协议表改为一段文字；H1 子空间表、第 0 步梯度比值表、H4 重合度表改为正文叙述（各任务均值、地板值和关键置信区间保留）；假设结论汇总表删去，结论保留在各节加粗句和讨论末句。
- **图**：删去 H4 逐层图 `h4_layers.pdf`（逐层结论保留在正文）；其余 4 张保留，H3 热力图缩到 0.82 倍页宽。
- **文字**：摘要、引言、背景、方法、讨论和结论压缩措辞；GW-EVA 五个步骤和对照组说明改为段落；贡献表改为段落。
- **小修正**：贡献部分去掉 `[LI YOURAN]` 的方括号和多余逗号，`Yin Chenyu` 统一为大写。

## 提交前待办

- [x] 在 `main.tex` 的 `\author{}`、`\affiliations{}` 和文末 Contributions 中，把 `[Name]`、`[Matric No.]`、`[email]` 替换为真实信息。
- [ ] 全组确认成员署名和贡献。
- [ ] 如果全组同意上面第 2、7 条修改，主 README 中的运行次数和 H4 结论也要同步修改。
