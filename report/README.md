# report/ — 项目报告（AAAI-24 模板）

本文件夹是 AI6103 小组项目报告的第一版完整稿。它用 AAAI-24 Author Kit 中的 CameraReady LaTeX 模板排版，在 main 分支的 `summary/REPORT_DRAFT.md` 基础上补全而成。

报告里的所有数字都来自 main 分支的 `summary/tables.md`，该文件由 `summary/collect.py` 从各组的运行结果 JSON 生成。我写完后把正文里的小数逐一自动比对过，全部能在这些文件中找到。

## 文件

| 文件 | 作用 |
|---|---|
| `REPORT_AAAI.pdf` | 编译好的报告：双栏，共 9 页（正文约 8 页，参考文献约 1 页） |
| `main.tex` | LaTeX 源文件，单文件（AAAI 不允许用 `\input` 拆分正文） |
| `refs.bib` | 参考文献，14 条，作者、标题和会议已逐条核对 |
| `aaai24.sty`、`aaai24.bst` | AAAI-24 模板的样式文件和参考文献格式，原样取自 AuthorKit24/CameraReady，没有改动 |
| `figures/` | 报告用到的 5 张图，均为矢量 PDF |

`figures/` 中每张图对应报告中的编号和生成脚本（脚本路径均相对于 main 分支的仓库根目录）：

| 文件 | 报告中的位置 | 生成脚本 |
|---|---|---|
| `h1_overlap.pdf` | 图 1：H1，EVA 与 GW-EVA 的子空间重合度 | `summary/figures.py` |
| `step0_vs_accuracy.pdf` | 图 2：第 0 步梯度与精度 | `summary/figures.py` |
| `lowres_lr.pdf` | 图 3：低资源下精度与学习率的关系 | `summary/figures.py` |
| `h4_layers.pdf` | 图 4：H4，按层类型和深度的子空间重合度 | `summary/figures.py` |
| `g34_rank_heatmap_diff.pdf` | 图 5：H3，G4 − G3 逐层秩差 | `G3G4/jobs/plot_ranks.py` |

图 1–4 的编号与主 README 一致。

## 编译

**Overleaf**：把本文件夹的全部文件（含 `figures/`）上传到一个新项目，编译器选 pdfLaTeX，点 Recompile 即可。

**本机**（需要 TeX Live，包含 times、helvet、courier、natbib、caption、amsmath、amssymb、booktabs）：

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
| 4 Experimental Design | 9 个实验组、假设 H1–H4、共享协议、选定的学习率、统计方法、运行次数和算力 |
| 5 Results | 主结果与配对差；H1/H2；第 0 步梯度分析；低资源；H4；H3；稳健性（RTE 塌缩、GPU 影响） |
| 6 Discussion and Limitations | 假设结论汇总表，7 条局限性 |
| 7 Conclusion | 结论 |
| Contributions、References | 成员贡献，参考文献 |

## 模板使用说明

- **CameraReady 版而不是匿名投稿版**：课程要求在首页列出所有成员的姓名、学号和 NTU 邮箱。
- **`\nocopyright`**：关闭首页底部的 AAAI 版权声明。这是课程报告，不是 AAAI 正式发表的论文。
- **`\setcounter{secnumdepth}{2}`**：开启章节编号，正文多处按节号互相引用。模板注释允许这样修改。
- **新增宏包**：只加了 `amsmath`、`amssymb`、`booktabs`，都不在模板列出的禁用清单里。没有使用任何禁用的宏包或命令。
- **篇幅**：AAAI 正式论文的限制是正文 7 页加参考文献，本报告正文约 8 页。课程要求里没有页数限制，所以没有压缩。

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

## 提交前待办

- [ ] 在 `main.tex` 的 `\author{}`、`\affiliations{}` 和文末 Contributions 表中，把 `[Name]`、`[Matric No.]`、`[email]` 替换为真实信息。
- [ ] 全组确认 Contributions 表。
- [ ] 请 G5/G6 负责人核对 §3.4 GW-EVA 第 3 步的表述，并确认 G5/G6 使用的 GPU 型号。
- [ ] 如果全组同意上面第 2、7 条修改，主 README 中的运行次数和 H4 结论也要同步修改。
