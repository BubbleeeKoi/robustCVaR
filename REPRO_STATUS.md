# 复现状态记录

## 文件作用

本文件记录复现状态和边界：哪些参数来自 PDF 原文，哪些数据从 Figure 1 图像恢复，哪些实现细节属于必要假设，以及当前严格复现输出与论文图像抽取路径之间的比较结果。

本项目对论文 Figure 1 GridWorld 实验做透明复现，不伪造论文未公开的作者原始数据。

## PDF 明确公开的信息

- 论文题目：`Robust Risk-Sensitive Reinforcement Learning with Conditional Value-at-Risk`
- 作者：Xinyi Ni 和 Lifeng Lai
- PDF 元数据会议：IEEE ITW 2024
- PDF 元数据 DOI：`10.1109/ITW61385.2024.10806953`
- 网格大小：`64 x 53`
- 起点：`(60, 50)`
- 终点：`(60, 2)`
- 动作：east、south、west、north
- 目标方向转移概率：`0.95`
- 其他三个相邻方向概率：`0.05 / 3`
- 障碍物数量：`80`
- 安全移动代价：`1`
- 障碍物碰撞代价：`40`
- CVaR 置信水平：`alpha = 0.48`
- Figure 1b-c 的 RN 和 KL budget：`K = 2`
- RN 等价置信水平：`alpha'_CVaR = 0.24`
- KL/EVaR 等价置信水平：`alpha'_EVaR = 0.03`
- 插值采样点数量：`21`
- 几何网格规则：`y_{i+1} = theta * y_i`
- Figure 1d 的决策相关 budget 范围：`[1, 2]`

## 从 Figure 1 图像恢复的信息

PDF 没有以数据文件形式公开障碍物坐标或路径坐标。因此，本项目从 PDF 第 5 页的四张 Figure 1 嵌入图像中抽取近似目标。`scripts/extract_paper_figure_data.py` 可以在缺少图像文件时，用 PyMuPDF 直接从 PDF 生成 `outputs/pdf_images/page5_image*.png`。

生成的图像抽取产物：

- `outputs/paper_extracted_obstacles.csv`
- `outputs/paper_extracted_path_a_no_uncertainty.csv`
- `outputs/paper_extracted_path_b_rn_k_2.csv`
- `outputs/paper_extracted_path_c_kl_k_2.csv`
- `outputs/paper_extracted_path_d_unfix_kappa.csv`

抽取出的障碍物坐标也已经固化在 `robust_cvar_repro/gridworld.py` 的 `PAPER_FIGURE_OBSTACLES` 中，作为默认环境布局。

## 必须显式说明的假设

以下细节论文没有公开，无法在没有作者代码或原始实验文件的情况下精确恢复：

- 折扣因子 `gamma`；当前代码使用 `0.95`。
- 碰撞后的动力学；当前代码把障碍物格子视为终止状态。
- 边界行为；当前代码把撞墙处理为留在原地并产生碰撞代价。
- 初始值函数细节；当前代码使用风险中性 value iteration 初始化 CVaR 表面。
- 作者生成论文图时使用的精确停止准则。
- 几何比例 `theta`；论文只说明 `y_{i+1}=theta*y_i` 和 21 个采样点，没有公开具体 `theta`，当前代码使用 `2.067`。
- KL/EVaR 的原始实现细节；当前代码按 Section III.B 使用 `kappa=2`，得到等价 `alpha'_EVaR ≈ 0.113` 与 `KL 半径 ≈ 2.18`。论文图注写的 `0.03` 与 Section III.B 公式在 `kappa=2` 时不自洽，因此不再直接把 `-log(0.03)` 当作半径。
- Figure 1d 的完整决策相关 `kappa(x,a)` 向量。
- 作者原始绘图时使用的色条归一化、路径渲染和坐标到像素映射方式。

当前 Figure 1d 使用：

```text
kappa(x,a) = 1 + normalized_expected_obstacle_collision_risk(x,a)
```

该假设满足论文给出的 `[1,2]` 范围，并且具有可解释性，但不是论文未公开的原始 `kappa(x,a)`。

## 当前代码范围

保留的核心脚本：

- `scripts/extract_paper_figure_data.py`
- `scripts/run_strict_gridworld.py`
- `scripts/compare_paths_to_paper.py`

已删除的冗余内容：

- 旧的单 alpha 近似复现入口。
- 旧的 Figure 1 视觉校准复现入口。
- 已被 `solve_robust_cvar_pwl` 取代的旧 CVaR 辅助实现。
- Python 和 matplotlib 缓存文件。
- 旧近似/校准脚本生成的输出。

## 验证命令

在仓库根目录运行：

```bash
python -m py_compile robust_cvar_repro/*.py scripts/*.py
python -m unittest discover -s tests -v
python scripts/extract_paper_figure_data.py
python scripts/run_strict_gridworld.py
python scripts/compare_paths_to_paper.py
```

严格求解器较慢，因为 Algorithm 2 风格的 CVaR/NCVaR value iteration 会遍历状态、置信水平、动作和后继状态。

## 最近一次路径比较记录

2026-06-25 将 Figure 1c 主图与公式诊断拆分后，主图比较结果如下：

- 无不确定性：`paper_points=59`，`reproduced_points=53`，`mean_nearest_grid_distance=0.528`
- RN K=2：`paper_points=98`，`reproduced_points=91`，`mean_nearest_grid_distance=1.505`
- KL calibrated radius=0.03：`paper_points=102`，`reproduced_points=97`，`mean_nearest_grid_distance=1.722`
- 决策相关 kappa：`paper_points=118`，`reproduced_points=53`，`mean_nearest_grid_distance=13.019`

Figure 1c 主图使用半径 `0.03`，恢复了此前的中部绕行路径。Section III.B 在 `kappa=2` 下给出的半径约 `2.18`，结果仅有 16 个路径点，已单独保存为 `diagnostic_path_c_kl_kappa_2.csv`，用于说明论文图注、正文公式与实际图之间存在未公开实现差异。
