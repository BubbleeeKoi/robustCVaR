# Robust Risk-Sensitive RL with CVaR：Figure 1 复现说明

## 文件作用

本文件是项目总入口说明，负责告诉读者这个仓库复现哪篇论文、复现 Figure 1 的哪些内容、项目文件如何组织、依赖环境如何安装，以及完整复现实验应该按什么命令顺序运行。

本仓库复现以下论文中的 GridWorld 实验：

> Xinyi Ni and Lifeng Lai, "Robust Risk-Sensitive Reinforcement Learning with Conditional Value-at-Risk", IEEE ITW 2024.

复现目标是论文 Figure 1。代码尽量遵循论文公开的方法和参数，同时明确标注 PDF 中没有公开、必须由复现者自行假设的实验细节。

## 复现内容

论文研究有限 MDP 中的鲁棒 CVaR 优化，主要包括：

- 固定 ambiguity budget：鲁棒 CVaR 可以转换为调整置信水平后的标准 CVaR 或 EVaR 问题。
- 决策相关 Radon-Nikodym budget：通过 NCVaR 和增广状态 `(x, y)` 上的 value iteration 处理。
- `64 x 53` GridWorld：80 个障碍物，起点 `(60, 50)`，终点 `(60, 2)`，四邻域随机转移，障碍物碰撞代价为 `40`。

论文没有公开精确障碍物坐标、折扣因子、碰撞后动力学、停止准则、几何置信网格比例 `theta`、Figure 1d 的原始 `kappa(x,a)` 向量和绘图归一化方式。因此，本项目从 PDF 嵌入的 Figure 1 图像中抽取障碍物和路径数据，并在 `REPRO_STATUS.md` 中记录剩余假设。

## 项目结构

- `Robust_Risk-Sensitive_Reinforcement_Learning_with_Conditional_Value-at-Risk.pdf`：源论文 PDF。
- `robust_cvar_repro/gridworld.py`：GridWorld 环境、转移模型、从论文图像提取的障碍物坐标，以及假设的决策相关 `kappa(x,a)`。
- `robust_cvar_repro/value_iteration.py`：风险中性初始化、分段线性 CVaR/NCVaR value iteration、EVaR 对偶 KL 半径、风险状态更新和路径提取。
- `robust_cvar_repro/evaluation.py`：随机轨迹 Monte Carlo 评价，输出成本均值、VaR、CVaR、到达率和碰撞率。
- `tests/`：核心风险分配、EVaR 半径、warm start 和 Monte Carlo 指标的回归测试。
- `robust_cvar_repro/render.py`：生成 2x2 Figure 1 风格复现图。
- `scripts/extract_paper_figure_data.py`：必要时从 PDF 中抽取 Figure 1 嵌入图像，然后抽取障碍物和红色路径目标。
- `scripts/run_strict_gridworld.py`：严格复现主入口。
- `scripts/compare_paths_to_paper.py`：比较复现路径与 PDF 图像抽取路径。
- `outputs/pdf_images/`：从 PDF 第 5 页抽出的 Figure 1 四个面板图像。

## 环境

使用已有 Anaconda 环境：

```bash
conda activate portfolio
```

依赖列在 `requirements.txt` 中。当前代码使用 `numpy`、`Pillow`、`matplotlib` 和 `pymupdf`，不需要外部数据集。

## 复现步骤

所有命令都在仓库根目录运行。

1. 检查代码是否能编译。

```bash
python -m py_compile robust_cvar_repro/*.py scripts/*.py
python -m unittest discover -s tests -v
```

2. 从 PDF 抽取 Figure 1 参考数据。

```bash
python scripts/extract_paper_figure_data.py
```

预期输出：

- `outputs/paper_extracted_obstacles.csv`
- `outputs/paper_extracted_path_a_no_uncertainty.csv`
- `outputs/paper_extracted_path_b_rn_k_2.csv`
- `outputs/paper_extracted_path_c_kl_k_2.csv`
- `outputs/paper_extracted_path_d_unfix_kappa.csv`

该脚本会先确认 `outputs/pdf_images/page5_image*.png` 是否存在；如果不存在，会从 PDF 第 5 页抽取四张嵌入图像，然后把障碍物像素和红色路径像素映射到 `64 x 53` 网格。

3. 运行严格复现。

```bash
python scripts/run_strict_gridworld.py
```

预期输出：

- `outputs/paper_figure_obstacles.csv`
- `outputs/path_a_strict_cvar_no_uncertainty.csv`
- `outputs/path_b_strict_cvar_rn_k_2.csv`
- `outputs/path_c_strict_kl_evar_alpha_0_03.csv`
- `outputs/diagnostic_path_c_kl_kappa_2.csv`
- `outputs/path_d_strict_ncvar_decision_kappa.csv`
- `outputs/value_*.npy`
- `outputs/value_surface_*.npy`
- `outputs/policy_surface_*.npy`
- `outputs/strict_reproduction_gridworld.png`
- `outputs/strict_reproduction_gridworld_paper_style.png`
- `outputs/evaluation_*.json`
- `outputs/reproduction_manifest.json`

4. 比较复现路径和论文图像抽取路径。

```bash
python scripts/compare_paths_to_paper.py
```

比较脚本会输出路径点数量和 mean nearest Manhattan grid distance。这个结果用于复现诊断，不应被理解为逐像素图像复刻测试。

## Figure 1 面板对应关系

- Figure 1a：无模型不确定性的 CVaR，`alpha = 0.48`。
- Figure 1b：RN ambiguity，`K = 2`，实现为在 `alpha / K = 0.24` 处读取 CVaR 策略和值函数。
- Figure 1c：主四图恢复使用 calibrated KL 半径 `0.03`，因为它能重现论文红线的主要形状。严格按 Section III.B 计算的 KL 半径另存为 `outputs/diagnostic_*`，不再覆盖主图。主图 1c 是透明标注的视觉/行为复现，不是严格 EVaR 等价证明。
- Figure 1d：决策相关 RN ambiguity，`kappa(x,a) in [1, 2]`。代码已修复旧版把所有 `kappa>=1` 截成 1 的错误；对论文未定义的 `y*xi>1`，采用风险中性端点扩展并将策略风险状态投影到 1。论文没有公开完整 `kappa(x,a)`，当前局部碰撞风险假设仍不能恢复原图。

## 实用说明

`scripts/run_strict_gridworld.py` 支持 `--cold-start`、`--episodes` 和 `--reuse-evaluation`。默认允许用已有价值表面 warm start，但仍执行 Bellman 更新验证残差；科研最终运行应使用 `--cold-start`。
