# 状态-动作依赖鲁棒 CVaR 投资组合实验：代码实现与结果说明

本文档对应 `robust_cvar_portfolio_plan_mathjax.html` 中的执行计划，说明**新增了哪些代码、每一步实验做什么、逻辑是什么、如何复现、以及可直接写入论文的结果**。

---

## 1. 一键复现

```bash
conda activate portfolio
cd "Robust Risk-Sensitive Reinforcement Learning with Conditional Value-at-Risk"
python robust_cvar_portfolio/scripts/run_full_experiment.py
```

环境依赖（已在 `portfolio` 环境中安装）：

```bash
pip install akshare numpy pandas scipy matplotlib pyyaml torch
```

---

## 2. 新增代码结构

```
robust_cvar_portfolio/
├── configs/etf10.yaml              # ETF10 数据与实验超参
├── data/processed/                 # akshare 下载后的价格/收益
├── src/
│   ├── data_loader.py              # akshare 美股 ETF 数据管线
│   ├── features.py                 # 市场状态 z_t 与 κ(s,w)
│   ├── risk_metrics.py             # VaR/CVaR/Sharpe/MDD 等
│   ├── robust_cvar_layer.py        # RCVaR 对偶风险层
│   ├── rolling_rcvar.py            # 非 RL rolling 组合（A/B/C/D）
│   ├── baselines.py                # 等权、最小方差
│   ├── env_portfolio.py            # PortfolioEnv（RL 环境）
│   ├── backtest.py                 # 指标汇总与作图
│   ├── evaluation.py               # 最终论文表合并
│   └── agents/ppo.py               # PPO / CVaR-PPO / Robust CVaR-PPO
├── scripts/run_full_experiment.py  # 六步流水线主入口
├── tests/                          # RCVaR 退化关系单元测试
└── outputs/                        # 全部实验输出
```

---

## 3. 实验流水线（六步）

### Step 1：数据管线（akshare）

| 项目 | 内容 |
|------|------|
| **做什么** | 下载 10 只 US ETF 日频收盘价，计算日收益，划分 train/val/test |
| **逻辑** | 使用 `ak.stock_us_daily(symbol=...)` 拉取 SPY/QQQ/IWM/TLT/GLD/XLF/XLK/XLE/XLV/XLU；`returns = prices.pct_change()`；按 yaml 切分 |
| **时间划分** | train: 2010–2017；val: 2018–2019；test: 2020–2024 |
| **输出** | `data/processed/prices.csv`, `returns.csv`, `splits.json`, `dataset_summary.csv` |

**实际样本量**（有效交易日，2016-02-02 起 10 标的齐整）：

| 划分 | 天数 | 资产数 |
|------|------|--------|
| train | 482 | 10 |
| val | 503 | 10 |
| test | 1258 | 10 |

---

### Step 2：Robust CVaR 风险层验证

| 项目 | 内容 |
|------|------|
| **做什么** | 验证 RCVaR 退化关系是否正确 |
| **逻辑** | 对损失样本 \(Z\)，求解 \(\max_{q\in\mathcal Q_{\alpha,\kappa}}\sum q_i Z_i\)，约束 \(0\le q_i\le \kappa_i/(\alpha B)\)，\(\sum q_i=1\)；按损失从大到小贪心分配权重 |
| **退化检查** | \(\kappa=1\) ≈ 普通 CVaR；\(\kappa=K=2\) > 普通 CVaR；高损失样本 \(\kappa\) 增大时 RCVaR 增大 |

**单元测试结果**（252 日等权组合损失）：

| 指标 | 数值 |
|------|------|
| plain CVaR\(_{5\%}\) | 0.01267 |
| RCVaR (\(\kappa=1\)) | 0.01282 |
| RCVaR (\(\kappa=2\)) | 0.01656 |

输出：`outputs/tables/rcvar_degeneracy_check.json`

---

### Step 3：非 RL Rolling Robust CVaR 组合（核心验证）

| 项目 | 内容 |
|------|------|
| **做什么** | 在 test 期每月调仓，用过去 M=252 日历史损失优化权重 |
| **优化目标** | \(\min_w \text{RCVaR}_{\alpha,\kappa(s,w)}\big(\{-w^\top r_\tau + c\|w-w_{t-1}\|_1\}_{\tau=t-M}^{t-1}\big)\) |
| **约束** | \(w\in\Delta^{10}\)（softmax 参数化） |
| **交易成本** | \(c=0.001\)，仅在调仓日收取 |

**四个版本（消融）**：

| 版本 | κ 设定 | 含义 |
|------|--------|------|
| A | κ=1 | 普通 CVaR 组合 |
| B | κ=2（固定） | Fixed Robust CVaR |
| C | κ(s)=1+κ_max·σ(β₁Vol_z+β₂DD_z) | 状态依赖鲁棒 |
| D | κ(s,w)=C + β₃·Conc(w) | 状态-动作依赖鲁棒 |

**κ 设计**（第一版，κ_max=1，κ∈[1,2]）：

\[
\kappa(s,w)=1+\kappa_{\max}\cdot\sigma\big(\beta_1\widetilde{Vol}+\beta_2\widetilde{DD}+\beta_3\widetilde{Conc}\big)
\]

输出：`outputs/rolling_portfolio/rolling_metrics.csv`、NAV/回撤对比图、各策略权重序列。

---

### Step 4：传统 Baseline

| 策略 | 逻辑 |
|------|------|
| Equal Weight | 始终 1/10 等权 |
| Min Variance | 每月末用 252 日协方差求最小方差权重 |

在 **test 期（2020–2024）** 与 rolling 方法对比。

---

### Step 5：PortfolioEnv + RL（PPO 系列）

| 项目 | 内容 |
|------|------|
| **状态** | \(s_t = (R_{t-L:t}, z_t, w_{t-1})\)，L=20 |
| **动作** | logits → softmax → 组合权重 |
| **奖励** | 净收益 \(R^{net}_{t+1}=-L_{t+1}\) |
| **训练** | 2010–2017 训练，2020–2024 测试 |
| **五种 RL** | PPO / CVaR-PPO / Fixed-Robust-PPO / State-Robust-PPO / SAD-Robust-PPO |

RL 目标在 PPO 损失中加入 RCVaR 项（见 `agents/ppo.py` 中 `_objective_loss`）。

> **说明**：当前 RL 仅训练 30 iter × 64 steps（快速验证版），五种策略测试结果相同，说明策略尚未充分分化；**论文主结论应基于 Step 3 rolling 结果**。后续可增大 `train_iters`、调学习率再做 RL 对比。

---

### Step 6：论文汇总表

合并 rolling + baseline + RL，按 **CVaR\(_{5\%}\)**（样本外净损失）排序。

输出：

- `outputs/tables/final_paper_metrics.csv`
- `outputs/tables/final_summary.json`

---

## 4. 可直接写入论文的结果（Test 2020–2024）

**主指标：样本外 CVaR\(_{5\%}\)（净损失，越低越好）**

| 方法 | Ann.Return | Ann.Vol | Sharpe | MDD | VaR\(_{5\%}\) | **CVaR\(_{5\%}\)** | Calmar | 2020危机损失 | 2022高波动损失 |
|------|-----------|---------|--------|-----|-------------|-------------------|--------|-------------|---------------|
| **C 状态依赖 RCVaR（rolling）** | 3.84% | 11.25% | 0.342 | **18.31%** | 1.10% | **1.63%** | 0.210 | **2.27%** | 9.11% |
| B 固定 κ=2 RCVaR | 4.26% | 11.41% | 0.373 | 18.64% | 1.12% | 1.65% | 0.228 | 3.02% | 9.45% |
| A 普通 CVaR | 2.68% | 11.60% | 0.231 | 20.00% | 1.09% | 1.67% | 0.134 | 4.19% | 11.0% |
| D 状态-动作 RCVaR | **4.59%** | 11.58% | **0.397** | 19.86% | 1.13% | 1.68% | **0.231** | 3.84% | 9.14% |
| Min Variance | 8.85% | 13.76% | 0.643 | 22.18% | 1.20% | 2.06% | 0.399 | — | — |
| Equal Weight | 10.12% | 17.51% | 0.578 | 30.29% | 1.54% | 2.60% | 0.334 | — | — |

### 4.1 核心结论（可写进 Abstract / Results）

1. **RCVaR 风险层有效**：四种 rolling RCVaR 策略的 CVaR\(_{5\%}\)（1.63%–1.68%）均显著低于 Equal Weight（2.60%）与 Min Variance（2.06%），符合“降低尾部风险”的研究目标。
2. **状态依赖 κ(s) 在尾部风险上最优**：C 策略 CVaR\(_{5\%}\)=**1.63%**，为 rolling 系列最低；MDD **18.31%** 也是 rolling 中最低；2020 COVID 危机期损失仅 **2.27%**（A 为 4.19%）。
3. **Fixed robust κ=K** 在收益-风险折中上表现良好：B 策略年化 4.26%、Sharpe 0.37，CVaR 1.65%，优于 A。
4. **状态-动作 κ(s,w)** 提高了收益（4.59%）和 Sharpe（0.40），但本次实验中 CVaR 略高于 C（1.68% vs 1.63%）；`success_check.D_cvar_lt_A = false`。论文中可如实报告，并讨论 Conc 项权重 β₃ 需进一步调参。
5. **Rolling 优于当前 RL 初版**：RL 快速训练版 CVaR=2.75%，尚未超过 rolling；论文主线建议以 **Step 3 非 RL rolling 验证** 作为 RCVaR 有效性的主要证据。

### 4.2 论文表述建议

> 在 10 ETF、2020–2024 样本外测试中，状态依赖鲁棒 CVaR 组合（κ(s)）将 CVaR\(_{5\%}\) 从普通 CVaR 的 1.67% 降至 **1.63%**，最大回撤从 20.0% 降至 **18.3%**，COVID 危机期损失从 4.2% 降至 **2.3%**，而年化收益由 2.7% 提升至 3.8%。

---

## 5. 数学逻辑对照（与计划文档一致）

| 概念 | 实现位置 |
|------|----------|
| 净损失 \(L_{t+1}=-w_t^\top r_{t+1}+c\|w_t-w_{t-1}\|_1\) | `risk_metrics.net_portfolio_loss` |
| 普通 CVaR | `risk_metrics.cvar_alpha` |
| RCVaR 对偶形式 | `robust_cvar_layer.robust_cvar` |
| κ(s,w) | `features.kappa_state_action` |
| Rolling 优化 | `rolling_rcvar.optimize_weights` |
| 退化 κ=1 → CVaR | `tests/test_robust_cvar_layer.py` |

---

## 6. 输出文件索引

| 文件 | 用途 |
|------|------|
| `outputs/rolling_portfolio/rolling_metrics.csv` | Rolling A/B/C/D 全指标 |
| `outputs/rolling_portfolio/rolling_nav_comparison.png` | NAV 对比图（论文 Figure） |
| `outputs/rolling_portfolio/rolling_drawdown_comparison.png` | 回撤对比图 |
| `outputs/tables/baseline_metrics.csv` | 传统 baseline |
| `outputs/tables/final_paper_metrics.csv` | **论文主表** |
| `outputs/tables/final_summary.json` | 最优策略摘要 |
| `outputs/tables/rl_metrics.csv` | RL 快速验证结果 |

---

## 7. 已知限制与后续改进

| 问题 | 说明 | 建议 |
|------|------|------|
| RL 未分化 | 5 种 PPO 结果相同 | 增大 `train_iters`（200+）、加 entropy bonus、独立随机种子 |
| D 未在 CVaR 上超过 C | β₃、κ_max 未调参 | 在 val 集上网格搜索 β₁–β₃ |
| 数据源 | akshare 美股接口，有效样本从 2016 起 | 可换更长历史数据源或 A 股 ETF |
| Rolling 耗时 | 四策略全量约 9 分钟 | 已实现结果缓存，重复运行跳过 Step 3 |
| 计划中的 SP100-30 / 100 资产 | 尚未扩展 | 复制 `etf10.yaml` 改 ticker 列表即可 |

---

## 8. 与原文 GridWorld 复现的关系

- `robust_cvar_repro/`：Ni & Lai 论文 Figure 1 GridWorld 复现（已完成，保持不变）。
- `robust_cvar_portfolio/`：**新方法**——将 robust CVaR + κ(s,w) 迁移到真实 ETF 动态组合。
- 执行顺序遵循计划：**先验证 RCVaR 风险层（rolling），再接 RL**。

---

*实验完成时间：2026-06-29；运行环境：`conda activate portfolio`；主脚本：`robust_cvar_portfolio/scripts/run_full_experiment.py`*
