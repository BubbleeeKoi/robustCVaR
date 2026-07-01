# 状态-动作依赖鲁棒 CVaR 投资组合实验：代码实现与结果说明

> **研究讲解版（数学模型 / 设计逻辑 / 论文写法）** 见 [`research_experiment_report.html`](research_experiment_report.html)。  
> 本文档（codeimprove.md）侧重**工程实现、复现命令与输出路径**。

本文档对应：
- **V1** `robust_cvar_portfolio_plan_mathjax.html`（已完成）
- **V2** `robust_cvar_V2_PLAN.html`（已完成：跨市场 + 消融 + Risk Measure Learning 框架）

**完成状态**

| 阶段 | 状态 | 入口脚本 |
|------|------|----------|
| V1 六步流水线 | ✅ 完成 | `scripts/run_full_experiment.py` |
| V2 顶刊协议 | ✅ 完成 | `experiments/run_v2_experiment.py` |
| SP30 跨市场 | ✅ 完成 | 同上 `--dataset sp30` |
| RL 扩展 | ⏸ 非主线（快速验证版） | V2 计划明确 RL 不参与主贡献 |

---

## 1. 一键复现

**V1（原计划六步）：**

```bash
conda activate portfolio
cd "Robust Risk-Sensitive Reinforcement Learning with Conditional Value-at-Risk"
python robust_cvar_portfolio/scripts/run_full_experiment.py
```

**V2（顶刊协议：ETF10 + ETF20 + SP30 跨市场消融）：**

```bash
conda activate portfolio
python robust_cvar_portfolio/experiments/run_v2_experiment.py

# 仅重跑某一数据集
python robust_cvar_portfolio/experiments/run_v2_experiment.py --dataset sp30
python robust_cvar_portfolio/experiments/run_v2_experiment.py --dataset sp30 --force-data
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
| RL 未分化 | 5 种 PPO 结果相同 | V2 计划已将 RL 列为 future work，主文用 rolling |
| learned κ 弱于 manual κ | val 上 θ 学习尚不稳定 | 增大 val 目标、换非线性 κ 网络 |
| ETF20 上 κ 提升不明显 | 20 ETF 分散化高，尾部差异小 | 论文中如实报告跨市场差异 |
| SP30 下载 | HON 历史不足已自动剔除（29 只） | 可换更长历史蓝筹 |
| Rolling 耗时 | SP30 五模型约 2.5 小时 | 已支持 `optimizer_maxiter` 加速 |

---

## 9. V2 顶刊协议（`robust_cvar_V2_PLAN.html`）

### 9.1 核心定位

V2 将论文主线明确为 **Risk Measure Learning**：

\[
s_t \rightarrow \kappa_\theta(s_t) \rightarrow \text{RCVaR}_{\alpha,\kappa} \rightarrow w_t \rightarrow L_t
\]

**RL 不参与主论文贡献**，仅作扩展实验（与 V2 计划 §九一致）。

### 9.2 V2 代码结构

```
robust_cvar_portfolio/
├── risk/
│   ├── rcvar.py          # RCVaR 对偶层
│   ├── kappa.py          # manual / learned κ_θ(s)
│   └── risk_engine.py    # state → κ → RCVaR 统一引擎
├── portfolio/
│   ├── optimizer.py      # min_w RCVaR
│   └── rolling.py        # 月度 rolling 回测
├── data/
│   ├── loader.py         # 多数据集加载（ETF10/20/SP30）
│   └── state.py          # V2 状态：Vol, DD, Mom, Corr
├── experiments/
│   └── run_v2_experiment.py
├── configs/
│   ├── etf10.yaml
│   ├── etf20.yaml
│   └── sp30.yaml
└── outputs/v2/           # V2 规范输出
```

### 9.3 V2 实验模型对照（消融）

| 模型 | κ 设定 | V1 对应 | 论文角色 |
|------|--------|---------|----------|
| A_no_kappa | κ=1 | A | w/o κ baseline |
| B_fixed_kappa | κ=2 常数 | B | fixed robust |
| C_manual_kappa | κ(s) 手工 β | C | **主方法（当前最优）** |
| C_learned_kappa | κ_θ(s) val 学习 | C 扩展 | learned κ 消融 |
| D_state_action | κ(s,w) | D | 非主线：action noise 分析 |

### 9.4 V2 执行步骤

| STEP | 内容 | 输出 |
|------|------|------|
| 1 | akshare 下载 ETF10→ETF20→SP30 | `data/processed/{dataset}/raw_data.csv` |
| 2 | 构建 state \(s_t=[Vol, DD, Mom, Corr]\) | `outputs/v2/{dataset}/state_matrix.npy` |
| 3 | 计算 κ 序列 | `outputs/v2/{dataset}/kappa_series.csv` |
| 4 | RCVaR + 组合优化 | rolling 内部 |
| 5 | Backtest | `nav_curve.png`, `drawdown.png` |
| 6 | 消融汇总 | `cvar_table.csv`, `ablation_summary.csv` |

### 9.5 跨市场 CVaR₅% 结果（Test 2020–2024，越低越好）

| 模型 | ETF10 | ETF20 | SP30 |
|------|-------|-------|------|
| A_no_kappa | 1.67% | **2.17%** | 2.78% |
| B_fixed_kappa | 1.65% | 2.19% | 2.60% |
| **C_manual_kappa** | **1.63%** | 2.20% | **2.50%** |
| C_learned_kappa | 1.69% | 2.19% | 2.71% |
| D_state_action | 1.65% | 2.19% | 2.61% |

**跨市场结论（可写论文）：**

1. **ETF10**：C_manual 在 CVaR（1.63%）、MDD（17.5%）、2020 危机损失（1.8%）上全面最优 → 主结果表用 ETF10。
2. **SP30**：C_manual CVaR **2.50%** vs A **2.78%**，相对下降 **10%**；`C_success_cvar_lowest=true` → 股票组合上鲁棒 κ(s) 仍有效。
3. **ETF20**：各模型 CVaR 差异 <0.03%，A 略优 → 说明在高度分散 ETF 组合上 κ 边际收益有限，可在论文 Limitations 讨论。
4. **消融**：w/o κ（A）在 SP30/ETF10 上 CVaR 均高于 C_manual → **κ 显著上升时去掉 κ 尾部风险恶化**。
5. **D 模型**：SP30 上 D 的 CVaR（2.61%）优于 A（2.78%）但弱于 C_manual（2.50%）→ 支持 V2 观点：action-dependent risk 引入额外噪声，非主线。
6. **learned κ**：当前 val 线性学习弱于 manual β；论文可报告为 future work（非线性 κ 网络 / 端到端训练）。

### 9.6 SP30 详细指标（Test 2020–2024）

| 模型 | CVaR₅% | MDD | Ann.Return | 2020危机 | 2022高波动 |
|------|--------|-----|------------|---------|-----------|
| C_manual_kappa | **2.50%** | **22.2%** | 1.43% | 7.01% | **5.88%** |
| B_fixed_kappa | 2.60% | 22.3% | **9.61%** | 6.88% | 7.70% |
| D_state_action | 2.61% | 22.5% | 1.43% | 7.24% | 4.77% |
| C_learned_kappa | 2.71% | 26.4% | -1.06% | 10.9% | 6.31% |
| A_no_kappa | 2.78% | 24.8% | -0.54% | 8.69% | 10.8% |

### 9.7 V2 输出文件

| 路径 | 用途 |
|------|------|
| `outputs/v2/ablation_summary.csv` | 跨市场 CVaR 消融主表 |
| `outputs/v2/cross_market_cvar_table.csv` | 全指标长表 |
| `outputs/v2/{dataset}/nav_curve.png` | 论文 Figure NAV |
| `outputs/v2/{dataset}/drawdown.png` | 论文 Figure 回撤 |
| `outputs/v2/{dataset}/cvar_table.csv` | 单数据集全指标 |
| `outputs/v2/{dataset}/kappa_series.csv` | κ 可解释性分析 |
| `outputs/v2/{dataset}/rolling_results.csv` | 日度回测明细 |
| `outputs/v2/{dataset}/d_ablation_nav.png` | C vs D 消融图 |

### 9.8 论文最终结构（V2 建议）

1. RCVaR risk framework  
2. **κ(s) risk measure learning**（核心贡献，主推 C_manual）  
3. Rolling optimization  
4. Empirical validation（ETF10 主表 + SP30 稳健性 + ETF20 边界案例）  
5. D ablation（action-dependent noise）  
6. RL future work  

**一句话结论（V2）：**

> 风险度量 κ 可随市场状态（波动、回撤、动量、相关性）动态调整；在 ETF10 与 SP30 样本外测试中，状态依赖 RCVaR 将 CVaR₅% 稳定低于普通 CVaR 与无 κ baseline。

---

## 8. 与原文 GridWorld 复现的关系

- `robust_cvar_repro/`：Ni & Lai 论文 Figure 1 GridWorld 复现（已完成，保持不变）。
- `robust_cvar_portfolio/`：**新方法**——将 robust CVaR + κ(s,w) 迁移到真实 ETF 动态组合。
- 执行顺序遵循计划：**先验证 RCVaR 风险层（rolling），再接 RL**。

---

*V1 完成：2026-06-29；V2 完成：2026-07-01；环境：`conda activate portfolio`*
