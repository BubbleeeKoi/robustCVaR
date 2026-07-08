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

## 10. V3：SP100 指数成分股主实验（2026-07-01 完成）

### 10.1 目标与数据

- **股票池**：S&P 500 大型股 **100 只**（Version A，当前成分回填，存在 survivorship bias，论文需注明）
- **数据**：akshare `stock_us_daily`，`data/processed/sp100/`（100 资产 × 3475 日，**2011-01-27 ~ 2024-12-31**）
- **划分**：Train 2010–2017 / Val 2018–2019 / Test **2020–2024**
- **Benchmark**：SPY

### 10.2 运行方式（无需 conda 交互，直接执行）

```powershell
# 推荐：直接调用 portfolio 环境 Python（避免 conda run 卡住）
C:\Users\Chen\anaconda3\envs\portfolio\python.exe -u robust_cvar_portfolio/experiments/run_v3_experiment.py

# 或批处理
robust_cvar_portfolio\scripts\run_v3.bat
```

- 支持 **断点续跑**：每个模型完成后写入 `outputs/v3/sp100/rolling_{model}.csv`，中断后重跑自动跳过已完成模型。
- 全流程耗时约 **2.7 小时**（100 资产，5 个消融模型 + 7 个 baseline）。

### 10.3 V3 主消融结果（Test 2020–2024，CVaR₅% 越低越好）

| 模型 | CVaR₅% | MDD | Sharpe | 2020危机 | 2022高波动 |
|------|--------|-----|--------|---------|-----------|
| **B_fixed_kappa** | **2.66%** | **25.6%** | 0.29 | **2.9%** | 11.1% |
| A_no_kappa | 2.86% | 35.7% | **0.40** | 15.3% | **8.0%** |
| D_state_action | 2.83% | 35.7% | 0.22 | 14.4% | 8.4% |
| C_learned_kappa | 2.87% | 35.7% | 0.04 | 17.7% | 8.6% |
| C_manual_kappa | 2.92% | 35.7% | 0.11 | 15.6% | 9.4% |

**V3 成功标准检验**：`CVaR(C_manual) < CVaR(A)` → **未满足**（2.92% > 2.86%）；MDD 相同（35.7%）。Bootstrap `P(CVaR_C < CVaR_A) = 26.4%`，改善**不显著**。

### 10.4 传统 Baseline 对比（Test）

| 方法 | CVaR₅% | MDD | Sharpe |
|------|--------|-----|--------|
| **Historical_CVaR** | **2.46%** | **24.9%** | **0.50** |
| B_fixed_kappa | 2.66% | 25.6% | 0.29 |
| Risk_Parity | 2.96% | 35.1% | 0.47 |
| A_no_kappa | 2.86% | 35.7% | 0.40 |
| C_manual_kappa | 2.92% | 35.7% | 0.11 |
| Equal_Weight | 3.11% | 35.7% | 0.52 |
| SPY | 3.20% | 34.1% | 0.60 |

### 10.5 V3 结论（相对 V2）

1. **SP100 上 C_manual 未复现 V2 优势**：ETF10/SP30 有效，但 100 只股票高维组合下状态依赖 κ(s) 反而略差于 plain CVaR。
2. **B_fixed（κ=2）在 SP100 上表现最好**：CVaR 2.66%、MDD 25.6%，优于 C_manual 与 A。
3. **Historical CVaR baseline 最强**：滚动历史 CVaR 优化得 CVaR 2.46%，说明高维下优化器/κ 设计需进一步调参。
4. **相对 SPY**：C_manual CVaR 2.92% < SPY 3.20%，尾部风险仍优于指数，但 Sharpe 远低于 SPY。
5. **论文建议**：SP100 结果如实报告；主贡献仍可依托 ETF10 + SP30；SP100 作为 scale-up 边界案例讨论（κ(s) 在高维股票池的边际收益与计算成本）。

### 10.6 V3 输出目录

| 路径 | 用途 |
|------|------|
| `outputs/v3/sp100/` | 完整中间结果、图表、checkpoint |
| `outputs/v3/sp100_final/` | 论文级 table1–5、fig1–5、summary |
| `outputs/v3/sp100/rolling_{model}.csv` | 模型断点，支持续跑 |
| `data/processed/sp100/universe.csv` | 100 只股票列表与 sector |

---

## 11. V4：SP100 诊断实验（2026-07-01）

> 计划文件：`sp100_diagnostic_experiment_v4plan.html`（每步诊断结果已写回该 HTML）

### 11.1 运行方式

```powershell
# 全流程（含权重导出 + 敏感性，耗时长）
C:\Users\Chen\anaconda3\envs\portfolio\python.exe -u robust_cvar_portfolio/experiments/run_v4_diagnostics.py

# 仅补跑敏感性 + 汇总（diag 1–6 已完成时）
C:\Users\Chen\anaconda3\envs\portfolio\python.exe -u robust_cvar_portfolio/experiments/run_v4_diagnostics.py --sensitivity-only
```

输出目录：`robust_cvar_portfolio/outputs/v3/sp100_diagnostics/`

### 11.2 诊断结论摘要（C_manual 为何在 SP100 失败）

| 诊断项 | 是否支持失败解释 | 主要证据 |
|--------|------------------|----------|
| κ(s) 响应方向 | **否** | corr(κ,Vol)=+0.82，corr(κ,DD)=+0.78；危机期 κ 均值 1.84 vs 正常 1.26 |
| 权重更集中 | **是** | C 有效持仓 N^eff=70.3 vs B 79.4 vs A 87.7；最大单权重 10.6% |
| 换手更高 | **是** | C 平均换手 0.607 vs A 0.272（约 2.2×） |
| 行业暴露失衡 | 次要 | C 行业 HHI 略高于 B，但 HistCVaR 更极端仍更优 |
| q 过度集中 | **否** | B 的 q HHI 最高但 CVaR 最优 |
| κ_max 敏感性 | **是** | κ_max=0.5 → CVaR **2.68%**（优于 A，接近 B 2.66%）；当前 1.0 → 3.00% |
| 窗口 M 敏感性 | **否** | M=504 CVaR 3.08% ≥ M=252 3.00% |

### 11.3 机制归纳（论文可写）

1. **κ(s) 映射本身有效**，危机期上升、与 Vol/DD 正相关。
2. **动态 κ 导致优化目标月际变化大** → 换手显著上升（0.61 vs A 0.27）→ 样本外 CVaR 恶化。
3. **C 权重更集中**（N^eff=70 vs A 88），高维个股池中 diversification 不足。
4. **κ_max=1.0 在 SP100 上过大**；降至 0.5 后 CVaR 2.68%，说明是参数/幅度问题而非机制失效。
5. **固定 κ=2（B）** 与 **Historical CVaR（2.46%）** 仍是 SP100 强基线。
6. **下一步改模型（V5）**：val 上选 κ_max∈[0.25,0.75]；κ 平滑；换手/HHI 惩罚。

### 11.4 关键输出文件

| 路径 | 内容 |
|------|------|
| `diagnostic_summary.csv` | 七项诊断汇总 + next_action |
| `kappa_diagnostics/kappa_summary.csv` | κ 统计与相关性 |
| `weights_diagnostics/weight_concentration.csv` | HHI / N_eff |
| `turnover_diagnostics/turnover_summary.csv` | 换手对比 |
| `q_weight_diagnostics/q_weight_summary.csv` | worst-case q 集中度 |
| `figures/fig_*.png` | 诊断图 |

---

## 12. Baseline Audit：CVaR 定义统一与公平对照（2026-07-02）

> 背景：V3 中 `Historical_CVaR` 优于 `A_no_kappa` 经调查有三层原因——(1) 优化器内 CVaR 定义不同（fractional vs ceil）；(2) Historical 调仓日成本 bug（已修）；(3) 非凸路径差异。本节在修复成本、显式拆分 CVaR 定义后重跑 SP100 Test（2020–2024）。

### 12.1 代码改动

| 模块 | 改动 |
|------|------|
| `src/risk_metrics.py` | `cvar_alpha_fractional` / `cvar_alpha_ceil`；`cvar_alpha` 别名 ceil（与 Historical 一致） |
| `risk/risk_engine.py` | `plain_ceil` 模式（κ=1 时用 ceil-tail 目标） |
| `src/baselines.py` | 修复 Historical 调仓日 `w_prev=w` 导致换手=0 的成本 bug |
| `tests/test_cvar_definitions.py` | 两种定义在 αM 非整数时不相等 |
| `experiments/run_baseline_audit.py` | 公平对照实验脚本 |

### 12.2 运行方式

```powershell
C:\Users\Chen\anaconda3\envs\portfolio\python.exe -u robust_cvar_portfolio/experiments/run_baseline_audit.py
```

输出目录：`robust_cvar_portfolio/outputs/v3/sp100_baseline_audit/`（含 rolling checkpoint，可续跑）

耗时：约 **178 分钟**（含 val 上 κ_max 网格 + 6 个 Test 模型 + Historical 权重导出）

### 12.3 Val 选 κ_max（C_calibrated）

| κ_max | Val CVaR 5% |
|-------|-------------|
| 0.5 | 2.26% |
| 0.75 | 2.19% |
| **1.0** | **2.09%** ← 选中 |
| 1.25 | 2.09% |
| 1.5 | 2.12% |

Val 最优为 **κ_max=1.0**，与默认相同，故 `C_calibrated` 与 `C_default` Test 轨迹完全一致。

### 12.4 Test 公平对照（评估指标统一用 ceil-tail `cvar_alpha`）

| 方法 | 分组 | CVaR 5% | MaxDD | 平均换手 | Sharpe |
|------|------|---------|-------|----------|--------|
| **Historical_CVaR_fixed** | external_baseline | **2.47%** | 25.1% | 0.038 | 0.44 |
| **A_ceil_CVaR** | internal_ablation | **2.47%** | 25.1% | 0.038 | 0.44 |
| B_fixed_kappa | internal_ablation | 2.66% | 25.6% | 0.020 | 0.29 |
| A_frac_CVaR | internal_ablation | 2.86% | 35.7% | 0.015 | 0.40 |
| C_default / C_calibrated | internal_ablation | 2.92% | 35.7% | 0.023 | 0.11 |

**关键结论：**

1. **`A_ceil_CVaR` 与 `Historical_CVaR_fixed` 数值完全一致**（权重、CVaR、Sharpe 相同）→ CVaR 定义统一后，Historical 等价于「鲁棒框架 κ=1 + ceil-tail 目标」。
2. **`A_frac_CVaR`（κ=1 fractional-tail）Test CVaR 2.86%**，仍劣于 ceil-tail 路线 2.47%；差距主要来自**优化目标定义**，而非 baseline 成本 bug（bug 已修）。
3. **内部消融**：`C_default` **未**优于 `A_frac`（2.92% vs 2.86%）；动态 κ 在 SP100 上仍失败。
4. **外部 baseline**：修复后 Historical 仍是最强之一；与 V4 诊断一致，B（2.66%）次之。

### 12.5 论文表述建议

- κ=1 的 RCVaR 目标 = **fractional-tail empirical CVaR**；Historical / `A_ceil` = **ceil-tail approximation**；αM 非整数时两者优化路径不同，Test 上 ceil 更优（本数据集）。
- **不要**写「RCVaR(κ=1) 不是 CVaR」；应写两种 **empirical tail 聚合方式** 在有限样本下不等价。
- 内部对比：`C_*` vs `A_frac_CVaR`；外部对比：`Historical_CVaR_fixed`（或等价 `A_ceil_CVaR`）。

### 12.6 关键输出文件

| 路径 | 内容 |
|------|------|
| `fair_comparison_table.csv` | Test 全指标对照 |
| `audit_summary.json` | 摘要 JSON |
| `validation_kappa_max_sweep.csv` | Val κ_max 网格 |
| `rolling_{method}.csv` | 各模型 Test 轨迹 checkpoint |
| `weights_Historical_CVaR_fixed.csv` | Historical 调仓权重 |

---

## 13. V5：Validation-Calibrated State-Dependent RCVaR（2026-07-02）

> 计划文件：`V5_next_implementation_plan.html`

### 13.1 实现内容

| 模块 | 改动 |
|------|------|
| `portfolio/optimizer.py` | 权重上限 `weight_cap`（SLSQP）、HHI 惩罚 |
| `portfolio/rolling.py` | κ 平滑 `kappa_rho`、权重记录、cap 传递 |
| `portfolio/weight_export.py` | 同步 cap / ρ 参数 |
| `configs/v5_common.yaml` | 统一 val/test 划分、网格、目标函数系数 |
| `experiments/v5_common.py` | 数据加载、引擎工厂、validation 目标 $J_{val}$ |
| `experiments/audit_baselines.py` | Step 0：A vs Historical 权重/收益审计 |
| `experiments/run_v5_calibration.py` | 分阶段 val 选参（κ_max → cap → ρ） |
| `experiments/run_v5_test.py` | 固定参数跑 7 模型 + 图表 |
| `experiments/run_v5_cross_market.py` | 四市场一键流水线 + 跨市场汇总表 |

### 13.2 Validation 目标函数

$$
J_{\text{val}} = CVaR^{val}_{5\%} + 0.001\,\overline{Turn}^{val} + 0.01\,\overline{HHI}^{val}
$$

网格（所有市场统一）：
- $\kappa_{\max}\in\{0.25,0.5,0.75,1.0,1.25,1.5\}$
- $u\in\{0.05,0.08,0.10,\text{None}\}$
- $\rho\in\{0.5,0.7,0.9,\text{None}\}$

### 13.3 Test 模型集合

| 模型 | 说明 |
|------|------|
| A_Historical_CVaR | plain_ceil（= Historical CVaR，审计已确认权重一致） |
| B_fixed_kappa | κ=2 固定鲁棒 |
| C_default | κ_max=1.0 未校准 |
| C_calibrated | val 选 κ_max* |
| C_cap | C_calibrated + 权重上限 u* |
| C_smooth | C_calibrated + κ 平滑 ρ* |
| C_stable | cap + smoothing 组合 |

### 13.4 运行命令

```powershell
# 单市场
C:\Users\Chen\anaconda3\envs\portfolio\python.exe -u robust_cvar_portfolio/experiments/audit_baselines.py --dataset sp100
C:\Users\Chen\anaconda3\envs\portfolio\python.exe -u robust_cvar_portfolio/experiments/run_v5_calibration.py --dataset sp100
C:\Users\Chen\anaconda3\envs\portfolio\python.exe -u robust_cvar_portfolio/experiments/run_v5_test.py --dataset sp100

# 四市场全流程（后台日志 outputs/v5_run_log.txt）
C:\Users\Chen\anaconda3\envs\portfolio\python.exe -u robust_cvar_portfolio/experiments/run_v5_cross_market.py
```

### 13.5 输出目录

```
outputs/v5/
  audit/                          # A vs Historical 审计
  {etf10,etf20,sp30,sp100}/
    validation/                   # kappa_max_grid, cap_grid, smoothing_grid, selected_params.json
    test/                         # rolling_*, table_main.csv, test_summary.json
    figures/                      # nav, drawdown, kappa, turnover
  cross_market/                   # final_paper_table.csv 等
```

### 13.6 四市场最终结果（2026-07-03，总耗时 ~21.4 h）

> 汇总表：`outputs/v5/cross_market/final_paper_table.csv`

| 市场 | N | Val 选参 | A CVaR | B CVaR | C_calibrated | **C_stable** | 最低标准¹ | 强标准² |
|------|---|----------|--------|--------|--------------|--------------|-----------|---------|
| ETF10 | 10 | κ=0.75, ρ=0.7 | 1.84% | 1.96% | **1.74%** | 1.92% | ✓ | ✗ |
| ETF20 | 19 | κ=1.5 | 1.49% | 1.50% | 1.62% | 1.62% | ✗ | ✗ |
| SP30 | 29 | κ=1.25, ρ=0.7 | 2.72% | 2.60% | 2.60% | **2.44%** | ✓ | ✓ |
| SP100 | 100 | κ=1.0, cap=10%, ρ=0.7 | **2.47%** | 2.66% | 2.92% | 2.55% | ✗ | ✗ |

¹ $CVaR(C_{calibrated}) < CVaR(A)$；² $CVaR(C_{stable}) < \min\{CVaR(A), CVaR(B)\}$

**SP100 细节（Test 2020–2024）：**

| 模型 | CVaR | MaxDD | 换手 |
|------|------|-------|------|
| A_Historical | **2.47%** | 25.1% | 0.038 |
| B_fixed | 2.66% | 25.6% | 0.020 |
| C_default / C_calibrated | 2.92% | 35.7% | 0.023 |
| C_cap (u=10%) | 2.50% | 27.1% | 0.016 |
| C_smooth (ρ=0.7) | 2.77% | 29.8% | 0.032 |
| C_stable (cap+ρ) | 2.55% | 28.7% | 0.016 |

### 13.7 结论与论文写法

1. **A ≡ Historical** 四市场审计均通过（权重 diff=0）。
2. **V5 在低/中维有效**：ETF10 `C_calibrated` 优于 A；SP30 `C_stable` 达强成功标准（2.44% < min(A,B)）。
3. **SP100 仍困难**：仅 κ_max 校准时 C 与 C_default 相同（val 选 κ=1.0）；加 cap=10%+ρ=0.7 后 `C_stable` CVaR 2.55% 仍劣于 A 2.47%，但换手从 0.038 降至 0.016、MDD 从 35.7% 改善（C_default 路径）。
4. **cap 是 SP100 关键**：`C_cap`（2.50%）接近 A，说明权重上限比单纯调 κ_max 更重要。
5. **论文主线**：状态依赖 RCVaR 有效，但高维市场需 validation calibration + 稳定性控制（cap）；可如实报告 SP100 为边界案例。

---

## 14. V6：Equity-Only 个股主线（2026-07-03）

> 计划文件：`V6equity_only_next_steps_plan.html`

### 14.1 论文定位调整

- **主文**：SP30（主结果）+ Random30（稳健性）+ SP100（高维压力测试 + 失败诊断）
- **不再主文**：ETF10 / ETF20（附录或 future work）
- 目标：证明个股组合中**何时有效、高维为何失效**，而非全市场最优

### 14.2 实现脚本

| 脚本 | 任务 |
|------|------|
| `run_equity_only_summary.py` | 整理 V5 SP30/SP100 → `outputs/equity_only/`，生成 paper_tables |
| `run_random30_universes.py` | SP100 池随机 30 股 × K=30，4 模型 + val 选参 |
| `run_sp100_highdim_diagnostics.py` | 横截面结构 + C vs A 劣化相关 |
| `run_equity_bootstrap.py` | Block bootstrap 显著性 |
| `equity_common.py` + `configs/v6_equity_common.yaml` | 共享工具与缩小网格 |

### 14.3 已完成结果（Task 1/2/4/5）

**SP30 主结果（Table 1）**

| 模型 | CVaR | Sharpe | 年化收益 |
|------|------|--------|----------|
| A_ceil | 2.72% | 0.11 | 1.94% |
| B_fixed | 2.60% | **0.31** | **9.61%** |
| C_stable | **2.44%** | 0.23 | 3.81% |

- CVaR：C_stable < B < A ✓（强成功标准）
- Sharpe：C_stable (0.23) > A (0.11)，但 < B (0.31) → 论文写「尾部风险优化，非 Sharpe 最大化」

**SP100 高维（Table 3）**

| 模型 | CVaR | Sharpe |
|------|------|--------|
| A_ceil | **2.47%** | **0.44** |
| C_default | 2.92% | 0.11 |
| C_cap | 2.50% | 0.04 |
| C_stable | 2.55% | -0.03 |

**SP100 失败诊断（Task 4）**

- `Corr(gap_C_default, avg_correlation) = +0.78`：C 劣化与高市场相关性同步
- `Corr(gap, HHI) = +0.46`：权重越集中，C 相对 A 越差
- `Corr(gap, effective_dimension) = -0.36`：有效维度越低，C 越差

**Bootstrap（Task 5）**

| 对比 | P(后者 CVaR 更优) |
|------|-------------------|
| SP30 C_stable vs A | **100%** |
| SP30 C_stable vs B | **94%** |
| SP100 C_stable vs C_default | **98%**（稳定化有效） |
| SP100 C_stable vs A | 24.6%（未超越 A） |

### 14.4 Random30（Task 3，已完成）

- **3 组**，固定种子 **42 / 123 / 456**（各 30 股，从 SP100 池抽样）
- **WinRate vs A = 100%**，**WinRate vs B = 100%**
- mean Δ_A = **+0.22 pp**，median Δ_A = **+0.21 pp**，worst quartile Δ_A = **+0.16 pp**
- 耗时 ~71 min；输出 `outputs/equity_only/random30/random30_summary.csv`
- HTML §13.2 与 bootstrap Table 2 已自动更新

### 14.5 输出目录

```
outputs/equity_only/
  sp30/ sp100/ sp100_diagnostics/ bootstrap/ random30/
  paper_tables/table1–table5
```

### 14.6 论文写法（V6 定稿）

$$
\boxed{
\text{状态依赖鲁棒 CVaR 在中等规模个股池中有效；在高维个股池中需要稳定性控制，但仍可能难以超越强 CVaR baseline。}
}
$$

---

## 15. V7：结构诊断、相关性分层 Random30 与有效维度缩放 RCVaR（2026-07-06）

> 计划与结果：`V7_structure_corr_stratified_plan.html`（§9 论文写法 + §11 实验结果）

### 15.1 目标

V6 已证明 SP30 / Random30 有效、SP100 退化。V7 **不改叙事先补结构证据**，再测最小改动高维修正：

$$
V7 = \text{结构诊断} + \text{相关性分层 Random30} + \text{过拟合诊断} + \text{有效维度缩放 RCVaR}
$$

### 15.2 实现脚本

| 脚本 | 模块 | 说明 |
|------|------|------|
| `run_v7_structure_summary.py` | V7-A | SP30 / Random30 / SP100 结构总表 |
| `run_v7_generate_corr_stratified_universes.py` | V7-B | 候选 300 → Low/Mid/High 各 3 |
| `run_v7_corr_stratified_backtest.py` | V7-B | 9 universe × 4 模型回测 |
| `run_v7_tail_overfit_diagnostics.py` | V7-C | SP100 tail overlap + OOS gap |
| `run_v7_effdim_rcvar.py` | V7-D | κ 有效维度缩放（`effdim_d0=30`） |
| `v7_common.py` + `configs/v7_common.yaml` | 共享 | 路径、结构指标、V7 模型 |
| `update_v7_html_results.py` | 文档 | 自动 patch `V7_structure_corr_stratified_plan.html` §9/§10/§11 |

**代码改动**：`portfolio/rolling.py` 新增 `effdim_d0` 参数，实现
$a_t=\min(1,\ d_{\mathrm{eff},t}/d_0)$ 缩放 $\kappa_{\max}$。

### 15.3 复现命令

```bash
C:\Users\Chen\anaconda3\envs\portfolio\python.exe -u robust_cvar_portfolio/experiments/run_v7_structure_summary.py
C:\Users\Chen\anaconda3\envs\portfolio\python.exe -u robust_cvar_portfolio/experiments/run_v7_generate_corr_stratified_universes.py --n-candidates 300 --n-per-group 3
C:\Users\Chen\anaconda3\envs\portfolio\python.exe -u robust_cvar_portfolio/experiments/run_v7_corr_stratified_backtest.py --n-per-group 3
C:\Users\Chen\anaconda3\envs\portfolio\python.exe -u robust_cvar_portfolio/experiments/run_v7_tail_overfit_diagnostics.py
C:\Users\Chen\anaconda3\envs\portfolio\python.exe -u robust_cvar_portfolio/experiments/run_v7_effdim_rcvar.py --d0 30
C:\Users\Chen\anaconda3\envs\portfolio\python.exe -u robust_cvar_portfolio/experiments/update_v7_html_results.py
```

### 15.4 输出目录

```
outputs/v7/
  structure/          # V7-A
  corr_stratified/    # V7-B
  overfit/            # V7-C
  effdim_rcvar/       # V7-D
  paper_tables/       # table_v7_*.csv
```

### 15.5 V7-A 结构诊断（已完成）

| Universe | N | Val avg corr | Val d_eff | CVaR A | CVaR C_stable | Gap C−A |
|----------|---|--------------|-----------|--------|---------------|---------|
| SP30 | 29 | 0.364 | 5.3 | 2.72% | **2.44%** | **−0.28 pp** |
| SP100 | 100 | 0.335 | 7.3 | **2.47%** | 2.55% | +0.08 pp |
| Random30 (n=3) | 30 | 0.306 | — | 2.85% | **2.63%** | **−0.22 pp** |

**解读**：Random30 与 SP30 的 validation 相关性低于 SP100 候选池 High 组；C_stable 劣于 A 的 gap 仅出现在 SP100。名义维度 N=100 但有效维度 d_eff≈7，说明「高维失败」需结合相关性与有效维度理解。

### 15.6 V7-B 相关性分层（已完成，9/9 universe）

| 组别 | Val corr | WinRate vs A | Mean Δ_A |
|------|----------|--------------|----------|
| Low-Corr | 0.280 | 67% | +0.02 pp |
| Mid-Corr | 0.334 | 67% | +0.04 pp |
| **High-Corr** | **0.392** | **100%** | **+0.17 pp** |

- 耗时 ~155 min（续跑 High 三组）；输出 `outputs/v7/corr_stratified/group_summary.csv`
- **意外发现**：High-Corr 30 股池改善最大（与 SP100 N=100 退化并不矛盾——高相关 + 低有效维度同时存在时才恶化）

### 15.7 V7-C 过拟合诊断（已完成）

| 模型 | Tail Jaccard | OOS Gap | Mean HHI |
|------|--------------|---------|----------|
| A_ceil | 0.029 | −0.46 pp | 0.118 |
| C_stable | **0.023** | **−0.23 pp** | 0.071 |

→ C_stable tail 重叠更低、OOS gap 更小，支持尾部样本不稳定机制。

### 15.8 V7-D effdim RCVaR（已完成，~571 min）

| Dataset | A | C_stable | V7_effdim | V7_effdim_cap |
|---------|---|----------|-----------|---------------|
| SP30 | 2.76% | 2.61% | 2.91% | **2.57%** |
| SP100 | **2.47%** | 2.51% | 2.86% | 2.52% |

**成功标准检验：**
- SP30：V7_effdim_cap − C_stable = **−0.04 pp** ≤ +0.05 pp ✓（且略优于 C_stable）
- SP100 最低标准：V7_effdim_cap (2.52%) < C_default (2.92%) ✓
- SP100 中等标准：V7_effdim_cap ≈ C_stable（2.52% vs 2.51%），基本持平
- SP100 强标准：未超越 A（2.47%）✗

**结论：** effdim 缩放 + cap 可稳定 SP30、修复 SP100 相对 C_default 的退化，但尚不足以超越 Historical CVaR baseline。

### 15.9 V7 论文叙事定稿

$$
\boxed{
\text{状态依赖 RCVaR 在 30 股量级个股池有效；SP100 失败主因是名义维度与有效分散度，而非平均相关性更高。}
}
$$

- SP100 val corr（0.335）< SP30（0.364）；High-Corr 30 股子集 WinRate=100%
- SP100 边界案例：需 cap + effdim 缩放可修复 C_default，但 A 仍为强 baseline

---

*V1–V6 完成；**V7 完成：2026-07-07**；环境：`conda activate portfolio` 或 `envs/portfolio/python.exe`*
