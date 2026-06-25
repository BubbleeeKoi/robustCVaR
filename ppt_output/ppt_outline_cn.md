# 中文 PPT 汇报提纲

## 论文定位

- 类型：方法 / 算法论文
- 汇报逻辑：problem-to-solution
- 主线：模型不确定下的风险敏感 RL → 固定 ambiguity set 的 CVaR/EVaR 等价 → 决策相关 uncertainty 的 NCVaR → GridWorld 验证

## 中心问题

标准 MDP 假设转移概率固定，但真实 RL 中模型估计误差普遍存在；风险中性 RMDP 又忽略尾部高代价事件。本文要解决的是：如何在模型不确定时最小化轨迹代价的最坏情形 CVaR。

## 核心贡献

1. 固定预算 ambiguity set 下，把 robust CVaR 转换为已有风险敏感 RL 问题。
2. RN ambiguity 对应更小置信水平的 CVaR；KL ambiguity 对应 EVaR。
3. 决策相关 uncertainty 下定义 NCVaR，并给出分解定理与 value iteration。
4. 用 GridWorld 显示不同 uncertainty set 会改变最优路径和价值函数。
5. 为混合听众补充术语地图和专业问答边界。

## 幻灯片结构

1. 标题与一句话定位
2. 一页讲清楚文章在做什么
3. 关键术语地图
4. RMDP 与 CVaR RL 的缺口
5. Robust CVaR 问题形式
6. RN 固定预算如何变成 CVaR
7. KL 固定预算如何连接 EVaR
8. 为什么 1d 是 NCVaR
9. NCVaR Bellman 分解
10. 线性插值 Algorithm 2
11. GridWorld 实验设计
12. Figure 1 核心证据
13. 验证逻辑和边界
14. 意义、局限和讨论
15. 专业追问与回答边界
16. 面向所有听众的三句话总结