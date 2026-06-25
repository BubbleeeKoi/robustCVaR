# HTML 文献解读与复现审计

## 文件作用

本目录提供一份重新设计的中文长篇 HTML 报告。它不是 PPT 的网页转换，而是按“论文问题、理论方法、实验设计、代码映射、复现操作、结果审计、改进路线”组织的可连续阅读文档。

## 打开方式

直接打开：

```text
html_output/index.html
```

页面不依赖 MathJax、前端框架或网络资源，离线可用。正文使用宋体优先，英文、公式和代码使用 Times New Roman 优先。

## 页面内容

1. 论文解决什么问题，以及为什么 Robust MDP 和 CVaR RL 单独使用都不够。
2. 固定 RN/KL ambiguity budget 如何与 CVaR/EVaR 对偶集合合并。
3. 决策相关 `kappa(x,a)` 为什么需要 NCVaR 和增广状态 `(x,y)`。
4. Algorithm 2 为什么对 `yV(x,y)` 做线性插值。
5. Figure 1 的设计逻辑、原图与项目输出对照。
6. 仓库中每个 Markdown、Python、CSV、NPY 和 PNG 文件的职责。
7. 从零设计和运行复现实验的完整顺序。
8. 当前路径差异与代码审计结论。
9. KL/EVaR 参数口径不一致、Figure 1d 中 `kappa` 被截断等关键问题。

## 图片资产

`html_output/assets/` 保存从论文 PDF 与当前复现图中整理出的面板图片。论文面板来自 `outputs/pdf_images/`，复现面板来自 `outputs/strict_reproduction_gridworld_paper_style.png`。
