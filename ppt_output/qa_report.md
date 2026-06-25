# QA 报告

- PPTX 生成状态：成功
- 文件：`final_presentation_cn.pptx`
- 幻灯片数量：16
- 嵌入媒体数量：1
- 语言：中文，保留必要英文术语、公式和变量名
- 字体：PPTX XML 已指定中文宋体 SimSun，英文/数字/公式 Times New Roman
- 结构：按混合听众演讲逻辑组织：总览、术语、缺口、方法、证据、边界、问答
- 讲解稿：已生成 `paper_explanation_cn.md`，包含术语解释、论文逻辑和现场问答口径
- 图像：使用 PDF 第 5 页 Figure 1 面板，并生成总览图
- 来源标注：每页底部包含小字号页码、公式或 Figure 来源
- Speaker notes：每页已写入简短中文讲解备注
- 自审修正：避免长段落，使用短标题、少量 bullet、图证据优先布局
- 结构检查：已用 python-pptx 重新打开并检查 shape 是否越界
- 越界问题：无
- `soffice` 检测：未在 PATH、/Applications、/opt/homebrew/bin、/usr/local/bin 中找到可执行 soffice
- 渲染预览：已使用 aspose.slides 在 portfolio 环境导出 16 张 PNG 预览；预览图带 Aspose evaluation watermark，仅用于 QA，不写入 PPTX。
- 依赖说明：PPTX 由 python-pptx 生成；PNG 预览可用 aspose.slides + mono-libgdiplus 完成。LibreOffice/soffice 不是当前必需路径。