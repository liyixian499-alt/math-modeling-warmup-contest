# 数学建模磨合赛

本仓库用于数学建模磨合赛的团队协作，统一管理题目、数据、建模代码、结果、论文与参考资料。

## 仓库结构

```text
.
├── problem/
│   ├── statement/          # 题目原文
│   └── attachments/        # 题目附件
├── data/
│   ├── raw/                # 原始数据
│   └── processed/          # 最终建模数据
├── docs/                   # 分析思路、过程记录及全部分析报告
├── src/
│   ├── 问题一/             # 问题一正式代码
│   ├── 问题二/             # 问题二正式代码
│   ├── 问题三/             # 问题三正式代码
│   └── 可视化/             # 绘图与可视化代码
├── results/
│   ├── figures/            # 确定保留的结果图片
│   ├── tables/             # 确定保留的结果表格
│   └── outputs/            # 确定保留的数据输出
├── paper/
│   ├── figures/            # 论文插图
│   ├── tables/             # 论文表格
│   ├── main.tex            # 完整论文主文件
│   ├── references.bib      # 论文参考文献
│   └── cumcmthesis.cls     # 数学建模论文 LaTeX 模板
├── references/             # 文献与参考资料
└── temp/                   # 测试、中间数据、缓存和其他临时内容
```

## 协作规则

所有成员在添加或移动文件前，必须阅读并遵守 [CONTRIBUTING.md](CONTRIBUTING.md)。

核心要求：

- 正式代码只放在 `src/` 的四个指定目录中。
- 测试、中间文件、缓存、日志、临时脚本和预览文件统一放在 `temp/`。
- `temp/` 中除说明文件外的内容不会提交到 Git。
- 建模、验证及分析报告统一放在 `docs/`。
- 用于切换到下一次 Codex 对话的“交付文档”统一采用 Markdown 格式并放在 `docs/`；除非明确要求，不生成 Word 或 PDF 版本。
- `results/` 只保存确定需要共享、复现或写入论文的最终图片、表格和数据输出。
- 不创建 `notebooks/`、`tests/`、`paper/final/` 或 `paper/sections/`。

## 环境安装

```bash
python -m pip install -r requirements.txt
```

## 问题一数值求解

从仓库根目录运行：

```bash
python -m src.问题一
```

程序会从 `problem/attachments/附件1 放热能力数据.xlsx` 读取真实 DSC 数据，执行六状态动态皮肤血流主模型、五状态准稳态极限对照、严格容差复核、指定的单因素敏感性分析以及血流响应时间结构敏感性分析，最终 CSV 保存在 `results/outputs/problem1/`。

