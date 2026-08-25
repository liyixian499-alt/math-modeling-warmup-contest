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
├── docs/                   # 分析、思路和过程记录
├── src/
│   ├── 问题一/             # 问题一正式代码
│   ├── 问题二/             # 问题二正式代码
│   ├── 问题三/             # 问题三正式代码
│   └── 可视化/             # 绘图与可视化代码
├── results/
│   ├── figures/            # 确定保留的结果图片
│   ├── tables/             # 确定保留的结果表格
│   ├── outputs/            # 确定保留的数据输出
│   └── reports/            # 验证与分析报告
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
- `results/` 只保存确定需要共享、复现或写入论文的最终结果。
- 不创建 `notebooks/`、`tests/`、`paper/final/` 或 `paper/sections/`。

## 环境安装

```bash
python -m pip install -r requirements.txt
```

