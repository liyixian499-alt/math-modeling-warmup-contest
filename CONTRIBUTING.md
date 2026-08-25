# 仓库协作与文件放置规则

本文件是仓库成员共同遵守的默认规则。提交文件前，应先确认文件类型和归属目录。

## 文件放置规则

| 文件类型 | 指定位置 |
| --- | --- |
| 题目原文 | `problem/statement/` |
| 题目附件 | `problem/attachments/` |
| 未修改的原始数据 | `data/raw/` |
| 清洗完成、可直接建模的数据 | `data/processed/` |
| 分析思路、会议记录和过程文档 | `docs/` |
| 问题一正式代码 | `src/问题一/` |
| 问题二正式代码 | `src/问题二/` |
| 问题三正式代码 | `src/问题三/` |
| 绘图与可视化代码 | `src/可视化/` |
| 最终结果图片 | `results/figures/` |
| 最终结果表格 | `results/tables/` |
| 最终数据输出 | `results/outputs/` |
| 验证或分析报告 | `results/reports/` |
| 论文插图 | `paper/figures/` |
| 论文表格 | `paper/tables/` |
| 文献与参考资料 | `references/` |
| 测试代码及测试结果 | `temp/tests/` |
| 中间数据 | `temp/interim/` |
| 运行日志 | `temp/logs/` |
| 缓存文件 | `temp/cache/` |
| 临时预览文件 | `temp/previews/` |
| 临时脚本和探索性实验 | `temp/scratch/` |

## 强制约定

1. 不在仓库根目录随意堆放代码、图片、表格或数据文件。
2. 不创建 `notebooks/`、`tests/`、`paper/final/` 或 `paper/sections/`。
3. 所有测试代码只能放在 `temp/tests/`，不得使用 `git add -f` 强制提交 `temp/` 内容。
4. 所有可重复生成的中间文件统一放在 `temp/`，不得混入 `data/`、`src/` 或 `results/`。
5. `data/raw/` 中的原始数据原则上只读；处理后的正式数据写入 `data/processed/`。
6. `results/` 只接收已经检查、需要共享或用于论文的最终结果。
7. Python 文件使用有意义的名称；同一问题内避免出现大量 `final`、`new`、`v2` 等版本后缀。
8. 不提交密码、令牌、个人信息或其他敏感内容。
9. 大型数据或二进制文件提交前应先与团队确认。

## 提交前检查

- 文件是否位于指定目录？
- 是否误提交了测试、缓存、日志或中间文件？
- 代码能否从仓库根目录按说明运行？
- 结果是否可以由正式代码复现？
- README 或依赖文件是否需要同步更新？

