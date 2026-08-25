# 临时工作区

此目录存放所有不应提交到 Git 的临时内容。首次使用时可按需建立以下子目录：

```text
temp/
├── tests/       # 测试代码和测试结果
├── interim/     # 中间数据
├── logs/        # 运行日志
├── cache/       # 缓存文件
├── previews/    # 临时预览文件
└── scratch/     # 临时脚本和探索性实验
```

除本说明文件外，`temp/` 中的内容均由 `.gitignore` 排除。不要使用 `git add -f` 强制提交。

