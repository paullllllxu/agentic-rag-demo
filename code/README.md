# 核心实现快照

本目录选取原工程中的11份文件，保留原有实现和相对依赖引用，供阅读方案如何落到代码。文件清单与SHA256见[code-snapshot.json](../evidence/code-snapshot.json)。

**这不是完整可运行代码包。** 原工程的配置、部分依赖模块、知识数据、索引、模型、安装流程及运行环境未打包，不支持从本目录直接启动服务。快照中的路径引用只说明原工程结构。

建议按顺序阅读：question_policy与planning → retrieval → pipeline → evidence与semantic_guards → conversation → update_reference → gradio_app与sharing → quality。

项目背景、验证口径与当前边界分别见[首页](../README.md)和[验证说明](../docs/evaluation.md)。本展示仓库未附通用开源许可证，第三方权利与发布范围见[NOTICE](../NOTICE.md)。
