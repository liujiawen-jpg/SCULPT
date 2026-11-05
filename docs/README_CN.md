# SCULPT（J0）匿名源码使用说明

本仓库是 **J0 联合训练版本**，保留共享作用反演、背景条件观测算子、选择性
STRING 连接、正表达形状生成和基因秩重排。不是 W1/W2 的 WaveGC 版本，也不使用
旧模型预测加残差的结构。

## 包含哪些内容

- 五个数据集的公开下载地址、版本与完整性检查：见 [数据说明](DATASETS.md)。
- 原始数据预处理、A2048 归一化、固定基因和靶点坐标。
- 五折训练/验证/测试划分代码及已发布的划分元数据。
- 从头联合训练、断点恢复、收敛判定、验证集选模。
- 每个条件生成 512×5979 细胞表达云，封存后评估。
- 扰动拟合和分布拟合两类评价，以及先平均种子、再平均细胞系的汇总。
- `Compared_methods/` 中 TxPert、CPA、PIE、scPILOT、STATE、CellFlow、PopPert 的官方版本、环境和配置。
- Target-Control 和直接绝对表达 Source-Mean 的无训练基线，五折各运行一次。

## 推荐运行顺序

所有完整命令见仓库首页 [README](../README.md)：

1. 配置 Python 3.11、PyTorch 2.7.1，安装本包。
2. 运行 `python -m pytest -q tests Compared_methods/tests` 和合成 smoke 测试。
3. 下载五份公开数据，运行 `cosit prepare` 与 `cosit audit-data`。
4. 按 [先验说明](PRIORS.md) 生成通用预训练表示与 STRING 图。
5. 运行 `cosit split --verify-frozen configs/splits` 验证划分。
6. 运行 `scripts/run_benchmark.py`，按五折和三个种子逐项训练、选模、测试。

## 结果如何解释

当前任务是**已有相同靶点来源证据的跨细胞系迁移**，输出 5979 个共有基因。
不能据此直接声称解决未知靶点、double OOD 或未测基因预测。

三个随机种子改变模型初始化和训练顺序，不改变划分。最终均值与方差先对每个
测试细胞系的三个种子取均值，再对五个细胞系汇总。不能把 15 次结果当作 15 个
独立细胞系样本。

仓库只包含源码、配置、说明及必要的公开 ID 元数据，不含表达矩阵、模型权重、
缓存或实验预测。完整功能已做合成验证；本次整理没有重新进行全量生物实验。

## 匿名性与实现核验

包内不包含 Git 历史、作者联系方式、服务器路径、权重或预测缓存。
`cosit` 是为保留原 J0 接口而沿用的包名，论文中的方法名是 SCULPT。
J0 的完整流程与第三方方法的配置/原生计算适配器有明确区分：新增比较方法
并不意味着把七个上游的完整训练驱动重写成一个通用脚本。
详见 [实现核验](J0_FIDELITY.md)、[匿名打包说明](ANONYMOUS_RELEASE.md)
和 [对比方法入口](../Compared_methods/README.md)。

## 本次整理入口

- `Makefile`：测试、合成流程、下载、预处理、划分、训练及五折运行快捷命令。
- `docs/REPRODUCTION_CHECKLIST.md`：按顺序复现的操作清单与交付范围。
- `scripts/download_resources.py`：通用外部先验下载，固定 Tahoe 版本并核验已登记哈希。
- `Compared_methods/PREPROCESSING.md`：对比方法共享输入、PIE 标签、PopPert 群体参数及 CellFlow PCA 的预处理。

本包提供 SCULPT 完整可移植训练流水线。对比方法提供官方固定版本下载、环境、配置、原生 API、预处理和输出接入；不将这些接口冒充七套已完整移植的训练调度器。对比方法的训练须依照各自配置和官方训练入口整合。
