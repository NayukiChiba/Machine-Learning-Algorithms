# Machine-Learning-Algorithms

机器学习算法学习与实现仓库。

这个项目目前以统一的 CLI 为入口，围绕 `src/mlAlgorithms` 组织代码，覆盖数据集构造、模型训练、评估、分析报告和结果可视化。仓库既保留了 `Basic/` 下的基础库练习，也提供了更清晰的端到端流水线实现。

在线文档：
[https://nayukichiba.github.io/Machine-Learning-Algorithms/](https://nayukichiba.github.io/Machine-Learning-Algorithms/)

## 项目特点

- 统一的命令行入口：通过 `main.py` 统一列出、分析和运行流水线
- 清晰的分层结构：`catalog -> datasets/training/evaluation/visualization -> workflows`
- 覆盖多类任务：分类、回归、聚类、降维、集成学习、概率与序列模型
- 输出结果可落盘：图表和相关产物统一写入 `outputs/`
- 有基础测试保障：包含 `unit`、`integration`、`smoke` 三层测试

## 环境要求

- Python 3.10+
- 建议在虚拟环境中安装依赖

安装依赖：

```bash
pip install -r requirements.txt
```

当前 `requirements.txt` 已包含以下开发和运行依赖：

- `scikit-learn`
- `matplotlib`
- `seaborn`
- `joblib`
- `packaging`
- `pytest`
- `ruff`
- `pre-commit`
- `hmmlearn`
- `lightgbm`
- `xgboost`

说明：

- `lightgbm`、`xgboost`、`hmmlearn` 在代码中按可选依赖处理
- 若环境中缺少对应库，相关流水线会提示缺失，部分集成测试会自动跳过

## 快速开始

在仓库根目录执行：

```bash
python main.py list
```

查看某条流水线的数据分析：

```bash
python main.py analyze classification.logistic_regression
```

运行单条流水线：

```bash
python main.py run classification.logistic_regression
```

运行某一组流水线：

```bash
python main.py suite classification
python main.py suite regression
python main.py suite clustering
python main.py suite dimensionality
python main.py suite ensemble
python main.py suite probabilistic
python main.py suite all
```

## CLI 命令说明

项目当前统一使用 [main.py](main.py) 作为入口。

支持命令：

- `python main.py list`
  列出全部可用流水线、数据集和输出目录
- `python main.py analyze <pipelineId>`
  只执行数据探索与终端报告输出
- `python main.py run <pipelineId>`
  执行单条完整流水线
- `python main.py suite <groupName>`
  按任务分组批量运行流水线

## 当前支持的流水线

### 分类

- `classification.logistic_regression`
- `classification.decision_tree`
- `classification.svc`
- `classification.naive_bayes`
- `classification.knn`
- `classification.random_forest`

### 回归

- `regression.linear_regression`
- `regression.svr`
- `regression.decision_tree`
- `regression.regularization`

### 聚类

- `clustering.kmeans`
- `clustering.dbscan`

### 降维

- `dimensionality.pca`
- `dimensionality.lda`

### 集成学习

- `ensemble.bagging`
- `ensemble.gbdt`
- `ensemble.lightgbm`
- `ensemble.xgboost`

### 概率与序列模型

- `probabilistic.em`
- `probabilistic.hmm`

## 项目结构

当前仓库的核心结构如下：

```text
Machine-Learning-Algorithms/
├── Basic/                         # 基础库练习
├── docs/                          # 文档站点
├── outputs/                       # 运行输出目录
├── src/
│   ├── __init__.py                # src 包入口
│   └── mlAlgorithms/
│       ├── analysis/              # 数据分析与终端报告
│       ├── catalog/               # 数据集与流水线注册表
│       ├── core/                  # 核心数据结构与上下文对象
│       ├── datasets/              # 数据集构造与目录组装
│       ├── evaluation/            # 评估指标计算
│       ├── training/              # 训练函数
│       ├── visualization/         # 数据图与结果图
│       └── workflows/             # 端到端执行流程
├── tests/
│   ├── integration/               # 集成测试
│   ├── smoke/                     # CLI 烟雾测试
│   └── unit/                      # 单元测试
├── config.py                      # 全局配置与输出目录解析
├── main.py                        # 统一 CLI 入口
├── requirements.txt
└── README.md
```

## 核心模块说明

### `src/mlAlgorithms/core`

定义项目里的基础数据结构，例如：

- `DatasetSpec`
- `PipelineSpec`
- `RunContext`
- `RunResult`
- `Registry`

### `src/mlAlgorithms/catalog`

负责注册所有数据集和流水线，是 CLI 查询和执行的统一入口。

- [src/mlAlgorithms/catalog/datasets.py](src/mlAlgorithms/catalog/datasets.py)
- [src/mlAlgorithms/catalog/pipelines.py](src/mlAlgorithms/catalog/pipelines.py)

### `src/mlAlgorithms/datasets`

负责构造可复现的数据集规格。

- 表格任务数据集位于 `datasets/tabular/`
- 序列任务数据集位于 `datasets/sequence/`

### `src/mlAlgorithms/training`

负责训练函数实现，按任务类型拆分：

- `classification/`
- `regression/`
- `clustering/`
- `dimensionality/`
- `probabilistic/`

### `src/mlAlgorithms/workflows`

负责把数据、训练、评估和可视化串成完整流程。

入口分发在：
[src/mlAlgorithms/workflows/executor.py](src/mlAlgorithms/workflows/executor.py)

### `src/mlAlgorithms/analysis` 与 `visualization`

- `analysis/` 负责探索性统计与终端可读报告
- `visualization/data/` 负责原始数据图
- `visualization/result/` 负责模型结果图

## 输出目录

项目输出默认写入 `outputs/`，目录由 [config.py](config.py) 中的 `resolveOutputDir` 管理。

常见行为：

- 已知 key 会写入预定义目录
- 未知 key 会写入 `outputs/<key>`
- 运行流水线时通常会写入 `outputs/<spec.outputKey>/`

## 测试

运行全部测试：

```bash
python -m pytest tests
```

按层运行：

```bash
python -m pytest tests/unit
python -m pytest tests/integration
python -m pytest tests/smoke
```

当前测试覆盖：

- 核心契约测试
- 分析器测试
- 图像保存与学习曲线输入测试
- 代表性流水线端到端集成测试
- CLI 烟雾测试
- 私有导入约束测试

## 代码风格与开发

格式化代码：

```bash
ruff format .
```

静态检查：

```bash
ruff check .
```

如需启用提交前检查：

```bash
pre-commit install
```

## 文档

本仓库包含 `docs/` 文档站点，可本地预览：

```bash
cd docs
npm ci
npm run docs:dev
```

## 许可

本项目使用 [Apache License 2.0](LICENSE)。
