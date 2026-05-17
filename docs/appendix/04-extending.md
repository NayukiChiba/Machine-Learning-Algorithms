---
title: 项目架构 — 扩展指南
outline: deep
---

# 扩展指南

## 本章目标

1. 掌握新增一个算法流水线的完整步骤。
2. 理解 `PipelineSpec` 各字段的填写规则。
3. 理解项目的关键设计决策及其理由。

---

## 1. 新增算法步骤

新增一个算法需要在 6 个位置添加代码：

### 步骤 1：数据层

在对应的 `*DatasetFactory` 中新增 `load*Dataset()` 方法。

```python
# 示例：src/mlAlgorithms/datasets/tabular/regressionDatasets.py
def loadNewAlgorithmDataset(self) -> DataFrame:
    """加载新算法数据。"""
    rng = np.random.RandomState(self.randomState)
    # ... 生成或加载数据 ...
    return DataFrame({...})
```

### 步骤 2：数据注册

在 [`datasetCatalog.py`](https://github.com/NayukiChiba/Machine-Learning-Algorithms/blob/main/src/mlAlgorithms/datasets/datasetCatalog.py) 的 `buildDatasetSpecs()` 中添加 `DatasetSpec`：

```python
DatasetSpec(
    "regression.new_algorithm",       # 与 PipelineSpec.datasetId 一致
    TaskType.REGRESSION,
    DataKind.TABULAR,
    regression.loadNewAlgorithmDataset,
    "price",                           # 标签列名
    None,                              # 自动推断特征列
    "新算法数据描述",
),
```

### 步骤 3：训练层

在对应的 `*Models.py` 中新增训练函数：

```python
def trainNewAlgorithmModel(XTrain, yTrain, randomState: int = 42):
    """训练新算法。"""
    model = SomeModel(param1=..., param2=..., random_state=randomState)
    model.fit(XTrain, yTrain)
    return model
```

### 步骤 4：流水线注册

在 [`pipelines.py`](https://github.com/NayukiChiba/Machine-Learning-Algorithms/blob/main/src/mlAlgorithms/catalog/pipelines.py) 的 `PIPELINE_REGISTRY` 注册循环中添加 `PipelineSpec`：

```python
PipelineSpec(
    "regression.new_algorithm",
    TaskType.REGRESSION,
    "regression.new_algorithm",
    RunnerType.REGRESSION,
    trainNewAlgorithmModel,
    "standardScaler",                  # 或 None
    "randomSplit",
    "default",
    "regression",
    "regression",
    ["correlationHeatmap", "featureTargetScatter"],
    ["featureImportance"],
    ["learningCurve"],
    "new_algorithm",
    metadata={
        "learningCurveEstimatorFactory": _buildLearningCurveFactory(
            "regression.new_algorithm"
        )
    },
),
```

### 步骤 5：工厂函数

如需学习曲线，在 `_buildLearningCurveFactory()` 中添加映射：

```python
"regression.new_algorithm": lambda: SomeModel(param1=..., random_state=42),
```

如需二维决策边界可视化（分类模型），在 `_buildVisualModelFactory()` 中添加类似映射。

### 步骤 6：文档

在 `docs/{domain}/{algorithm}/` 下创建 9 个文件：

```
docs/regression/new_algorithm/
├── index.md                       # 概述 + 定位对比 + 文件导航
├── 01-mathematics.md              # 数学原理 + 数学-代码映射
├── 02-data.md                     # 数据构成
├── 03-intuition.md                # 思路与直觉
├── 04-model.md                    # 模型构建
├── 05-training-and-prediction.md  # 训练与预测
├── 06-evaluation.md               # 评估与诊断
├── 07-implementation.md           # 工程实现
└── 08-exercises-and-references.md # 练习与参考文献
```

---

## 2. PipelineSpec 字段填写指南

| 字段 | 填写方式 |
|---|---|
| `id` | `{domain}.{algorithm}`——如 `"regression.svr"` |
| `taskType` | `TaskType` 枚举——决定数据探索报告格式 |
| `datasetId` | 与 `DatasetSpec.id` 完全一致 |
| `runnerType` | `RunnerType` 枚举——通常与 taskType 相同 |
| `trainer` | 训练函数引用——不调用，只传递 |
| `preprocessor` | `"standardScaler"`：RBF 核模型和正则化模型必须；`None`：树模型和线性回归 |
| `splitter` | `"randomSplit"`（回归/聚类）；`"stratifiedSplit"`（分类）；`None`（无需切分，如 HMM） |
| `predictor` | `"default"`（回归/聚类）；`"ldaClassifier"`（LDA）；`"hmmPredictor"`（HMM）；`"gmmPredictor"`（GMM） |
| `evaluator` | `"default"`（回归/分类）；`"transformOnly"`（PCA 等纯变换） |
| `analysisProfile` | 与 evaluator 相同或更具体（如 `"regression"`、`"classification"`） |
| `dataPlots` | 从已有列表选择：`correlationHeatmap` / `featureTargetScatter` / `classDistribution` / `labeledScatter2d` / `featureSpace2d` / `featureSpace3d` / `rawScatter2d` |
| `resultPlots` | 从已有列表选择：`featureImportance` / `confusionMatrix` / `rocCurve` / `classificationResult` / `decisionBoundary` |
| `diagnostics` | 从已有列表选择：`learningCurve` / `treeStructure` / `kmeansSweep` / `dbscanKDistance` / `dbscanEpsSweep` / `gmmComponentSweep` / `pcaTrainingCurve` |
| `outputKey` | 输出子目录名——产物保存到 `outputs/{outputKey}/` |
| `optionalDependencies` | 可选依赖包名——如 `("hmmlearn",)` `("xgboost",)` `("lightgbm",)` |
| `metadata` | `multiModel: True`（多模型）；`learningCurveEstimatorFactory`（学习曲线工厂）；`visualModelFactory`（决策边界工厂） |

---

## 3. 关键设计决策

### 3.1 为什么不用 scikit-learn Pipeline

当前代码**显式编排**每一步，而非使用 `sklearn.pipeline.Pipeline`：

- **教学透明**：每一步的输入输出都是命名清晰的中间变量——`X_train_s`、`y_pred`、`splitData`——方便调试和理解。
- **灵活性**：多模型模式、学习曲线工厂、二维投影等需求难以在标准 Pipeline 中表达。
- **类型边界清晰**：`RunContext` 和 `splitData` 字典提供了明确的状态边界。

### 3.2 为什么使用 Registry 模式

- **声明式配置**：所有流水线在 `pipelines.py` 一个文件中集中声明，一目了然。
- **CLI 解耦**：CLI 通过 `PIPELINE_REGISTRY.get(id)` 获取配置，不依赖 import 路径。
- **零修改扩展**：新增算法只需添加 `PipelineSpec` 条目，无需修改 CLI、Executor 或 Runner。

### 3.3 多模型模式（multiModel）

仅正则化回归使用。训练函数返回 `dict[str, 模型]` 时：

1. Runner 检测 `metadata["multiModel"] == True` → 进入多模型分支。
2. 循环 `models.items()` → 每个模型独立预测、评估、生成独立产物文件。
3. 各模型使用不同的输出子目录（`resolveOutputDir(modelName)`）。

### 3.4 学习曲线工厂

学习曲线内部的交叉验证需要**未训练的模型实例**——每次 CV 折必须从头训练。

`_buildLearningCurveFactory()` 返回 `lambda: SomeModel(...)`，确保每次调用都创建全新实例，避免状态污染。

### 3.5 训练函数签名不强制统一

不同模型的构造器参数差异很大（如 `SVR` 不需要 `random_state`，`Lasso` 需要）。`callTrainer()` 通过 `inspect.signature` 动态过滤关键字参数：

```python
def callTrainer(trainer, *args, **kwargs):
    signature = inspect.signature(trainer)
    accepted = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return trainer(*args, **accepted)
```

这样 `randomState` 只传给需要它的训练函数，不会因多余关键字参数而报错。

### 3.6 数据层不负责标准化

数据层只返回原始 DataFrame——标准化由运行器层的 `applyPreprocessor()` 执行。理由：

- 数据层的数据可被多个流水线复用（不同流水线可能需要不同的预处理）。
- 标准化发生在切分**之后**——`fit_transform` 仅用于训练集，`transform` 用于测试集。
- 保持数据工厂的纯粹性——只关心数据来源，不关心数据如何被使用。
