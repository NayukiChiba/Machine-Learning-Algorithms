---
title: 项目架构 — 核心抽象
outline: deep
---

# 核心抽象

## 本章目标

1. 理解 `PipelineSpec` 的全部 16 个字段及其配置方式。
2. 理解 `DatasetSpec`、`Registry`、`RunContext`、`RunResult` 的职责。
3. 掌握 `TaskType`、`RunnerType`、`DataKind` 三种枚举。

---

## 1. PipelineSpec —— 流水线声明

`PipelineSpec` 是本项目最核心的数据类。一个实例完整描述一条可执行算法流水线——数据来源、训练函数、预处理、评估和可视化全部在此声明。

**定义**：[`src/mlAlgorithms/core/pipelineSpec.py`](../src/mlAlgorithms/core/pipelineSpec.py)

### 1.1 字段速览

| 字段 | 类型 | 说明 | 示例 |
|---|---|---|---|
| `id` | `str` | 流水线唯一标识，格式 `{domain}.{algorithm}` | `"regression.svr"` |
| `taskType` | `TaskType` | 任务类型——决定数据探索报告的格式 | `TaskType.REGRESSION` |
| `datasetId` | `str` | 关联的数据集 ID——必须与 `DatasetSpec.id` 一致 | `"regression.svr"` |
| `runnerType` | `RunnerType` | 运行器类型——`executor.py` 据此分发 | `RunnerType.REGRESSION` |
| `trainer` | `Callable` | 训练函数——接收训练数据，返回模型（或 `dict`） | `trainSvrRegressionModel` |
| `preprocessor` | `str \| None` | 预处理方式——`"standardScaler"` 或 `None` | `"standardScaler"` |
| `splitter` | `str \| None` | 切分策略——`"randomSplit"` / `"stratifiedSplit"` / `None` | `"randomSplit"` |
| `predictor` | `str \| None` | 后处理策略——分类/聚类/LDA 特有 | `"default"` |
| `evaluator` | `str \| None` | 评估配置名称 | `"default"` |
| `analysisProfile` | `str` | 分析报告类型——按 TaskType 选择合适的分析器 | `"regression"` |
| `dataPlots` | `list[str]` | 训练前数据可视化列表 | `["correlationHeatmap"]` |
| `resultPlots` | `list[str]` | 训练后结果可视化列表 | `["featureImportance"]` |
| `diagnostics` | `list[str]` | 诊断性可视化列表 | `["learningCurve"]` |
| `outputKey` | `str` | 输出子目录名——产物存放到 `outputs/{outputKey}/` | `"svr"` |
| `optionalDependencies` | `tuple[str]` | 可选依赖——缺失时跳过而非崩溃 | `("hmmlearn",)` |
| `metadata` | `dict` | 额外配置——多模型标记、工厂函数等 | `{"multiModel": True}` |

### 1.2 注册示例

```python
PipelineSpec(
    "regression.linear_regression",    # pipeline ID
    TaskType.REGRESSION,               # 任务类型
    "regression.linear_regression",    # dataset ID
    RunnerType.REGRESSION,             # 运行器类型
    trainLinearRegressionModel,        # 训练函数
    None,                               # 预处理 — 无标准化
    "randomSplit",                      # 切分策略
    "default",                          # 后处理
    "regression",                       # analysisProfile
    "regression",                       # evaluator
    ["correlationHeatmap", "featureTargetScatter"],  # dataPlots
    ["featureImportance"],              # resultPlots
    ["learningCurve"],                  # diagnostics
    "linear_regression",                # outputKey
    metadata={
        "learningCurveEstimatorFactory": _buildLearningCurveFactory(
            "regression.linear_regression"
        )
    },
)
```

### 1.3 理解重点

- 一个 `PipelineSpec` 就是一个算法的**完整配置清单**——Runner 不需要任何额外信息即可执行。
- `preprocessor`、`dataPlots`、`resultPlots`、`diagnostics` 都是**声明式列表**——Runner 遍历列表逐项执行，新增可视化只需在列表中添加名称。
- `metadata` 是扩展点——`multiModel`、`learningCurveEstimatorFactory`、`visualModelFactory` 等特殊需求都通过它传递。

---

## 2. DatasetSpec —— 数据集声明

描述一个数据集的加载方式与元信息。

**定义**：[`src/mlAlgorithms/core/datasetSpec.py`](../src/mlAlgorithms/core/datasetSpec.py)

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | `str` | 数据集唯一标识 |
| `taskType` | `TaskType` | 所属任务类型 |
| `dataKind` | `DataKind` | 数据形态——`TABULAR` 或 `SEQUENCE` |
| `loader` | `Callable[[], DataFrame]` | 数据加载函数——每次调用返回新 DataFrame |
| `targetColumn` | `str \| None` | 标签列名 |
| `featureColumns` | `list[str] \| None` | 手动指定特征列——`None` 时自动从 targetColumn 推断 |
| `description` | `str` | 数据集中文描述 |

**关键方法**：

| 方法 | 说明 |
|---|---|
| `load()` | 调用 `loader()` 返回新 DataFrame |
| `resolveFeatureColumns(data)` | 根据 `featureColumns` / `targetColumn` 解析特征列名 |

### 理解重点

- `loader` 每次调用返回**全新** DataFrame——避免多次运行之间的状态污染。
- `featureColumns=None` 时自动推断：排除 `targetColumn` 外的所有列即为特征列。
- `dataKind=SEQUENCE` 仅用于 HMM——影响数据探索报告的生成方式。

---

## 3. Registry —— 简单注册表

基于字典的泛型注册表，是 `PIPELINE_REGISTRY` 和 `DATASET_REGISTRY` 的底层实现。

**定义**：[`src/mlAlgorithms/core/registry.py`](../src/mlAlgorithms/core/registry.py)

| 方法 | 说明 |
|---|---|
| `register(itemId, item)` | 注册对象——重复 ID 抛出 `KeyError` |
| `get(itemId)` | 获取对象——未注册抛出 `KeyError` |
| `keys()` | 返回所有已注册 ID |
| `values()` | 返回所有已注册对象 |
| `contains(itemId)` | 判断条目是否已注册 |

### 理解重点

- `Registry` 是泛型类——`Registry[PipelineSpec]` 和 `Registry[DatasetSpec]` 共享同一实现。
- 两条 `Registry` 在模块导入时完成注册——CLI 启动时即可直接查询。

---

## 4. RunContext —— 运行时上下文

一次流水线运行的共享状态容器——贯穿 Runner 的整个执行周期。

**定义**：[`src/mlAlgorithms/core/runContext.py`](../src/mlAlgorithms/core/runContext.py)

| 字段 | 类型 | 说明 |
|---|---|---|
| `spec` | `PipelineSpec` | 当前执行的流水线声明 |
| `datasetSpec` | `DatasetSpec` | 关联的数据集声明 |
| `data` | `DataFrame` | 完整原始数据 |
| `features` | `DataFrame \| None` | 特征列子集 |
| `target` | `Series \| None` | 标签列——无监督任务为 `None` |
| `outputDir` | `Path` | 产物输出目录 |
| `randomState` | `int` | 全局随机种子（42） |
| `analysisReport` | `Any \| None` | 数据探索报告——`runAnalysis()` 填充 |
| `extras` | `dict` | 扩展字段——Runner 间传递额外数据 |

### 理解重点

- `RunContext` 由 `buildRunContext()` 创建——加载数据、解析特征/标签、创建输出目录。
- `analysisReport` 在 Runner 执行早期填充——后续步骤可访问探索结果。

---

## 5. RunResult —— 运行结果

一次流水线执行的产物容器。

**定义**：[`src/mlAlgorithms/core/runResult.py`](../src/mlAlgorithms/core/runResult.py)

| 字段 | 类型 | 说明 |
|---|---|---|
| `model` | `Any` | 训练完成的模型——或 `dict[str, 模型]`（多模型模式） |
| `predictions` | `Any \| None` | 预测值数组 |
| `scores` | `Any \| None` | 预测分数（`predict_proba` / `decision_function`） |
| `metrics` | `dict` | 评估指标字典 |
| `artifacts` | `list[Path]` | 产物文件路径列表（PNG 图像等） |
| `extras` | `dict` | 扩展字段 |

### 理解重点

- `artifacts` 逐步累积——每生成一张图就 `appendArtifact()` 追加。
- 多模型模式下 `model` 是 `dict`，`metrics` 的键与模型名对应。

---

## 6. 枚举类型

**定义**：[`src/mlAlgorithms/core/taskTypes.py`](../src/mlAlgorithms/core/taskTypes.py)

### 6.1 TaskType（任务类型）

决定数据探索报告的生成方式和算法的领域归属。

| 枚举值 | 含义 | 包含算法 |
|---|---|---|
| `CLASSIFICATION` | 分类 | 逻辑回归、决策树、SVC、朴素贝叶斯、KNN、随机森林、Bagging、GBDT、LightGBM |
| `REGRESSION` | 回归 | 线性回归、SVR、决策树回归、正则化回归、XGBoost |
| `CLUSTERING` | 聚类 | KMeans、DBSCAN |
| `DIMENSIONALITY` | 降维 | PCA、LDA |
| `PROBABILISTIC` | 概率模型 | GMM（EM）、HMM |

### 6.2 RunnerType（运行器类型）

与 `TaskType` 值一一对应。`executor.py` 根据它分发到对应的 Runner 函数：

```
RunnerType.CLASSIFICATION   → runClassificationPipeline()
RunnerType.REGRESSION       → runRegressionPipeline()
RunnerType.CLUSTERING       → runClusteringPipeline()
RunnerType.DIMENSIONALITY   → runDimensionalityPipeline()
RunnerType.PROBABILISTIC    → runProbabilisticPipeline()
```

### 6.3 DataKind（数据形态）

| 枚举值 | 含义 | 使用场景 |
|---|---|---|
| `TABULAR` | 表格数据——每行一个样本，每列一个特征 | 除 HMM 外的所有算法 |
| `SEQUENCE` | 序列数据——不等长观测序列 | HMM |
