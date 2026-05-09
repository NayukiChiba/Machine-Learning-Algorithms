---
title: 项目架构 — 模块分层
outline: deep
---

# 模块分层

## 本章目标

1. 理解六层模块的职责边界——数据层、训练层、流水线注册层、运行器层、评估层、可视化层。
2. 理清各层之间的调用关系和数据流向。

---

## 1. 数据层（`datasets/`）

**职责**：生成或加载数据集，返回 `pandas.DataFrame`。不涉及任何预处理或切分。

### 1.1 核心文件

| 文件 | 职责 |
|---|---|
| `datasetCatalog.py` | `buildDatasetSpecs()` 构建全部 20 个 `DatasetSpec` |
| `tabular/classificationDatasets.py` | `ClassificationDatasetFactory` —— 6 个分类数据集 |
| `tabular/regressionDatasets.py` | `RegressionDatasetFactory` —— 4 个回归数据集 |
| `tabular/clusteringDatasets.py` | `ClusteringDatasetFactory` —— 2 个聚类数据集 |
| `tabular/ensembleDatasets.py` | `EnsembleDatasetFactory` —— 4 个集成数据集 |
| `tabular/dimensionalityDatasets.py` | `DimensionalityDatasetFactory` —— 2 个降维数据集 |
| `sequence/probabilisticDatasets.py` | `ProbabilisticDatasetFactory` —— 2 个概率模型数据集 |

### 1.2 数据工厂一览

| 工厂类 | 负责的数据集 |
|---|---|
| `ClassificationDatasetFactory` | 逻辑回归（线性可分二分类）、决策树（blob 多分类）、SVC（同心圆二分类）、朴素贝叶斯（Iris）、KNN（双月牙二分类）、随机森林（高维多分类） |
| `RegressionDatasetFactory` | 线性回归（合成线性房价）、SVR（Friedman1 非线性）、决策树回归（California Housing）、正则化回归（diabetes + 共线 + 噪声） |
| `ClusteringDatasetFactory` | KMeans（球形多簇）、DBSCAN（双月牙非线性） |
| `EnsembleDatasetFactory` | Bagging（高噪声双月牙）、GBDT（中等难度多分类）、XGBoost（回归）、LightGBM（高维四分类） |
| `DimensionalityDatasetFactory` | PCA（高维低秩合成）、LDA（Wine） |
| `ProbabilisticDatasetFactory` | EM/GMM（混合高斯）、HMM（离散序列） |

### 1.3 关键设计

- 每个 `load*Dataset()` 方法独立生成数据——**无共享状态**，无全局变量。
- 数据在 `DatasetSpec.load()` 时才会实际生成——**惰性加载**。
- 所有数据方法使用 `random_state` 参数保证**可复现性**。

---

## 2. 训练层（`training/`）

**职责**：封装 scikit-learn（及 hmmlearn / xgboost / lightgbm）模型的构建与 `fit()` 调用。**不包含预处理、切分和评估**。

### 2.1 核心文件

| 文件 | 包含的训练函数 |
|---|---|
| `classification/classificationModels.py` | 逻辑回归、决策树、SVC、朴素贝叶斯、KNN、随机森林、Bagging、GBDT、LightGBM |
| `regression/regressionModels.py` | 线性回归、SVR、决策树回归、正则化回归（Lasso/Ridge/ElasticNet）、XGBoost |
| `clustering/clusteringModels.py` | KMeans、DBSCAN |
| `dimensionality/dimensionalityModels.py` | PCA、LDA |
| `probabilistic/probabilisticModels.py` | GMM（EM）、HMM |

### 2.2 训练函数一览（回归示例）

| 函数 | 模型类 | 核心超参数 | 行数 | 特殊性 |
|---|---|---|---|---|
| `trainLinearRegressionModel` | `LinearRegression` | 无 | 3 | 无参构造 |
| `trainSvrRegressionModel` | `SVR` | `C=10.0, epsilon=0.1, kernel='rbf', gamma='scale'` | 2 | 本仓库最短 |
| `trainDecisionTreeRegressionModel` | `DecisionTreeRegressor` | `max_depth=6, min_samples_split=6, min_samples_leaf=3` | ~5 | — |
| `trainRegularizationModels` | `Lasso / Ridge / ElasticNet` | `alpha=0.15/2.0/0.2, l1_ratio=0.5` | ~10 | 返回 `dict` |
| `trainXgboostRegressionModel` | `XGBRegressor` | `n_estimators=300, learning_rate=0.05, ...` | ~15 | 可选依赖 |

### 2.3 关键设计

- 训练函数是**薄封装**——仅构建 + `fit()`，不做数据预处理或评估。
- 签名不强制统一——`callTrainer()` 通过 `inspect.signature` 按需过滤关键字参数。
- 正则化回归的训练函数返回 `dict[str, 模型]`——触发 Runner 的多模型模式。

---

## 3. 流水线注册层（`catalog/`）

**职责**：将所有 `PipelineSpec` 和 `DatasetSpec` 集中注册到全局 `Registry`。

### 3.1 核心文件

| 文件 | 职责 |
|---|---|
| `pipelines.py` | 声明全部 20 条 `PipelineSpec` + 工厂函数 |
| `datasets.py` | 调用 `buildDatasetSpecs()` 注册全部 `DatasetSpec` |

### 3.2 工厂函数

| 工厂 | 用途 | 说明 |
|---|---|---|
| `_buildLearningCurveFactory(pipelineId)` | 学习曲线 | 返回创建**未训练**模型实例的 lambda——供 CV 内部多次 fit |
| `_buildVisualModelFactory(pipelineId)` | 决策边界可视化 | 返回创建模型实例的 lambda——供二维投影上的边界拟合 |

### 3.3 关键设计

- 两条 `Registry` 是**全局单例**——CLI 入口通过它们查找流水线和数据集。
- 工厂函数将模型构造参数固化在 lambda 中——确保学习曲线和可视化使用的超参数与训练一致。
- 新增算法的 `PipelineSpec` 直接追加到 `PIPELINE_REGISTRY` 的注册循环中。

---

## 4. 运行器层（`workflows/`）

**职责**：编排一次完整流水线执行——从数据加载到产物输出。是项目的**执行核心**。

### 4.1 核心文件

| 文件 | 职责 |
|---|---|
| `executor.py` | 按 `RunnerType` 分发到对应的 Runner 函数 |
| `baseRunner.py` | 共享辅助函数——数据加载、切分、标准化、训练调用、二维投影等 |
| `classificationRunner.py` | 分类流水线——含决策边界和 ROC 逻辑 |
| `regressionRunner.py` | 回归流水线——含多模型模式和学习曲线逻辑 |
| `clusteringRunner.py` | 聚类流水线——含标签对齐和参数扫描逻辑 |
| `dimensionalityRunner.py` | 降维流水线——含 PCA 训练曲线和 LDA 分类评估 |
| `probabilisticRunner.py` | 概率模型流水线——含 GMM 分量扫描和 HMM 解码 |

### 4.2 执行链（以回归为例）

```
buildRunContext()              # 加载数据 → 解析特征/标签 → 创建输出目录
  └─ runAnalysis()             #   数据探索 → 终端打印报告
  └─ [dataPlots] 遍历          #   训练前可视化（相关性热力图等）
  └─ makeSplit()               #   切分训练集/测试集
  └─ applyPreprocessor()       #   标准化（如配置了 standardScaler）
  └─ callTrainer()             #   训练 → 返回模型
  └─ predict()                 #   预测
  └─ evaluate()                #   评估 + 打印指标
  └─ [resultPlots] 遍历        #   训练后可视化（特征重要性等）
  └─ [diagnostics] 遍历        #   诊断可视化（学习曲线等）
```

### 4.3 baseRunner 关键函数

| 函数 | 作用 |
|---|---|
| `buildRunContext(spec, datasetSpec)` | 加载数据、解析特征/标签、创建输出目录 → 返回 `RunContext` |
| `runAnalysis(context)` | 按 `TaskType` / `DataKind` 构建并打印数据探索报告 |
| `makeSplit(X, y, splitter, randomState)` | 切分数据——支持 `randomSplit`、`stratifiedSplit`、`None` |
| `applyPreprocessor(splitData, preprocessor)` | 仅在 `standardScaler` 时执行 `fit_transform` / `transform` |
| `callTrainer(trainer, *args, **kwargs)` | 调用训练函数——自动按签名过滤关键字参数 |
| `prepare2dProjection(...)` | 为二维可视化准备 PCA 投影 + 边界模型拟合 |
| `collectScoreOutput(model, XTestProcessed)` | 收集 `predict_proba` 或 `decision_function` 输出 |
| `makeRunResult(model, predictions, scores, metrics)` | 构建 `RunResult` |
| `appendArtifact(result, artifact)` | 向 `RunResult` 追加产物文件路径 |

### 4.4 多模型模式

当 `PipelineSpec.metadata["multiModel"] == True` 时（仅正则化回归）：

1. Runner 将训练返回值视为 `dict[str, 模型]`。
2. 循环 `models.items()`——每个模型独立预测、评估、生成产物。
3. 各模型使用不同的输出子目录（通过 `resolveOutputDir(modelName)`）。

---

## 5. 评估层（`evaluation/`）

**职责**：计算并打印模型评估指标到终端。

| 文件 | 支持指标 |
|---|---|
| `classificationEvaluator.py` | Accuracy、Precision、Recall、F1、ROC-AUC |
| `regressionEvaluator.py` | R²、MSE、RMSE、MAE、Explained Variance |
| `clusteringEvaluator.py` | Silhouette、Davies-Bouldin、Calinski-Harabasz |
| `dimensionalityEvaluator.py` | 累计解释方差比（PCA） |
| `sequenceEvaluator.py` | HMM 评分 |

---

## 6. 可视化层（`visualization/`）

**职责**：绘制并保存各类图表。分为训练前（`data/`）和训练后（`result/`）两个子模块。

| 子模块 | 包含图表 |
|---|---|
| `data/` | 分类分布、特征空间散点图、相关性热力图 |
| `result/` 分类 | 混淆矩阵、ROC 曲线、特征重要性、决策边界、分类结果展示 |
| `result/` 回归 | 残差图、回归结果展示、学习曲线、特征重要性、树结构 |
| `result/` 聚类 | 聚类结果散点图、KMeans 惯性曲线、DBSCAN k-距离图 / ε 扫描 |
| `result/` 降维 | PCA 训练曲线 |
| `result/` 序列 | HMM 状态解码图 |
