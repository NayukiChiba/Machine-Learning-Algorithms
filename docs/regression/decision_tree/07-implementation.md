---
title: 决策树回归 — 工程实现
outline: deep
---

# 工程实现

## 本章目标

1. 理解决策树回归流水线的模块分层——数据层、训练层、流水线注册层、运行器层和可视化层。
2. 理清从命令行入口到四类结果图落盘的完整调用链。
3. 理解决策树回归与线性回归、SVR 在工程实现上的关键差异——无标准化、树结构图、特征重要性。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `RegressionDatasetFactory` | 类 | 数据工厂——`loadDecisionTreeRegressionDataset()` 加载 California Housing |
| `trainDecisionTreeRegressionModel(...)` | 函数 | 构建并训练 `DecisionTreeRegressor`——回归树训练的唯一入口 |
| `PipelineSpec` | 数据类 | 声明式流水线配置——关联数据集、训练器、预处理、可视化 |
| `RegressionRunner` | 类 | 回归流水线运行器——读取 `PipelineSpec`，依次执行数据加载、预处理、训练、评估 |
| `plot_residuals(...)` | 函数 | 残差图绘制 |
| `plot_feature_importance(...)` | 函数 | 特征重要性图绘制 |
| `plot_learning_curve(...)` | 函数 | 学习曲线绘制 |
| `plot_tree_structure(...)` | 函数 | 树结构图绘制 |

## 1. 模块分层总览

### 参数速览

| 层 | 文件 | 职责 | 输出 |
|---|---|---|---|
| 数据层 | `src/mlAlgorithms/datasets/tabular/regressionDatasets.py` | 调用 `fetch_california_housing`，标签列重命名为 `price` | `DataFrame`，形状 `(20640, 9)` |
| 数据目录层 | `src/mlAlgorithms/datasets/datasetCatalog.py` | `DatasetSpec("regression.decision_tree", ...)`——注册数据集描述与加载器 | 数据集元信息 |
| 训练层 | `src/mlAlgorithms/training/regression/regressionModels.py` | `trainDecisionTreeRegressionModel(...)`——构建并训练 `DecisionTreeRegressor` | `DecisionTreeRegressor` 模型对象 |
| 流水线注册层 | `src/mlAlgorithms/catalog/pipelines.py` | `PipelineSpec("regression.decision_tree", ...)`——关联所有组件 | 流水线配置 |
| 运行器层 | `src/mlAlgorithms/workflows/regressionRunner.py` | 读取 PipelineSpec → 加载数据 → 预处理 → 训练 → 评估 → 可视化 | 终端日志 + 图像文件 |
| 可视化层 | `src/mlAlgorithms/visualization/` | 绘制残差图、特征重要性图、学习曲线、树结构图 | PNG 图像文件 |

### 理解重点

- 当前代码库采用**声明式流水线**架构——`PipelineSpec` 描述"要用什么数据、什么模型、什么预处理、什么可视化"，运行器根据 Spec 执行。
- 决策树回归的预处理为 `None`——在 `PipelineSpec` 中 scaler 位置为空。这是正确的：树模型不需要标准化。
- 诊断可视化列表为 `["featureImportance"]` + `["learningCurve", "treeStructure"]`——比其他回归模型多出特征重要性和树结构两项。

## 2. `PipelineSpec` 配置详情

### 参数速览

```python
PipelineSpec(
    "regression.decision_tree",        # pipeline ID
    TaskType.REGRESSION,               # 任务类型
    "regression.decision_tree",        # dataset ID
    RunnerType.REGRESSION,             # 运行器类型
    trainDecisionTreeRegressionModel,  # 训练函数
    None,                              # 预处理 —— 无标准化
    "randomSplit",                     # 切分策略
    "default",                         # 后处理
    "regression",                      # 输出目录前缀
    "regression",                      # 可视化目录前缀
    ["correlationHeatmap", "featureTargetScatter"],  # 训练前可视化
    ["featureImportance"],             # 训练后诊断可视化
    ["learningCurve", "treeStructure"],# 学习可视化
    "decision_tree_regression",        # 结果存储子目录
)
```

### 理解重点

- `None` scaler 是决策树回归与其他回归模型的关键工程差异——线性回归和 SVR 使用 `"standardScaler"`。
- 后处理 `"default"` 表示不进行特征选择或降维等额外处理——直接使用全部 8 个特征。
- `"treeStructure"` 是决策树独有的学习可视化——分类决策树也有，但其他回归模型（线性回归、SVR）没有。

## 3. 数据依赖关系

```
fetch_california_housing(as_frame=True)
    │
    ├─→ X = data.drop(columns=["price"])
    ├─→ y = data["price"]
    ├─→ feature_names = list(X.columns)
    │
    ├─→ train_test_split(test_size=0.2)
    │       │
    │       ├─→ X_train, y_train ──→ model.fit() ──→ model
    │       │       │
    │       │       └─→ plot_learning_curve(new DecisionTreeRegressor(), X_train, y_train)
    │       │
    │       └─→ X_test ──→ model.predict() ──→ y_pred
    │               │
    │               └─→ plot_residuals(y_test, y_pred)
    │
    ├─→ model ──→ plot_feature_importance(model, feature_names)
    ├─→ model ──→ plot_tree_structure(model, feature_names)
    │
    └─→ feature_names ──→ 特征重要性图 + 树结构图
```

### 理解重点

- `y_test` 仅用于评估——不参与训练，只在残差图中与 `y_pred` 对比。
- `feature_names` 是关键中间变量——必须在切分前从 `X.columns` 保存，因为后续特征重要性图和树结构图都需要特征名。
- 学习曲线使用**独立**的 `DecisionTreeRegressor(...)` 实例——不共享已训练 `model` 的状态。

## 4. 运行器层的执行链

### 参数速览

| 序号 | 步骤 | 说明 |
|---|---|---|
| 1 | 根据 `datasetId` 查找 `DatasetSpec` | 获取数据加载器和描述信息 |
| 2 | 调用 `loadDecisionTreeRegressionDataset()` | 加载 `(20640, 9)` DataFrame |
| 3 | 拆分 X / y + 保存 `feature_names` | 为后续可视化作准备 |
| 4 | `train_test_split(test_size=0.2)` | 随机切分——无标准化 |
| 5 | 调用 `trainDecisionTreeRegressionModel(X_train, y_train)` | CART 递归分裂训练 |
| 6 | `model.predict(X_test)` | 获取测试集预测值 |
| 7 | `plot_residuals(y_test, y_pred)` | 残差诊断图 |
| 8 | `plot_feature_importance(model, feature_names)` | 特征贡献图 |
| 9 | `plot_learning_curve(new DecisionTreeRegressor(...), X_train, y_train, scoring="r2")` | 学习曲线 |
| 10 | `plot_tree_structure(model, feature_names)` | 树结构可视化 |

### 理解重点

- 运行器层是纯粹的**编排者**——不自己造数据、不实现模型、不画图，只按顺序调用各层组件。
- 步骤 9 的学习曲线创建了一个新的 `DecisionTreeRegressor` 实例（使用 `_buildLearningCurveFactory` 工厂函数），参数与主训练模型一致。
- 步骤 10 的树结构图是本流水线的特有步骤——其他回归流水线（线性回归、SVR）没有这一输出。

## 5. 决策树回归 vs 线性回归 vs SVR 工程对比

| 工程维度 | 线性回归 | SVR | 决策树回归 |
|---|---|---|---|
| 训练函数 | `trainLinearRegressionModel` | `trainSvrRegressionModel` | **`trainDecisionTreeRegressionModel`** |
| 模型类 | `LinearRegression` | `SVR` | **`DecisionTreeRegressor`** |
| 预处理 | `standardScaler` | `standardScaler` | **`None`** |
| 切分策略 | `randomSplit` | `randomSplit` | `randomSplit` |
| 训练后诊断 | `["featureImportance"]` | `[]` | **`["featureImportance"]`** |
| 学习可视化 | `["learningCurve"]` | `["learningCurve"]` | **`["learningCurve", "treeStructure"]`** |
| 数据量 | 200（手动合成） | 200（`make_friedman1`） | **20640（真实数据）** |
| 超参数数 | 0 | 4 | **3** |

## 常见坑

1. 误以为运行器层直接导入可视化函数——实际上是通过诊断/学习可视化列表配置，由运行器根据列表动态调用。
2. 找不到决策树特有的 `treeStructure` 可视化在其他回归模型中出现——它是树模型独有的结构可视化。
3. 混淆 `PipelineSpec` 中的多个 ID 字段——`pipelineId`、`datasetId`、结果目录名是三个不同的标识符。

## 小结

- 决策树回归工程实现采用声明式流水线架构——`PipelineSpec` 配置所有组件，`RegressionRunner` 按序编排执行。
- 与线性回归/SVR 的四个关键工程差异：（1）预处理为 `None`；（2）多了 `treeStructure` 可视化；（3）数据规模大一个量级；（4）超参数从系数/支持向量变为树深度/叶子约束。
- 数据依赖图的核心节点：`feature_names`（贯穿重要性图和结构图）、`model`（分裂结果）、`y_pred`（残差图）。
