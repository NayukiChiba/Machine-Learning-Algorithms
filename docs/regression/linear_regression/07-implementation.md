---
title: 线性回归 — 工程实现
outline: deep
---

# 工程实现

## 本章目标

1. 理解线性回归流水线的模块分层——数据层、训练层、流水线注册层、运行器层和可视化层。
2. 理清从命令行入口到结果图落盘的完整调用链。
3. 理解线性回归与决策树回归在工程实现上的关键差异——最简训练函数、无标准化、无可视化差异。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `RegressionDatasetFactory` | 类 | 数据工厂——`loadLinearRegressionDataset()` 手工合成线性房价数据 |
| `trainLinearRegressionModel(...)` | 函数 | 构建并训练 `LinearRegression`——本仓库最简训练函数（3 行） |
| `PipelineSpec` | 数据类 | 声明式流水线配置——关联数据集、训练器、预处理、可视化 |
| `RegressionRunner` | 类 | 回归流水线运行器——读取 `PipelineSpec`，依次执行各阶段 |
| `plot_residuals(...)` | 函数 | 残差图绘制 |
| `plot_learning_curve(...)` | 函数 | 学习曲线绘制 |

## 1. 模块分层总览

### 参数速览

| 层 | 文件 | 职责 | 输出 |
|---|---|---|---|
| 数据层 | `src/mlAlgorithms/datasets/tabular/regressionDatasets.py` | `loadLinearRegressionDataset()`——按显式线性公式生成 200 样本 | `DataFrame`，形状 `(200, 4)` |
| 数据目录层 | `src/mlAlgorithms/datasets/datasetCatalog.py` | `DatasetSpec("regression.linear_regression", ...)`——注册数据集描述与加载器 | 数据集元信息 |
| 训练层 | `src/mlAlgorithms/training/regression/regressionModels.py` | `trainLinearRegressionModel(...)`——构建 `LinearRegression()` 并 fit | `LinearRegression` 模型对象 |
| 流水线注册层 | `src/mlAlgorithms/catalog/pipelines.py` | `PipelineSpec("regression.linear_regression", ...)`——关联所有组件 | 流水线配置 |
| 运行器层 | `src/mlAlgorithms/workflows/regressionRunner.py` | 读取 PipelineSpec → 加载数据 → 预处理（无）→ 训练 → 评估 → 可视化 | 终端日志 + 图像文件 |
| 可视化层 | `src/mlAlgorithms/visualization/` | 绘制残差图、学习曲线 | PNG 图像文件 |

### 理解重点

- 线性回归是当前代码库中**工程结构最简**的回归流水线——训练函数仅 3 行，预处理为 `None`，评估可视化仅 2 项。
- 与决策树回归的差异：决策树多了 `featureImportance` 和 `treeStructure` 两项可视化，训练函数更复杂（3 个超参数 + 结构日志）。
- 这种极简设计是有意的——线性回归作为回归学习的起点，工程结构越简单越利于理解核心调用链。

## 2. `PipelineSpec` 配置详情

```python
PipelineSpec(
    "regression.linear_regression",     # pipeline ID
    TaskType.REGRESSION,                # 任务类型
    "regression.linear_regression",     # dataset ID
    RunnerType.REGRESSION,              # 运行器类型
    trainLinearRegressionModel,         # 训练函数
    None,                               # 预处理 —— 无标准化
    "randomSplit",                      # 切分策略
    "default",                          # 后处理
    "regression",                       # 输出目录前缀
    "regression",                       # 可视化目录前缀
    ["correlationHeatmap", "featureTargetScatter"],  # 训练前可视化
    ["featureImportance"],              # 训练后诊断可视化
    ["learningCurve"],                  # 学习可视化
    "linear_regression",                # 结果存储子目录
)
```

### 理解重点

- `None` 预处理是线性回归与其他回归模型（SVR、正则化）的关键工程差异——但需区分：当前选择是因为数据量纲接近且关系简单，不代表所有场景下线性回归都不需要标准化。
- `["featureImportance"]` 在训练后可视化中——对线性回归来说，"特征重要性"即 `coef_` 的绝对值可视化（系数柱状图）。
- `["learningCurve"]` 使用 `_buildLearningCurveFactory` 工厂函数——传入新的 `LinearRegression()` 实例做 CV。

## 3. 数据依赖关系

```
loadLinearRegressionDataset()
    │
    ├─→ X = data.drop(columns=["price"])
    ├─→ y = data["price"]
    ├─→ feature_names = list(X.columns)
    │
    ├─→ train_test_split(test_size=0.2)
    │       │
    │       ├─→ X_train, y_train ──→ model.fit() ──→ model (coef_, intercept_)
    │       │       │
    │       │       └─→ plot_learning_curve(LinearRegression(), X_train, y_train)
    │       │
    │       └─→ X_test ──→ model.predict() ──→ y_pred
    │               │
    │               └─→ plot_residuals(y_test, y_pred)
    │
    └─→ model ──→ 终端日志: coef_ + intercept_ 打印
```

### 理解重点

- 数据依赖图比决策树回归更简单——没有 `featureImportance` 和 `treeStructure` 两条分支。
- `y_test` 仅用于评估对比——不参与训练，只在残差图中与 `y_pred` 对比。
- `model` 的核心输出是 `coef_` 和 `intercept_`——终端日志直接打印，是流水线最核心的训练产物。

## 4. 运行器层的执行链

| 序号 | 步骤 | 说明 |
|---|---|---|
| 1 | 根据 `datasetId` 查找 `DatasetSpec` | 获取数据加载器和描述信息 |
| 2 | 调用 `loadLinearRegressionDataset()` | 加载 `(200, 4)` DataFrame |
| 3 | 拆分 X / y + 保存 `feature_names` | 为后续日志和可视化作准备 |
| 4 | `train_test_split(test_size=0.2)` | 随机切分——无标准化 |
| 5 | 调用 `trainLinearRegressionModel(X_train, y_train)` | SVD 闭式求解——打印 `coef_` 和 `intercept_` |
| 6 | `model.predict(X_test)` | 获取测试集预测值 |
| 7 | `plot_residuals(y_test, y_pred)` |  残差诊断图 |
| 8 | `plot_learning_curve(LinearRegression(), X_train, y_train, scoring="r2")` | 学习曲线 |

### 理解重点

- 步骤 5 是本仓库最短的训练步骤——3 行代码，SVD 闭式解，瞬间完成，无迭代日志。
- 步骤 8 的学习曲线使用新 `LinearRegression()` 实例——`_buildLearningCurveFactory("regression.linear_regression")` 返回的工厂函数。
- 与决策树回归的执行链对比：少了 `plot_feature_importance` 和 `plot_tree_structure` 两个步骤。

## 5. 线性回归 vs 决策树回归 vs SVR 工程对比

| 工程维度 | 线性回归 | 决策树回归 | SVR |
|---|---|---|---|
| 训练函数 | `trainLinearRegressionModel` | `trainDecisionTreeRegressionModel` | `trainSvrRegressionModel` |
| 模型类 | `LinearRegression` | `DecisionTreeRegressor` | `SVR` |
| 训练函数行数 | **3 行** | ~5 行 | ~4 行 |
| 预处理 | `None` | `None` | `standardScaler` |
| 超参数数 | 0 | 3 | 4 |
| 训练后诊断 | `["featureImportance"]` | `["featureImportance"]` | `[]` |
| 学习可视化 | `["learningCurve"]` | `["learningCurve", "treeStructure"]` | `["learningCurve"]` |
| 数据量 | 200（手工合成） | 20640（真实数据） | 200（合成非线性） |
| 训练方式 | SVD 闭式解 | CART 贪心递归 | 凸优化（SMO 类算法） |

## 常见坑

1. 误以为运行器层直接导入可视化函数——实际上是通过诊断/学习可视化列表配置，由运行器根据列表动态调用。
2. 将 `PipelineSpec` 中的旧路径引用（如 `data_generation/`）当成当前代码库的实际结构——实际代码在 `src/mlAlgorithms/` 下。
3. 把 `trainLinearRegressionModel` 的极简实现误解为功能缺失——3 行代码是因为 `LinearRegression()` 无需超参数，这是设计上的有意简洁。

## 小结

- 线性回归工程实现采用声明式流水线架构——`PipelineSpec` 配置所有组件，`RegressionRunner` 按序编排执行。
- 与决策树回归/SVR 的工程差异：（1）训练函数最简（3 行）；（2）预处理为 `None`（无标准化）；（3）可视化最少（无 treeStructure）；（4）数据为手工合成小规模。
- 这种极简设计使线性回归成为理解整个回归流水线架构的最佳入口——先看懂最简单的，再对比复杂的。
