---
title: SVR 支持向量回归 — 工程实现
outline: deep
---

# 工程实现

## 本章目标

1. 理解 SVR 流水线的模块分层——数据层、训练层、流水线注册层、运行器层和可视化层。
2. 理清从命令行入口到结果图落盘的完整调用链。
3. 理解 SVR 与线性回归、正则化回归在工程实现上的关键差异——标准化 + 学习曲线 + 无特征重要性。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `RegressionDatasetFactory` | 类 | 数据工厂——`loadSvrDataset()` 生成 Friedman1 非线性数据 |
| `trainSvrRegressionModel(...)` | 函数 | 构建并训练 `SVR`——本仓库最短训练函数（2 行） |
| `PipelineSpec` | 数据类 | 声明式流水线配置——训练后诊断列表为 `[]` |
| `RegressionRunner` | 类 | 回归流水线运行器——读取 `PipelineSpec`，依次执行各阶段 |
| `plot_residuals(...)` | 函数 | 残差图绘制 |
| `plot_learning_curve(...)` | 函数 | 学习曲线绘制——使用 `learningCurveEstimatorFactory` 工厂 |

## 1. 模块分层总览

### 参数速览

| 层 | 文件 | 职责 | 输出 |
|---|---|---|---|
| 数据层 | `src/mlAlgorithms/datasets/tabular/regressionDatasets.py` | `loadSvrDataset()`——调用 `make_friedman1` 生成非线性数据 | `DataFrame`，形状 `(200, 11)` |
| 数据目录层 | `src/mlAlgorithms/datasets/datasetCatalog.py` | `DatasetSpec("regression.svr", ...)`——注册数据集描述与加载器 | 数据集元信息 |
| 训练层 | `src/mlAlgorithms/training/regression/regressionModels.py` | `trainSvrRegressionModel(...)`——构建 `SVR(C=10.0, epsilon=0.1, kernel='rbf', gamma='scale')` 并 fit | `SVR` 模型对象 |
| 流水线注册层 | `src/mlAlgorithms/catalog/pipelines.py` | `PipelineSpec("regression.svr", ...)`——关联所有组件 | 流水线配置 |
| 运行器层 | `src/mlAlgorithms/workflows/regressionRunner.py` | 读取 PipelineSpec → 加载 → 标准化 → 训练 → 评估 → 可视化 | 终端日志 + 图像文件 |
| 可视化层 | `src/mlAlgorithms/visualization/` | 绘制残差图、学习曲线 | PNG 图像文件 |

### 理解重点

- SVR 的工程结构与线性回归几乎相同——都是单模型、单次训练、两组可视化。差异在于 SVR 多了标准化步骤。
- SVR 与正则化回归的工程结构差异明显——正则化回归是多模型（`multiModel=True`）+ 无学习曲线，SVR 是单模型 + 有学习曲线。
- 训练函数仅 2 行——比线性回归（3 行）更短，因为所有超参数在构造器中一次性全部给出。

## 2. `PipelineSpec` 配置详情

```python
PipelineSpec(
    "regression.svr",                      # pipeline ID
    TaskType.REGRESSION,                   # 任务类型
    "regression.svr",                      # dataset ID
    RunnerType.REGRESSION,                 # 运行器类型
    trainSvrRegressionModel,               # 训练函数——2 行封装
    "standardScaler",                      # 预处理——RBF 核强制标准化
    "randomSplit",                         # 切分策略
    "default",                             # 后处理
    "regression",                          # 输出目录前缀
    "regression",                          # 可视化目录前缀
    ["correlationHeatmap", "featureTargetScatter"],  # 训练前可视化
    [],                                    # 训练后诊断可视化——空列表！（RBF 核无特征重要性）
    ["learningCurve"],                     # 学习可视化
    "svr",                                 # 结果存储子目录
    metadata={
        "learningCurveEstimatorFactory": _buildLearningCurveFactory(
            "regression.svr"               # → SVR(C=10.0, epsilon=0.1, kernel='rbf', gamma='scale')
        )
    },
)
```

### 理解重点

- `[]` 训练后诊断列表为空——这是 SVR（RBF 核）与线性回归/正则化回归在工程层面最显著的区别。RBF 核的 SVR 无法输出 `coef_` 或 `feature_importances_`。
- `"learningCurve"` 在学可视化列表中——SVR 有学习曲线（与正则化回归不同）。
- `learningCurveEstimatorFactory` 确保学习曲线使用与训练一致的超参数——`SVR(C=10.0, epsilon=0.1, kernel='rbf', gamma='scale')`。
- `"standardScaler"` 预处理——SVR 和正则化回归都需要标准化，线性回归和决策树不需要。

## 3. 数据依赖关系

```
loadSvrDataset()
    │
    ├─→ X = data.drop(columns=["price"])
    ├─→ y = data["price"]
    ├─→ feature_names = list(X.columns)
    │
    ├─→ train_test_split(test_size=0.2)
    │       │
    │       ├─→ StandardScaler().fit_transform(X_train) ──→ X_train_s
    │       ├─→ StandardScaler().transform(X_test) ──→ X_test_s
    │       │
    │       ├─→ trainSvrRegressionModel(X_train_s, y_train)
    │       │       │
    │       │       └─→ model (support_, dual_coef_, intercept_)
    │       │
    │       ├─→ y_pred = model.predict(X_test_s)
    │       │       │
    │       │       └─→ plot_residuals(y_test, y_pred)
    │       │
    │       └─→ plot_learning_curve(SVR(...), X_train_s, y_train, scoring="r2")
```

### 理解重点

- 数据依赖图与正则化回归的差异：SVR 没有多模型循环，没有 `featureImportance` 分支。
- 标准化分支在切分之后——`fit_transform` 用于训练集，`transform` 用于测试集。
- 学习曲线分支使用工厂函数创建的新 `SVR(...)` 实例——而非已训练的 `model`。
- `model.support_` 是训练的核心产物——终端日志直接打印其数量。

## 4. 运行器层的执行链

| 序号 | 步骤 | 说明 |
|---|---|---|
| 1 | 根据 `datasetId` 查找 `DatasetSpec` | 获取数据加载器和描述信息 |
| 2 | 调用 `loadSvrDataset()` | 加载 `(200, 11)` DataFrame |
| 3 | 拆分 X / y + 保存 `feature_names` | 为后续日志和可视化作准备 |
| 4 | `train_test_split(test_size=0.2)` | 随机切分 |
| 5 | `StandardScaler().fit_transform(X_train)` | 标准化训练集——RBF 核必需 |
| 6 | `StandardScaler().transform(X_test)` | 标准化测试集 |
| 7 | 调用 `trainSvrRegressionModel(X_train_s, y_train)` | SMO 求解——打印支持向量数量 |
| 8 | `model.predict(X_test_s)` | 核函数加权求和预测 |
| 9 | `plot_residuals(y_test, y_pred)` | 残差诊断图 |
| 10 | `plot_learning_curve(SVR(...), X_train_s, y_train, scoring="r2")` | 学习曲线——使用工厂创建的新实例 |

### 理解重点

- 步骤 7 的训练耗时取决于 SMO 收敛速度——200 样本上几乎瞬时，但数据量大时是瓶颈。
- 步骤 8 的预测复杂度与支持向量数成正比——`nSV` 越多预测越慢。
- 与线性回归对比：多了标准化（5-6），少了 `coef_` 打印（RBF 核没有 `coef_`）。
- 与正则化回归对比：少了多模型循环，多了学习曲线（10）。

## 5. SVR vs 线性回归 vs 正则化回归 工程对比

| 工程维度 | 线性回归 | 正则化回归 | SVR |
|---|---|---|---|
| 训练函数 | `trainLinearRegressionModel` | `trainRegularizationModels` | **`trainSvrRegressionModel`** |
| 模型类 | `LinearRegression` | `Lasso`, `Ridge`, `ElasticNet` | **`SVR`** |
| 训练函数行数 | 3 行 | ~10 行 | **2 行——最简** |
| 预处理 | `None` | **`standardScaler`** | **`standardScaler`** |
| 超参数数 | 0 | 1~2 | **4** |
| 训练后诊断 | `["featureImportance"]` | `["featureImportance"]` | **`[]`——RBF 核无特征重要性** |
| 学习可视化 | `["learningCurve"]` | `[]`——无学习曲线 | **`["learningCurve"]`** |
| 数据量 | 200（手工合成） | 442（真实 + 构造） | **200（合成非线性）** |
| 训练方式 | SVD 闭式解 | 坐标下降 / 闭式解 | **SMO——序列最小优化** |
| PipelineSpec 元数据 | 无特殊 metadata | **`{"multiModel": True}`** | **`{"learningCurveEstimatorFactory": ...}`** |

## 常见坑

1. 误以为 SVR 也有特征重要性可视化——`PipelineSpec` 中训练后诊断列表明确为 `[]`，RBF 核不可输出。
2. 将 `learningCurveEstimatorFactory` 的作用与 `multiModel` 混淆——前者是"学习曲线用什么模型实例"，后者是"训练函数返回多个模型"。
3. 忽略标准化在运行器层而非数据层——数据层返回原始 Friedman1 数据，运行器负责 `StandardScaler` 调用。
4. 将 `PipelineSpec` 中的 `"svr"`（输出子目录名）与其他模型混淆——输出文件在 `outputs/svr/` 下。

## 小结

- SVR 工程实现采用声明式流水线架构——`PipelineSpec` 配置所有组件，`RegressionRunner` 按序编排执行。
- SVR 在回归分册中的工程定位是"标准化 + 学习曲线 + 无特征重要性"——填补了正则化回归（标准化 + 无学习曲线）和线性回归（无标准化 + 有学习曲线）之间的配置空白。
- 训练函数仅 2 行——`SVR(C=10.0, epsilon=0.1, kernel='rbf', gamma='scale').fit(XTrain, yTrain)`，是本仓库最简训练封装。
- 支持向量数量是终端日志中的关键诊断信息——比任何数值指标都更直观地反映模型复杂度。
