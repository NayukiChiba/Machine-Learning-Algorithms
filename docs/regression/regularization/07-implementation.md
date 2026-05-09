---
title: 正则化回归 — 工程实现
outline: deep
---

# 工程实现

## 本章目标

1. 理解正则化回归流水线的模块分层——数据层、训练层、流水线注册层、运行器层和可视化层。
2. 理清从命令行入口到结果图落盘的完整调用链，特别是多模型循环分支。
3. 理解正则化回归与线性回归、决策树回归在工程实现上的关键差异——标准化 + 多模型 + 无学习曲线。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `RegressionDatasetFactory` | 类 | 数据工厂——`loadRegularizationDataset()` 构造 diabetes + 共线 + 噪声 |
| `trainRegularizationModels(...)` | 函数 | 构建并训练三个正则化模型——返回 `dict` |
| `PipelineSpec` | 数据类 | 声明式流水线配置——`multiModel=True` 标记多模型模式 |
| `RegressionRunner` | 类 | 回归流水线运行器——多模型分支下循环评估每个模型 |
| `plot_residuals(...)` | 函数 | 残差图绘制——每个模型独立调用 |
| `plot_feature_importance(...)` | 函数 | 系数柱状图绘制——每个模型独立调用 |

## 1. 模块分层总览

### 参数速览

| 层 | 文件 | 职责 | 输出 |
|---|---|---|---|
| 数据层 | `src/mlAlgorithms/datasets/tabular/regressionDatasets.py` | `loadRegularizationDataset()`——加载 diabetes 并追加共线和噪声 | `DataFrame`，形状 `(442, 22)` |
| 数据目录层 | `src/mlAlgorithms/datasets/datasetCatalog.py` | `DatasetSpec("regression.regularization", ...)`——注册数据集描述与加载器 | 数据集元信息 |
| 训练层 | `src/mlAlgorithms/training/regression/regressionModels.py` | `trainRegularizationModels(...)`——构建三模型 `dict` 并 `fit` | `dict[str, 模型]` |
| 流水线注册层 | `src/mlAlgorithms/catalog/pipelines.py` | `PipelineSpec("regression.regularization", ...)`——关联所有组件 | 流水线配置 |
| 运行器层 | `src/mlAlgorithms/workflows/regressionRunner.py` | 读取 PipelineSpec → 加载 → 标准化 → 训练 → 循环评估 → 可视化 | 终端日志 + 图像文件 |
| 可视化层 | `src/mlAlgorithms/visualization/` | 绘制残差图、系数图 | PNG 图像文件 |

### 理解重点

- 正则化回归的工程结构比线性回归多两样：标准化和多模型循环——其余结构完全一致。
- 与决策树回归的差异：决策树有学习曲线和树结构可视化，正则化回归有标准化和多模型但无学习曲线。
- `multiModel=True` 是运行器层的关键分支标志——告诉 `RegressionRunner` 训练返回的是 `dict` 而非单个模型。

## 2. `PipelineSpec` 配置详情

```python
PipelineSpec(
    "regression.regularization",           # pipeline ID
    TaskType.REGRESSION,                   # 任务类型
    "regression.regularization",           # dataset ID
    RunnerType.REGRESSION,                 # 运行器类型
    trainRegularizationModels,             # 训练函数——返回 dict
    "standardScaler",                      # 预处理——正则化强制标准化
    "randomSplit",                         # 切分策略
    "default",                             # 后处理
    "regression",                          # 输出目录前缀
    "regression",                          # 可视化目录前缀
    ["correlationHeatmap", "featureTargetScatter"],  # 训练前可视化
    ["featureImportance"],                 # 训练后诊断可视化——系数图
    [],                                    # 学习可视化——无学习曲线
    "ridge",                               # 结果存储子目录
    metadata={"multiModel": True},         # 多模型标记——运行器据此分支
)
```

### 理解重点

- `"standardScaler"` 是正则化回归与线性回归、决策树回归在 `PipelineSpec` 层面的关键差异——前者为 `"standardScaler"`，后两者为 `None`。
- `metadata={"multiModel": True}` 告诉运行器训练函数返回的是模型字典——运行器会循环评估每个模型。
- `[]` 学习可视化列表为空——正则化回归不生成学习曲线，这在回归分册中是独特的。
- `"ridge"` 是结果存储子目录名——但实际输出包含 lasso/ridge/elasticnet 三个模型的各自文件。

## 3. 数据依赖关系

```
loadRegularizationDataset()
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
    │       ├─→ trainRegularizationModels(X_train_s, y_train)
    │       │       │
    │       │       ├─→ models["lasso"] = Lasso(...).fit()
    │       │       ├─→ models["ridge"] = Ridge(...).fit()
    │       │       └─→ models["elasticnet"] = ElasticNet(...).fit()
    │       │
    │       └─→ for name, model in models.items():
    │               │
    │               ├─→ y_pred = model.predict(X_test_s)
    │               ├─→ plot_residuals(y_test, y_pred)
    │               └─→ plot_feature_importance(model, feature_names)
```

### 理解重点

- 标准化分支（`fit_transform` / `transform`）是正则化回归独有的——线性回归和决策树回归的数据依赖图中没有这一分支。
- 训练分支产出三个模型——后续所有可视化步骤循环执行三次。
- 没有学习曲线分支——`plot_learning_curve` 不出现在此数据依赖图中。

## 4. 运行器层的执行链

| 序号 | 步骤 | 说明 |
|---|---|---|
| 1 | 根据 `datasetId` 查找 `DatasetSpec` | 获取数据加载器和描述信息 |
| 2 | 调用 `loadRegularizationDataset()` | 加载 `(442, 22)` DataFrame |
| 3 | 拆分 X / y + 保存 `feature_names` | 为后续日志和可视化作准备 |
| 4 | `train_test_split(test_size=0.2)` | 随机切分 |
| 5 | `StandardScaler().fit_transform(X_train)` | 标准化训练集——**正则化回归独有** |
| 6 | `StandardScaler().transform(X_test)` | 标准化测试集 |
| 7 | 调用 `trainRegularizationModels(X_train_s, y_train)` | 训练三个模型——返回 `dict` |
| 8 | 检测 `multiModel=True` → 进入多模型循环 | **多模型分支——其他回归模型无此步骤** |
| 9 | 循环内：`model.predict(X_test_s)` | 每个模型独立预测 |
| 10 | 循环内：`plot_residuals(y_test, y_pred)` | 每个模型独立生成残差图 |
| 11 | 循环内：`plot_feature_importance(model, feature_names)` | 每个模型独立生成系数图 |

### 理解重点

- 步骤 5-6 是正则化回归与线性回归/决策树回归在运行器层的根本差异——多了标准化步骤。
- 步骤 8 是多模型检测分支——运行器检查 `metadata["multiModel"]`，若为 `True` 则对 `models.items()` 循环评估。
- 步骤 9-11 在循环内执行三次——相比线性回归的单模型路径，多出了两次 `predict` 和两次可视化调用。

## 5. 正则化回归 vs 线性回归 vs 决策树回归 工程对比

| 工程维度 | 线性回归 | 决策树回归 | 正则化回归 |
|---|---|---|---|
| 训练函数 | `trainLinearRegressionModel` | `trainDecisionTreeRegressionModel` | **`trainRegularizationModels`** |
| 模型类 | `LinearRegression` | `DecisionTreeRegressor` | **`Lasso`, `Ridge`, `ElasticNet`** |
| 训练函数行数 | 3 行 | ~5 行 | **~10 行** |
| 预处理 | `None` | `None` | **`standardScaler`** |
| 超参数数 | 0 | 3 | **Lasso: 1, Ridge: 1, EN: 2** |
| 训练后诊断 | `["featureImportance"]` | `["featureImportance"]` | **`["featureImportance"]`** |
| 学习可视化 | `["learningCurve"]` | `["learningCurve", "treeStructure"]` | **`[]`——无学习曲线** |
| 数据量 | 200（手工合成） | 20640（真实数据） | **442（真实 + 构造）** |
| 训练方式 | SVD 闭式解 | CART 贪心递归 | **坐标下降（Lasso/EN）+ 闭式解（Ridge）** |
| PipelineSpec 元数据 | 无 | 无 | **`{"multiModel": True}`** |

## 常见坑

1. 误以为运行器对所有回归模型统一处理——`multiModel=True` 导致运行器走多模型循环分支，与单模型路径不同。
2. 把 `"ridge"`（输出子目录名）理解成只输出 Ridge 的结果——实际三个模型都有独立输出。
3. 期待正则化回归也有学习曲线——`PipelineSpec` 中学可视化列表明确为 `[]`。
4. 忽略标准化在运行器层而非数据层执行——数据层返回原始值，运行器负责 `StandardScaler` 调用。

## 小结

- 正则化回归工程实现采用声明式流水线架构——`PipelineSpec` 配置所有组件，`RegressionRunner` 按 `multiModel=True` 分支执行。
- 与线性回归/决策树回归的工程差异：（1）唯一使用 `standardScaler` 预处理；（2）唯一返回多模型 `dict`；（3）唯一没有学习曲线；（4）唯一使用 `multiModel` 元数据。
- 标准化 + 多模型循环是正则化回归工程实现的两大核心特征——理解这两点就理解了正则化回归的工程全貌。
