---
title: XGBoost — 工程实现
outline: deep
---

# 工程实现

## 本章目标

1. 理解 XGBoost 流水线的模块分层——数据生成层、模型训练层、流水线编排层、可视化层。
2. 理清 `run()` 内部的函数调用链和数据流动路径——注意回归任务的无标准化、无分层特点。
3. 理解 XGBoost 与其他集成模型在工程实现上的关键差异——可选依赖、回归评估、无数据预处理。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `EnsembleData.xgboost()` | 静态方法 | 返回加州房价真实数据集——`fetch_california_housing` |
| `train_model(...)` | 函数 | 构建并训练 `XGBRegressor`——含可选依赖检查和 12 个可配置参数 |
| `run()` | 函数 | 回归流水线编排——6 步串联数据拆分、训练、预测和两项评估 |
| `plot_residuals(...)` | 函数 | 绘制残差散点图和分布图——回归专用 |
| `plot_feature_importance(...)` | 函数 | 绘制特征重要性柱状图 |

## 1. 模块分层总览

### 参数速览

| 层 | 文件 | 职责 | 输出 |
|---|---|---|---|
| 数据生成层 | `data_generation/ensemble.py` → `data_generation/__init__.py` | 加载加州房价真实数据并导出 `xgboost_data` | 全局 `DataFrame`（20640 行 × 9 列） |
| 模型训练层 | `model_training/ensemble/xgboost.py` | 封装 `XGBRegressor` 训练——含 `ImportError` 处理 + 装饰器 | `XGBRegressor` 模型对象 |
| 流水线编排层 | `pipelines/ensemble/xgboost.py` | 串联数据拆分、训练、预测和两项评估——端到端入口 | 终端日志 + 调用两个可视化函数 |
| 可视化层 | `result_visualization/residual_plot.py`、`feature_importance.py` | 生成两项评估图表 | 2 个 PNG 文件 |

### 理解重点

- XGBoost 的可视化层使用 `residual_plot.py`（回归专用）替代了 `confusion_matrix.py` 和 `roc_curve.py`（分类专用）。
- 训练层有三重保护：`try/except ImportError`（可选依赖）+ `@print_func_info`（调用日志）+ `@timeit`（耗时日志）。
- 与其他集成模型的核心工程差异：（1）无标准化步骤；（2）无分层抽样；（3）使用残差图替代混淆矩阵/ROC。

## 2. `run()` 内部的函数调用链

### 参数速览

| 序号 | 调用 | 输入 | 输出 | 目的 |
|---|---|---|---|---|
| 1 | `xgboost_data.copy()` | — | `DataFrame`，形状 `(20640, 9)` | 避免修改全局变量 |
| 2 | `data.drop(columns=["price"])` | `DataFrame` | `DataFrame`，形状 `(20640, 8)` | 分离 8 维特征 X |
| 3 | `data["price"]` | `DataFrame` | `Series`，形状 `(20640,)` | 分离连续回归目标 y |
| 4 | `list(X.columns)` | `DataFrame` | `list[str]`，长度 8 | 提取特征名——供特征重要性图表使用 |
| 5 | `train_test_split(X, y, test_size=0.2)` | `(DataFrame, Series)` | `(X_train, X_test, y_train, y_test)` | 训练/测试切分（无 stratification） |
| 6 | `train_model(X_train, y_train)` | `(DataFrame, Series)` | `XGBRegressor` | 训练 300 棵二阶正则化树 |
| 7 | `model.predict(X_test)` | `DataFrame`，`(4128, 8)` | `ndarray`，`(4128,)` | 连续房价预测值 |
| 8 | `plot_residuals(y_test, y_pred, ...)` | `(Series, ndarray)` | PNG 文件 | 残差散点图 + 分布图 |
| 9 | `plot_feature_importance(model, feature_names, ...)` | `(model, list)` | PNG 文件 | 8 个特征重要性排序柱状图 |

### 理解重点

- 步骤 5 无 `stratify=y` 参数——回归任务的连续目标没有类别可分层。
- 步骤 6 无标准化——树模型天然对特征缩放不敏感，跳过预处理环节。
- 步骤 8 使用残差分析替代分类的混淆矩阵/ROC——回归评估的根本差异。
- XGBoost 的流水线是最简洁的——6 步 vs Bagging 7 步、GBDT 9 步、LightGBM 7 步。

## 3. 数据依赖关系

```
xgboost_data (全局 DataFrame)
    │
    ├─→ X = data.drop(columns=["price"])  ──→ feature_names = list(X.columns) ──┐
    ├─→ y = data["price"]                                                        │
    │                                                                             │
    ├─→ train_test_split(X, y, test_size=0.2)                                    │
    │   ├─→ X_train (16512, 8) ──────────────────────────────────────┐          │
    │   ├─→ y_train (16512,) ────────────────────────┐               │          │
    │   │                                             │               │          │
    │   ├─→ X_test (4128, 8) ────────────────────┐   │               │          │
    │   └─→ y_test (4128,) ───────────────┐      │   │               │          │
    │                                       │      │   │               │          │
    │   ┌───────────────────────────────────┘      │   │               │          │
    │   │                                          │   │               │          │
    │   │  train_model(X_train, y_train) ──→ model │   │               │          │
    │   │      │                                    │   │               │          │
    │   │      ├─→ model.predict(X_test) ──→ y_pred─┘   │               │          │
    │   │      │                                         │               │          │
    │   │      ├─→ model.feature_importances_ ──→ + feature_names ──────┘          │
    │   │      │                                         │                         │
    │   │      plot_residuals(y_test, y_pred, ...) ←─────┘                         │
    │   │      plot_feature_importance(model, feature_names, ...) ←────────────────┘
    │   │
    │   └──────────────────────────────────────────────────────────────────────────┘
```

### 理解重点

- XGBoost 的数据依赖图是最简洁的——无 `StandardScaler` 分支、无 `plot_learning_curve` 分支、无 `predict_proba` 分支。
- `y_train` 仅参与训练，`y_test` 仅参与残差分析——没有混淆矩阵和 ROC 的数据需求。
- `feature_names` 与 `feature_importances_` 交汇于特征重要性可视化——流程清晰。

## 4. 输出文件一览

### 参数速览

| 输出项 | 路径 | 格式 | 说明 |
|---|---|---|---|
| 残差分析图 | `outputs/xgboost/residual_plot.png` | PNG | 残差散点图（预测值 vs 残差）+ 残差分布直方图 |
| 特征重要性 | `outputs/xgboost/feature_importance.png` | PNG | 8 个特征的重要性排序柱状图 |
| 终端日志 | 标准输出 | 文本 | 9 项训练超参数 + 运行耗时 |

### 示例代码

```bash
python -m pipelines.ensemble.xgboost
```

### 输出

```text
============================================================
XGBoost 回归流水线
============================================================
模型训练完成
n_estimators: 300
learning_rate: 0.05
max_depth: 6
min_child_weight: 1
subsample: 0.9
colsample_bytree: 0.9
gamma: 0.0
reg_alpha: 0.0
reg_lambda: 1.0
模型训练耗时: 3.11s

============================================================
XGBoost 流水线完成！
============================================================
```

### 理解重点

- XGBoost 输出 2 个 PNG 文件——四个集成模型中最少（Bagging 2 个、GBDT 4 个、LightGBM 3 个）。
- 训练耗时比 GBDT（~2s）长——因为 20640 个样本远多于 GBDT 的 500 个，但 `n_jobs=-1` 列块并行在一定程度上抵消了数据规模增长。
- 终端日志打印 9 项超参数——四个模型中最多，体现 XGBoost 参数体系的丰富程度。

## 5. 训练层细节：与其他集成模型的对比

| 工程维度 | Bagging | GBDT | LightGBM | XGBoost |
|---|---|---|---|---|
| 任务 | 分类 | 分类 | 分类 | **回归** |
| 模型类 | `BaggingClassifier` | `GradientBoostingClassifier` | `LGBMClassifier` | **`XGBRegressor`** |
| 依赖 | sklearn 内置 | sklearn 内置 | `pip install lightgbm` | **`pip install xgboost`** |
| 导入保护 | `try/except TypeError` | 无 | `try/except ImportError` | `try/except ImportError` |
| 装饰器 | 无 | `timer` | `@print_func_info` + `@timeit` + `timer` | `@print_func_info` + `@timeit` + `timer` |
| 标准化 | 有 | 有 | 有 | **无** |
| 分层抽样 | 有 | 有 | 有 | **无** |
| 评估项 | 混淆矩阵 + ROC | 混淆矩阵 + ROC + 特征重要性 + 学习曲线 | 混淆矩阵 + ROC + 特征重要性 | **残差图 + 特征重要性** |
| 超参数数 | 5 | 4 | 6 | **9** |

### 理解重点

- XGBoost 的训练层参数是四个模型中最丰富的——从 `gamma` 到 `reg_lambda`，体现更精细的控制粒度。
- 无标准化和无分层的设计使得 XGBoost 的流水线最简洁——树模型的工程便利性在此充分体现。
- XGBoost 与 LightGBM 共享可选的依赖处理模式——两者都不是 sklearn 原生，需要 `try/except` 保护。

## 阅读顺序

1. `data_generation/ensemble.py` — 了解 `xgboost()` 的数据加载逻辑（加州房价真实数据）
2. `model_training/ensemble/xgboost.py` — 理解 `XGBRegressor` 的构建、可选依赖和二阶训练
3. `pipelines/ensemble/xgboost.py` — 看清端到端回归流程和两项评估的串联
4. `result_visualization/residual_plot.py` — 了解残差分析图实现
5. `result_visualization/feature_importance.py` — 了解特征重要性图表实现

## 常见坑

1. 在不含 `xgboost` 的环境中直接 `from model_training.ensemble.xgboost import train_model`——会触发 `ImportError`，需先 `pip install xgboost`。
2. 在回归数据上传递 `stratify=y`——回归无类别可分层，会直接报错。
3. 直接修改 `xgboost_data` 而不先 `copy()`——会污染其他模块引用的同一全局变量。
4. 期望 XGBoost 流水线输出混淆矩阵——回归任务没有混淆矩阵概念。

## 小结

- XGBoost 工程实现遵循本仓库标准四层架构：数据生成层 → 模型训练层 → 流水线编排层 → 可视化层（含 2 个模块）。
- `run()` 是四个集成模型中最简洁的编排函数——6 步完成数据拆分、训练、预测和两项评估，无预处理步骤。
- 与其他集成模型的三个关键工程差异：（1）回归任务——无分类评估；（2）真实数据——无需标准化；（3）参数体系最丰富——9 项可配置超参数。
