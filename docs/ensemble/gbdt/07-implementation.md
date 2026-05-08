---
title: GBDT 梯度提升树 — 工程实现
outline: deep
---

# 工程实现

## 本章目标

1. 理解 GBDT 流水线的模块分层——数据生成层、模型训练层、流水线编排层、可视化层。
2. 理清 `run()` 内部的函数调用链和数据流动路径。
3. 理解 GBDT 与 Bagging 在工程实现上的关键差异——串行训练、无版本兼容、更多评估项。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `EnsembleData.gbdt()` | 方法 | 生成多类别分类数据——`make_classification(n_classes=3, n_features=8)` |
| `train_model(...)` | 函数 | 构建并训练 `GradientBoostingClassifier`——无 sklearn 版本兼容处理 |
| `run()` | 函数 | 端到端流水线编排——串联数据准备、标准化、训练、预测和四项可视化 |
| `plot_confusion_matrix(...)` | 函数 | 绘制测试集混淆矩阵 |
| `plot_roc_curve(...)` | 函数 | 绘制多分类 ROC 曲线 |
| `plot_feature_importance(...)` | 函数 | 绘制特征重要性柱状图 |
| `plot_learning_curve(...)` | 函数 | 绘制学习曲线（训练集/验证集准确率变化） |

## 1. 模块分层总览

### 参数速览

| 层 | 文件 | 职责 | 输出 |
|---|---|---|---|
| 数据生成层 | `data_generation/ensemble.py` → `data_generation/__init__.py` | 生成多类别分类数据并导出为模块变量 `gbdt_data` | 全局 `DataFrame`（500 行 × 9 列） |
| 模型训练层 | `model_training/ensemble/gbdt.py` | 封装 `GradientBoostingClassifier` 训练——含超参数日志 | `GradientBoostingClassifier` 模型对象 |
| 流水线编排层 | `pipelines/ensemble/gbdt.py` | 串联数据准备、标准化、训练、预测和四项评估——端到端入口 | 终端日志 + 调用四个可视化函数 |
| 可视化层 | `result_visualization/confusion_matrix.py`、`roc_curve.py`、`feature_importance.py`、`learning_curve.py` | 生成四项评估图表 | 4 个 PNG 文件 |

### 理解重点

- GBDT 的可视化层比 Bagging 多了两个模块——`feature_importance.py` 和 `learning_curve.py`，体现 GBDT 更丰富的诊断能力。
- 训练层使用 `@print_func_info` 和 `@timeit` 装饰器——自动打印函数调用信息和耗时。
- 与 Bagging 对比：GBDT 的训练层没有 `try/except TypeError` 版本兼容处理——因为 `GradientBoostingClassifier` 的 API 更稳定。

## 2. `run()` 内部的函数调用链

### 参数速览

| 序号 | 调用 | 输入 | 输出 | 目的 |
|---|---|---|---|---|
| 1 | `gbdt_data.copy()` | — | `DataFrame`，形状 `(500, 9)` | 避免修改全局变量 |
| 2 | `data.drop(columns=["label"])` | `DataFrame` | `DataFrame`，形状 `(500, 8)` | 分离 8 维特征 X |
| 3 | `data["label"]` | `DataFrame` | `Series`，形状 `(500,)` | 分离三分类标签 y |
| 4 | `list(X.columns)` | `DataFrame` | `list[str]`，长度 8 | 提取特征名——供特征重要性图表使用 |
| 5 | `train_test_split(X, y, test_size=0.2, stratify=y)` | `(DataFrame, Series)` | `(X_train, X_test, y_train, y_test)` | 分层训练/测试切分 |
| 6 | `scaler.fit_transform(X_train)` | `DataFrame`，形状 `(400, 8)` | `ndarray`，形状 `(400, 8)` | 训练集标准化 |
| 7 | `scaler.transform(X_test)` | `DataFrame`，形状 `(100, 8)` | `ndarray`，形状 `(100, 8)` | 测试集标准化 |
| 8 | `train_model(X_train_s, y_train)` | `(ndarray, Series)` | `GradientBoostingClassifier` | 串行训练 200 棵浅层回归树 |
| 9 | `model.predict(X_test_s)` | `ndarray`，形状 `(100, 8)` | `ndarray`，形状 `(100,)` | 硬预测（加权累加 + softmax + argmax） |
| 10 | `plot_confusion_matrix(y_test, y_pred, ...)` | `(Series, ndarray)` | PNG 文件 | 混淆矩阵可视化 |
| 11 | `model.predict_proba(X_test_s)` | `ndarray`，形状 `(100, 8)` | `ndarray`，形状 `(100, 3)` | 软概率输出（softmax） |
| 12 | `plot_roc_curve(y_test, y_scores, ...)` | `(Series, ndarray)` | PNG 文件 | 多分类 ROC 曲线 |
| 13 | `plot_feature_importance(model, feature_names, ...)` | `(model, list)` | PNG 文件 | 特征重要性柱状图 |
| 14 | `plot_learning_curve(GradientBoostingClassifier(...), X_train_s, y_train, ...)` | `(estimator, ndarray, Series)` | PNG 文件 | 学习曲线 |

### 理解重点

- 步骤 4（`feature_names`）是 GBDT 独有的——Bagging 没有特征重要性评估，不需要这个步骤。
- 步骤 13 直接使用训练好的 `model` 提取 `feature_importances_`。
- 步骤 14 实例化了一个**新的** `GradientBoostingClassifier(n_estimators=100)`——不同于主模型的 200 棵树，以减少计算开销。
- 与 Bagging 的 11 步调用链对比，GBDT 多了 3 步（feature_names 提取 + 特征重要性 + 学习曲线）。

## 3. 数据依赖关系

```
gbdt_data (全局 DataFrame)
    │
    ├─→ X = data.drop(columns=["label"])  ──→ feature_names = list(X.columns) ──┐
    ├─→ y = data["label"]                                                        │
    │                                                                             │
    ├─→ train_test_split(X, y, test_size=0.2, stratify=y)                        │
    │   ├─→ X_train (400, 8) ──→ scaler.fit_transform() ──→ X_train_s ──┐       │
    │   ├─→ y_train (400,) ─────────────────────────────────────────────┤       │
    │   │                                                                 │       │
    │   ├─→ X_test (100, 8) ──→ scaler.transform() ──→ X_test_s ──┐     │       │
    │   └─→ y_test (100,) ─────────────────────────────────┐       │     │       │
    │                                                       │       │     │       │
    │   ┌───────────────────────────────────────────────────┘       │     │       │
    │   │                                                           │     │       │
    │   │  train_model(X_train_s, y_train) ──→ model               │     │       │
    │   │      │                                                     │     │       │
    │   │      ├─→ model.predict(X_test_s) ──→ y_pred ──┐          │     │       │
    │   │      │                                         │          │     │       │
    │   │      ├─→ model.predict_proba(X_test_s) ──→ y_scores ──┐  │     │       │
    │   │      │                                                  │  │     │       │
    │   │      ├─→ model.feature_importances_ ──→ + feature_names ──┼──┼─────┼───┐  │
    │   │      │                                                  │  │     │   │  │
    │   │      plot_confusion_matrix(y_test, y_pred, ...) ←───────┘  │     │   │  │
    │   │      plot_roc_curve(y_test, y_scores, ...) ←───────────────┘     │   │  │
    │   │      plot_feature_importance(model, feature_names, ...) ←────────┘   │  │
    │   │      plot_learning_curve(new_GBDT, X_train_s, y_train, ...) ←────────┘  │
    │   │                                                                          │
    │   └──────────────────────────────────────────────────────────────────────────┘
```

### 理解重点

- `feature_names` 是一个独立的横向数据流——从数据准备阶段流向特征重要性可视化，不参与训练和预测。
- `y_train` 同时参与训练（`train_model`）和学习曲线（`plot_learning_curve`）——后者会再次进行交叉验证切分。
- 与 Bagging 的数据依赖图对比——GBDT 多了 `feature_names` 分支和 `feature_importances_` 分支，以及学习曲线的额外 `GradientBoostingClassifier` 实例。

## 4. 输出文件一览

### 参数速览

| 输出项 | 路径 | 格式 | 说明 |
|---|---|---|---|
| 混淆矩阵 | `outputs/gbdt/confusion_matrix.png` | PNG | 测试集 3×3 混淆矩阵热力图 |
| ROC 曲线 | `outputs/gbdt/roc_curve.png` | PNG | 多分类 ROC（one-vs-rest，每类一条 + 平均） |
| 特征重要性 | `outputs/gbdt/feature_importance.png` | PNG | 8 个特征的重要性排序柱状图 |
| 学习曲线 | `outputs/gbdt/learning_curve.png` | PNG | 训练/验证准确率 vs 训练样本数 |
| 终端日志 | 标准输出 | 文本 | 训练超参数 + 运行耗时 |

### 示例代码

```bash
python -m pipelines.ensemble.gbdt
```

### 输出

```text
============================================================
GBDT 分类流水线
============================================================
模型训练完成
n_estimators: 200
learning_rate: 0.1
max_depth: 3
subsample: 1.0
模型训练耗时: 2.15s

============================================================
GBDT 流水线完成！
============================================================
```

### 理解重点

- GBDT 输出 4 个 PNG 文件——比 Bagging 多 2 个（特征重要性 + 学习曲线）。
- 训练耗时通常比 Bagging 长——因为 200 棵树必须串行训练（Bagging 的 80 棵树可以并行）。
- 终端日志没有 OOB 得分——GBDT 没有 Bootstrap 采样，无 OOB 概念。

## 5. 训练层细节：与 Bagging 的对比

| 工程维度 | Bagging | GBDT |
|---|---|---|
| sklearn 版本兼容 | 有 `try/except TypeError`（`estimator` vs `base_estimator`） | 无——`GradientBoostingClassifier` 参数名稳定 |
| 训练并行 | `n_jobs=-1`——80 棵树并行 | 串行——每棵树依赖前序结果 |
| 基学习器创建 | 显式 `DecisionTreeClassifier(max_depth=None)` | 由 `GradientBoostingClassifier` 内部创建（用户只指定 `max_depth=3`） |
| OOB 得分 | 有 `oob_score_` | 无 |
| 日志内容 | `n_estimators, max_samples, max_features, bootstrap, OOB 得分` | `n_estimators, learning_rate, max_depth, subsample` |
| 返回模型 | `BaggingClassifier`（分类树基学习器） | `GradientBoostingClassifier`（回归树基学习器） |

### 理解重点

- GBDT 不需要显式创建基学习器——`GradientBoostingClassifier` 内部自动使用 `DecisionTreeRegressor`（注意是回归树，不是分类树）。
- GBDT 的训练是纯串行的——没办法像 Bagging 那样用 `n_jobs=-1` 并行训练多棵树。
- GBDT 的训练日志没有 OOB 得分——因为没有 Bootstrap 采样。

## 阅读顺序

1. `data_generation/ensemble.py` — 了解 `gbdt()` 的数据生成逻辑和参数设计
2. `model_training/ensemble/gbdt.py` — 理解 GBDT 模型的构建和串行训练
3. `pipelines/ensemble/gbdt.py` — 看清端到端流程和四项评估的串联
4. `result_visualization/confusion_matrix.py` — 了解混淆矩阵实现
5. `result_visualization/roc_curve.py` — 了解多分类 ROC 实现
6. `result_visualization/feature_importance.py` — 了解特征重要性图表实现
7. `result_visualization/learning_curve.py` — 了解学习曲线实现

## 常见坑

1. 直接修改 `gbdt_data` 而不先 `copy()`——会污染其他模块引用的同一变量。
2. 在测试集上使用 `fit_transform` 而非 `transform`——标准信息泄露。
3. 忘记提取 `feature_names`——没有特征名时特征重要性图表只有索引号。
4. 把 GBDT 的基学习器误解为分类树——GBDT 内部使用的是回归树（拟合连续负梯度值）。

## 小结

- GBDT 工程实现遵循本仓库标准四层架构：数据生成层 → 模型训练层 → 流水线编排层 → 可视化层（含 4 个模块）。
- `run()` 是薄编排函数——14 步调用串联数据准备、标准化、串行训练、预测和四项评估。
- 与 Bagging 的三个关键工程差异：（1）训练串行不可并行化；（2）无需 sklearn 版本兼容处理；（3）多 2 项评估输出（特征重要性 + 学习曲线）。
