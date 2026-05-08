---
title: Bagging 集成学习 — 工程实现
outline: deep
---

# 工程实现

## 本章目标

1. 理解 Bagging 流水线的模块分层——数据生成层、模型训练层、流水线编排层、可视化层。
2. 理清 `run()` 内部的函数调用链和数据流动路径。
3. 了解输出文件结构和阅读顺序。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `EnsembleData.bagging()` | 方法 | 生成高噪声双月牙二分类数据——`make_moons(noise=0.35)` |
| `train_model(...)` | 函数 | 构建并训练 `BaggingClassifier`——含 sklearn 版本兼容处理 |
| `run()` | 函数 | 端到端流水线编排——串联数据准备、标准化、训练、预测、可视化 |
| `plot_confusion_matrix(...)` | 函数 | 绘制测试集混淆矩阵 |
| `plot_roc_curve(...)` | 函数 | 绘制 ROC 曲线（条件：`predict_proba` 可用） |

## 1. 模块分层总览

### 参数速览

| 层 | 文件 | 职责 | 输出 |
|---|---|---|---|
| 数据生成层 | `data_generation/ensemble.py` → `data_generation/__init__.py` | 生成高噪声双月牙数据并导出为模块变量 `bagging_data` | 全局 `DataFrame`（500 行 × 3 列） |
| 模型训练层 | `model_training/ensemble/bagging.py` | 封装 `BaggingClassifier` 训练——含基学习器创建、sklearn 版本兼容、OOB 日志 | `BaggingClassifier` 模型对象 |
| 流水线编排层 | `pipelines/ensemble/bagging.py` | 串联数据准备、标准化、训练、预测、评估——端到端入口 | 终端日志 + 调用可视化函数 |
| 可视化层 | `result_visualization/confusion_matrix.py`、`result_visualization/roc_curve.py` | 生成混淆矩阵热力图和 ROC 曲线图 | PNG 文件 |

### 理解重点

- 四层架构是本仓库所有算法的通用模式——数据生成、模型训练、流水线编排、结果可视化各司其职。
- 数据生成层使用 `@dataclass` 管理参数——可灵活调整噪声水平而不影响其他模块。
- 训练层使用 `@print_func_info` 和 `@timeit` 装饰器——自动打印函数调用信息和耗时。
- 流水线层是薄编排层——不包含算法逻辑，只负责串起各步骤。

## 2. `run()` 内部的函数调用链

### 参数速览

| 序号 | 调用 | 输入 | 输出 | 目的 |
|---|---|---|---|---|
| 1 | `bagging_data.copy()` | — | `DataFrame`，形状 `(500, 3)` | 避免修改全局变量 |
| 2 | `data.drop(columns=["label"])` | `DataFrame` | `DataFrame`，形状 `(500, 2)` | 分离特征 X |
| 3 | `data["label"]` | `DataFrame` | `Series`，形状 `(500,)` | 分离标签 y |
| 4 | `train_test_split(X, y, test_size=0.2, stratify=y)` | `(DataFrame, Series)` | `(X_train, X_test, y_train, y_test)` | 分层训练/测试切分 |
| 5 | `scaler.fit_transform(X_train)` | `DataFrame`，形状 `(400, 2)` | `ndarray`，形状 `(400, 2)` | 训练集标准化 |
| 6 | `scaler.transform(X_test)` | `DataFrame`，形状 `(100, 2)` | `ndarray`，形状 `(100, 2)` | 测试集标准化 |
| 7 | `train_model(X_train_s, y_train)` | `(ndarray, Series)` | `BaggingClassifier` | 训练含 80 棵树的 Bagging 模型 |
| 8 | `model.predict(X_test_s)` | `ndarray`，形状 `(100, 2)` | `ndarray`，形状 `(100,)` | 硬投票预测 |
| 9 | `plot_confusion_matrix(y_test, y_pred, ...)` | `(Series, ndarray)` | PNG 文件 | 混淆矩阵可视化 |
| 10 | `model.predict_proba(X_test_s)` | `ndarray`，形状 `(100, 2)` | `ndarray`，形状 `(100, 2)` | 软概率输出 |
| 11 | `plot_roc_curve(y_test, y_scores, ...)` | `(Series, ndarray)` | PNG 文件（条件） | ROC 曲线可视化 |

### 理解重点

- 步骤 5 和 6 的差异至关重要——`fit_transform` 在训练集上同时计算统计量和变换，`transform` 在测试集上仅使用训练集的统计量。
- 步骤 7 的内部并行训练 80 棵决策树——`n_jobs=-1` 利用全部 CPU 核心。
- 步骤 10-11 是条件执行的——`hasattr(model, "predict_proba")` 检查通过后才执行。对 Bagging 而言始终满足条件。

## 3. 数据依赖关系

```
bagging_data (全局 DataFrame)
    │
    ├─→ X = data.drop(columns=["label"])  ──┐
    ├─→ y = data["label"]                   │
    │                                        │
    ├─→ train_test_split(X, y, test_size=0.2, stratify=y)
    │   ├─→ X_train (400, 2) ──→ scaler.fit_transform() ──→ X_train_s ──┐
    │   ├─→ y_train (400,) ─────────────────────────────────────────────┤
    │   │                                                                  │
    │   ├─→ X_test (100, 2) ──→ scaler.transform() ──→ X_test_s ──┐      │
    │   └─→ y_test (100,) ─────────────────────────────────┐       │      │
    │                                                       │       │      │
    │   ┌───────────────────────────────────────────────────┘       │      │
    │   │                                                           │      │
    │   │  train_model(X_train_s, y_train) ──→ model               │      │
    │   │      │                                                     │      │
    │   │      ├─→ model.predict(X_test_s) ──→ y_pred ──┐          │      │
    │   │      │                                         │          │      │
    │   │      └─→ model.predict_proba(X_test_s) ──→ y_scores ──┐  │      │
    │   │                                                        │  │      │
    │   │  plot_confusion_matrix(y_test, y_pred, ...) ←─────────┘  │      │
    │   │  plot_roc_curve(y_test, y_scores, ...) ←─────────────────┘      │
    │   │                                                                   │
    │   └───────────────────────────────────────────────────────────────────┘
```

### 理解重点

- `y` 是最"忙碌"的变量——同时参与了训练（`y_train` → `train_model`）和两项评估（`y_test` → 混淆矩阵 + ROC 曲线）。
- 与 PCA/LDA 的数据依赖图对比——Bagging 的 `y` 承担了"训练目标"和"评估基准"双重角色，PCA 的标签仅用于可视化着色。

## 4. 输出文件一览

### 参数速览

| 输出项 | 路径 | 格式 | 说明 |
|---|---|---|---|
| 混淆矩阵 | `outputs/bagging/confusion_matrix.png` | PNG | 测试集 100 个样本的硬投票分类结果热力图 |
| ROC 曲线 | `outputs/bagging/roc_curve.png` | PNG（条件） | 测试集概率输出的 ROC 曲线——含 AUC 标注 |
| 终端日志 | 标准输出 | 文本 | 训练超参数 + OOB 得分 + 运行耗时 |

### 示例代码

```bash
python -m pipelines.ensemble.bagging
```

### 输出

```text
============================================================
Bagging 分类流水线
============================================================
模型训练完成
n_estimators: 80
max_samples: 0.8
max_features: 1.0
bootstrap: True
OOB 得分: 0.8975
模型训练耗时: 0.32s

============================================================
Bagging 流水线完成！
============================================================
```

### 理解重点

- 终端是唯一能看到 OOB 得分的地方——不会单独存为文件。
- `outputs/bagging/` 目录在流水线运行时自动创建——无需手动创建。
- 混淆矩阵始终生成——ROC 曲线在 `BaggingClassifier` 下也始终生成（因为 `predict_proba` 始终可用）。

## 5. 训练层细节：sklearn 版本兼容

### 示例代码

```python
# 兼容不同 sklearn 版本
try:
    model = BaggingClassifier(
        estimator=base,          # sklearn ≥ 1.2
        n_estimators=n_estimators,
        ...
    )
except TypeError:
    model = BaggingClassifier(
        base_estimator=base,     # sklearn < 1.2
        n_estimators=n_estimators,
        ...
    )
```

### 理解重点

- sklearn 1.2 版本将 `base_estimator` 参数名改为 `estimator`——`try/except TypeError` 处理了这个 API 变化。
- 首次使用 `estimator` 参数名——如果当前 sklearn 版本 < 1.2，会抛出 `TypeError`，随后用旧参数名重试。
- 这两个分支创建的是完全相同的模型对象——只是参数名不同。

## 阅读顺序

1. `data_generation/ensemble.py` — 了解数据来源和噪声设计意图
2. `model_training/ensemble/bagging.py` — 理解 Bagging 模型的构建和训练
3. `pipelines/ensemble/bagging.py` — 看清端到端流程
4. `result_visualization/confusion_matrix.py` — 了解混淆矩阵的实现
5. `result_visualization/roc_curve.py` — 了解 ROC 曲线的实现

## 常见坑

1. 直接修改 `bagging_data` 而不先 `copy()`——会污染其他模块引用的同一变量。
2. 在测试集上使用 `fit_transform` 而非 `transform`——标准信息泄露。
3. 混淆 `estimator` 和 `base_estimator` 参数名——当前源码已兼容，但写作评估代码时需要注意 sklearn 版本。
4. 忘记检查 `hasattr(model, "predict_proba")`——虽然 Bagging 始终支持，但换用其他模型时可能崩溃。

## 小结

- Bagging 工程实现遵循本仓库标准四层架构：数据生成层 → 模型训练层 → 流水线编排层 → 可视化层。
- `run()` 是薄编排函数——10 步调用串联数据准备、标准化、训练、预测和评估。
- sklearn 版本兼容（`estimator` vs `base_estimator`）和防御性 `hasattr` 检查是本实现的两个工程细节亮点。
