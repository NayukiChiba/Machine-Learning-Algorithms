---
title: EM 与 GMM — 工程实现
outline: deep
---

# 工程实现

## 本章目标

1. 理解 EM 流水线的模块分层——数据生成层、模型训练层、流水线编排层、聚类可视化层。
2. 理清 `run()` 内部的函数调用链和数据流动路径——注意无监督特征（无 `y_train`、无切分）。
3. 理解 EM 与 KMeans/DBSCAN 在工程实现上的异同——同为聚类，但模型内部结构完全不同。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `ProbabilisticData.em()` | 方法 | 手动合成 3 分量非球形 GMM 数据 |
| `train_model(...)` | 函数 | 构建并训练 `GaussianMixture`——无监督，无 `y_train` 参数 |
| `run()` | 函数 | 无监督聚类流水线编排——5 步串联标准化、EM 训练、预测和可视化 |
| `plot_clusters(...)` | 函数 | 绘制双面板聚类分布对比图——预测标签 vs 真实标签 |
| `model.predict(X)` | 方法 | 硬聚类标签 |
| `model.predict_proba(X)` | 方法 | 软归属——后验责任矩阵 |

## 1. 模块分层总览

### 参数速览

| 层 | 文件 | 职责 | 输出 |
|---|---|---|---|
| 数据生成层 | `data_generation/probabilistic.py` | 手动合成 3 分量非球形 GMM 数据并导出 `em_data` | 全局 `DataFrame`（500 行 × 3 列） |
| 模型训练层 | `model_training/probabilistic/em.py` | 封装 `GaussianMixture` 训练——含装饰器 | `GaussianMixture` 模型对象 |
| 流水线编排层 | `pipelines/probabilistic/em.py` | 串联标准化、EM 训练、预测和聚类可视化——端到端入口 | 终端日志 + 聚类分布图 |
| 可视化层 | `result_visualization/cluster_plot.py` | 生成双面板聚类分布对比图 | 1 个 PNG 文件 |

### 理解重点

- EM 的模块分层与 KMeans/DBSCAN 的结构完全一致——数据生成 → 训练封装 → 流水线编排 → 聚类可视化。
- 训练层使用 `@print_func_info` + `@timeit` + `timer`——与 GBDT/LightGBM/XGBoost 的装饰器风格一致。
- 与集成分类的最关键区别：**训练层不接收 `y_train`**——EM 是无监督学习。

## 2. `run()` 内部的函数调用链

### 参数速览

| 序号 | 调用 | 输入 | 输出 | 目的 |
|---|---|---|---|---|
| 1 | `em_data.copy()` | — | `DataFrame`，形状 `(500, 3)` | 避免修改全局变量 |
| 2 | `data["true_label"].values` | `DataFrame` | `ndarray`，`(500,)` | 提取真实标签——仅供评估对比 |
| 3 | `data.drop(columns=["true_label"])` | `DataFrame` | `DataFrame`，`(500, 2)` | 分离 2 维特征 X |
| 4 | `scaler.fit_transform(X)` | `DataFrame` | `ndarray`，`(500, 2)` | 全量数据 Z-score 标准化 |
| 5 | `train_model(X_scaled)` | `ndarray` | `GaussianMixture` | EM 迭代训练——无 `y_train` |
| 6 | `model.predict(X_scaled)` | `ndarray` | `ndarray`，`(500,)` | 硬聚类标签 |
| 7 | `plot_clusters(X_scaled, labels_pred, y_true, ...)` | `(ndarray, ndarray, ndarray)` | PNG 文件 | 双面板聚类对比图 |

### 理解重点

- 步骤 2-3 顺序不可交换——必须先提取 `true_label`，再 `drop`。如果先 `drop`，`true_label` 将丢失。
- 步骤 5 无 `y_train` 参数——这是 EM 与集成分类训练函数的根本差异。
- 与 KMeans 流水线唯一的区别：`plot_clusters` 多传了一个 `labels_true` 参数以实现双面板对比。

## 3. 数据依赖关系

```
em_data (全局 DataFrame)
    │
    ├─→ y_true = data["true_label"].values ──→ 仅供评估 ──────────┐
    ├─→ X = data.drop(columns=["true_label"])                      │
    │      │                                                        │
    │      ├─→ scaler.fit_transform(X) ──→ X_scaled ──┐             │
    │      │                                           │             │
    │      │   train_model(X_scaled) ──→ model        │             │
    │      │      │                                    │             │
    │      │      └─→ model.predict(X_scaled) ──→ labels_pred ──┐  │
    │      │                                                      │  │
    │      │   plot_clusters(X_scaled, labels_pred, y_true, ...) ←┘  │
    │      │        └─────────────────────────────────────────────────┘
    │      │
    │      └──────────────────────────────────────────────────────────┘
```

### 理解重点

- `y_true` 是一个独立的横向数据流——从数据提取阶段直接流向可视化，完全不经过训练和预测。
- 没有 `train_test_split` 分支——聚类在整个数据集上训练和评估。
- 与 KMeans 的数据依赖图结构一致——只是 `train_model` 的输入参数不同（无 `y_train`）。

## 4. 输出文件一览

### 参数速览

| 输出项 | 路径 | 格式 | 说明 |
|---|---|---|---|
| 聚类分布图 | `outputs/gmm/data_cluster_distribution.png` | PNG | 双面板对比——左：EM 预测标签 / 右：真实分量标签 |
| 终端日志 | 标准输出 | 文本 | 训练超参数 + 对数似然下界 + 运行耗时 |

### 示例代码

```bash
python -m pipelines.probabilistic.em
```

### 输出

```text
============================================================
EM (GMM) 聚类流水线
============================================================
模型训练完成
n_components: 3
covariance_type: full
log-likelihood: -2.1457
模型训练耗时: 0.15s

============================================================
EM (GMM) 流水线完成！
============================================================
```

### 理解重点

- EM 输出 1 个 PNG 文件——与 KMeans/DBSCAN 相同（都是聚类图），但多了 `labels_true` 对比面板。
- 训练耗时通常极短（亚秒级）——500 样本 × 2 维 × 3 分量，EM 收敛很快。
- 终端日志打印 `log-likelihood`——这是 EM 独有的诊断输出，KMeans 和 DBSCAN 都没有。

## 5. 训练层细节：与 KMeans 的对比

| 工程维度 | KMeans | EM (GMM) |
|---|---|---|
| 模型类 | `KMeans` | **`GaussianMixture`** |
| 核心参数 | `n_clusters`、`init`、`n_init` | **`n_components`、`covariance_type`、`max_iter`** |
| 训练输入 | `fit(X)`——无 `y` | `fit(X)`——无 `y` |
| 预测输出 | `predict(X)` ☑ `predict_proba` ☒ | `predict(X)` ☑ `predict_proba(X)` ☑ |
| 模型属性 | `cluster_centers_`、`inertia_`、`labels_` | **`means_`、`covariances_`、`weights_`、`lower_bound_`** |
| 装饰器 | 无 | `@print_func_info` + `@timeit` + `timer` |
| 日志 | `n_clusters`、`inertia_` | **`n_components`、`covariance_type`、`log-likelihood`** |

### 理解重点

- EM 的参数体系比 KMeans 多一个关键维度——`covariance_type` 控制簇形状的灵活性。
- EM 的输出比 KMeans 更丰富——多概率输出（`predict_proba`）和概率模型组件（`means_`、`covariances_`、`weights_`）。
- EM 的训练初始化依赖 KMeans（`init_params="kmeans"`）——两者在工程上是合作关系。

## 阅读顺序

1. `data_generation/probabilistic.py` — 了解 `em()` 的 GMM 数据合成逻辑
2. `model_training/probabilistic/em.py` — 理解 `GaussianMixture` 的构建和 EM 迭代训练
3. `pipelines/probabilistic/em.py` — 看清无监督聚类端到端流程
4. `result_visualization/cluster_plot.py` — 了解聚类双面板对比图实现

## 常见坑

1. 在调用 `drop("true_label")` 之前未提取 `y_true`——`true_label` 列被丢弃后将无法用于可视化对比。
2. 把 `train_model` 当成有监督训练——它接收的参数只有 `X_train`，无 `y_train`。
3. 直接修改 `em_data` 而不先 `copy()`——污染全局变量。
4. 在测试集上使用 `fit_transform`——EM 的聚类场景下没有测试集概念，但如果在其他场景误用，仍然会造成信息泄露。

## 小结

- EM 工程实现遵循本仓库标准四层架构：数据生成层 → 模型训练层 → 流水线编排层 → 可视化层（聚类图模块）。
- `run()` 是极简编排函数——5 步完成标签提取、特征分离、标准化、训练、预测和可视化。
- 与 KMeans/DBSCAN 的核心工程共同点：同为无监督聚类（无 `y_train`、无 `train_test_split`）；核心差异：EM 有更丰富的概率输出（`predict_proba`、`means_`、`covariances_`、`weights_`、`lower_bound_`）。
