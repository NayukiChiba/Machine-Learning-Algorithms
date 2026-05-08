---
title: Bagging 集成学习 — 模型构建
outline: deep
---

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `BaggingClassifier`（含基学习器的创建）。
2. 理解 `BaggingClassifier` 的核心构造器参数（`n_estimators`、`max_samples`、`bootstrap`、`oob_score`）及其数学对应关系。
3. 看清训练完成后最重要的模型属性——`oob_score_`（OOB 得分）、`estimators_`（基学习器列表）、`predict_proba`（概率输出）。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `sklearn.ensemble.BaggingClassifier` 模型，打印 OOB 得分日志 |
| `BaggingClassifier(...)` | 类 | scikit-learn 提供的 Bagging 集成分类器——通过 Bootstrap 采样 + 并行投票实现方差缩减 |
| `DecisionTreeClassifier(...)` | 类 | 基学习器——完全生长的决策树（`max_depth=None`），高方差低偏差 |
| `model.fit(X_train, y_train)` | 方法 | 并行训练 $n$ 个基学习器——每个在各自的 Bootstrap 子集上独立 `fit` |
| `model.oob_score_` | 属性 | OOB 得分——用未参与训练的样本估计泛化能力 |
| `model.predict(X)` | 方法 | 投票聚合——$n$ 个基学习器多数投票决定最终预测 |

## 1. `train_model(...)` 的函数签名

### 参数速览

适用函数：`train_model(X_train, y_train, n_estimators=80, max_samples=0.8, max_features=1.0, bootstrap=True, oob_score=True, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like`，形状 `(400, 2)` | 标准化后的训练特征矩阵，传入 `BaggingClassifier.fit()` | `X_train_s` |
| `y_train` | `array_like`，形状 `(400,)` | 训练标签 $\{0, 1\}$——二分类监督信息 | `y_train` |
| `n_estimators` | `int` | 基学习器数量。当前设为 `80`——在方差缩减与计算成本间取得平衡 | `10`、`50`、`80`、`200` |
| `max_samples` | `float` | 每个 Bootstrap 子集的样本比例。`0.8` 表示子集大小为 $0.8 \times N$ | `0.5`、`0.8`、`1.0` |
| `max_features` | `float` | 每个 Bootstrap 子集的随机特征比例。`1.0` 表示使用全部特征 | `0.5`、`1.0` |
| `bootstrap` | `bool` | 是否使用有放回 Bootstrap 采样。`True`——Bagging 的核心操作 | `True`、`False` |
| `oob_score` | `bool` | 是否启用 OOB 得分估计。`True`——训练后可用 `model.oob_score_` | `True`、`False` |
| `random_state` | `int` | 随机种子，保证 Bootstrap 采样和基学习器可复现。默认 `42` | `42` |
| 返回值 | `BaggingClassifier` | 已完成 `fit()` 的模型对象，含 `oob_score_`、`estimators_` 等 | — |

### 示例代码

```python
from model_training.ensemble.bagging import train_model

model = train_model(X_train_s, y_train)
```

### 理解重点

- `train_model(...)` 是有监督训练——**必须有 `y_train` 参数**。这是分类算法与降维/聚类算法最根本的差异。
- `n_estimators=80` 和 `max_samples=0.8` 是教学平衡选择——80 棵树在秒级完成训练，0.8 的采样比例使子集间有足够差异。
- `train_model(...)` 内部还会创建 `DecisionTreeClassifier` 作为基学习器——它是 Bagging 的核心组件。

## 2. 基学习器：`DecisionTreeClassifier`

Bagging 的方差缩减效果高度依赖基学习器的特性。

### 参数速览

适用 API：`DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `max_depth` | `int` 或 `None` | 树的最大深度。`None`——完全生长，不限制深度（最低偏差、最高方差） | `None`、`5`、`10` |
| `min_samples_split` | `int` | 内部节点再分裂的最小样本数。`2`——允许分裂到极致 | `2`、`5`、`10` |
| `min_samples_leaf` | `int` | 叶节点的最小样本数。`1`——允许叶节点只含一个样本 | `1`、`5` |
| `random_state` | `int` | 随机种子。传入 `BaggingClassifier` 的 `random_state` | `42` |

### 理解重点

- 这三项参数（`None`、`2`、`1`）的组合刻意让每棵树**充分生长、高度敏感**——对 Bootstrap 子集的微小差异产生截然不同的树结构。
- 这正是 Bagging 方差缩减的前提——基学习器方差越大（但偏差保持极低），Bagging 的改善越显著。
- 如果改为 `max_depth=3`（浅层树），Bagging 的改善将非常有限——因为浅层树本身的方差就不高。

## 3. `BaggingClassifier` 构造器参数

### 参数速览

适用 API：`BaggingClassifier(estimator=..., n_estimators=80, max_samples=0.8, max_features=1.0, bootstrap=True, oob_score=True, n_jobs=-1, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `BaseEstimator` | 基学习器对象。当前为完全生长的 `DecisionTreeClassifier`。scikit-learn 1.2+ 使用此参数名，旧版本使用 `base_estimator` | `DecisionTreeClassifier(max_depth=None)` |
| `n_estimators` | `int` | 基学习器数量。默认 `80`——Bagging 的核心参数，越大方差越低但边际递减 | `10`、`80`、`200` |
| `max_samples` | `int` 或 `float` | 每个 Bootstrap 子集的样本数/比例。`0.8` 表示 80% 的样本量 | `0.5`、`0.8`、`1.0` |
| `max_features` | `int` 或 `float` | 每个 Bootstrap 子集的随机特征数/比例。`1.0` 表示使用全部特征 | `0.5`、`1.0` |
| `bootstrap` | `bool` | 是否 Bootstrap 采样。`True`——Bagging 的核心；`False` 时使用全部样本（退化为简单投票） | `True`、`False` |
| `bootstrap_features` | `bool` | 是否对特征也做 Bootstrap 采样。默认 `False` | `False`、`True` |
| `oob_score` | `bool` | 是否计算 OOB 得分。`True`——训练后 `model.oob_score_` 可用 | `True`、`False` |
| `n_jobs` | `int` | 并行作业数。`-1` 使用所有 CPU 核心——Bagging 天然并行 | `-1`、`1`、`4` |
| `random_state` | `int` | 随机种子，保证 Bootstrap 采样可复现。默认 `42` | `42` |
| `verbose` | `int` | 日志详细程度。默认 `0` | `0`、`1` |

### 示例代码

```python
base = DecisionTreeClassifier(
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=random_state,
)

# sklearn 版本兼容
try:
    model = BaggingClassifier(
        estimator=base,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=bootstrap,
        oob_score=oob_score,
        random_state=random_state,
        n_jobs=-1,
    )
except TypeError:
    model = BaggingClassifier(
        base_estimator=base,  # 旧版本参数名
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=bootstrap,
        oob_score=oob_score,
        random_state=random_state,
        n_jobs=-1,
    )
```

### 理解重点

- `n_estimators` 是 Bagging 最重要的参数——它直接控制方差缩减的程度。80 棵树通常已经足够。
- `max_samples=0.8`（而非 `1.0`）使每个子集更小、差异更大——进一步增强了模型多样性。
- `n_jobs=-1` 利用 Bagging 的天然并行性——80 棵树可以同时训练，大幅缩短训练时间。
- 源码中的 `try/except TypeError` 是 sklearn 版本兼容处理——`estimator` 参数名在 1.2 版本从 `base_estimator` 改为 `estimator`。

## 4. 训练完成后的关键属性

### 参数速览

| 属性名 | 类型 | 数学含义 | 说明 |
|---|---|---|---|
| `oob_score_` | `float` | OOB 得分 $\in [0, 1]$ | 用未参与训练的样本估计的泛化准确率——$1 - \text{OOB Error}$。仅 `oob_score=True` 时可用 |
| `estimators_` | `list`，长度 `n_estimators` | 基学习器集合 $\{f_b\}_{b=1}^{n}$ | 80 个已完成 `fit()` 的 `DecisionTreeClassifier` 对象 |
| `estimators_features_` | `list` | 每个基学习器使用的特征子集 | `max_features < 1.0` 时有意义——当前为全特征 |
| `classes_` | `ndarray`，形状 `(2,)` | 类别标签 | 二分类——`[0, 1]` |
| `n_features_in_` | `int` | 特征维度 $d$ | 训练时输入的特征维数，当前为 `2` |
| `oob_decision_function_` | `ndarray` | 各样本在各类别上的 OOB 投票概率 | `oob_score=True` 时可用——提供比 `oob_score_` 更细粒度的信息 |

### 示例代码

```python
print(f"n_estimators: {n_estimators}")
print(f"max_samples: {max_samples}")
print(f"max_features: {max_features}")
print(f"bootstrap: {bootstrap}")
if oob_score:
    print(f"OOB 得分: {model.oob_score_:.4f}")
```

### 理解重点

- `oob_score_` 是 Bagging 独有的输出——它提供了一个无需额外划分验证集的泛化能力估计。当前源码打印到 4 位小数。
- `estimators_` 是集成学习的标志性属性——它存储了所有 80 个基学习器，可用于单独检查或分析。
- 与单模型分类器（如 SVC）的关键对比：Bagging 有 `estimators_`（基学习器集合）和 `oob_score_`（免费泛化估计），单模型分类器没有。

## 5. `predict()` 与 `predict_proba()`

### 参数速览

| 方法 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `predict(X)` | `array_like`，形状 `(n, 2)` | `ndarray`，形状 `(n,)`，取值 $\{0, 1\}$ | 80 棵树投票——多数获胜 |
| `predict_proba(X)` | `array_like`，形状 `(n, 2)` | `ndarray`，形状 `(n, 2)` | 80 棵树预测概率的平均——用于 ROC 曲线 |

### 理解重点

- `predict()` 是硬投票——每棵树投一票，取多数。
- `predict_proba()` 是软投票——取 80 棵树预测概率的平均值。对 ROC 曲线而言，概率输出比硬分类标签更有信息量。
- 当前流水线用 `hasattr(model, "predict_proba")` 做条件判断——`BaggingClassifier` 始终支持 `predict_proba`（只要基学习器支持），但条件判断是防御性工程习惯。

## 常见坑

1. 把基学习器设为低方差模型——Bagging 的方差缩减效果完全依赖基学习器的高方差特性。
2. 忽略 `n_estimators` 的边际递减效应——80→200 的改善远小于 10→80 的改善。
3. 忘记 `oob_score=True`——放弃了 Bagging 独有的免费泛化估计。
4. 混淆 `max_samples` 和 `max_features`——前者控制样本采样比例（每棵树看到的数据量），后者控制特征采样比例（每棵树看到的特征子集）。

## 小结

- `train_model(...)` 是本仓库 Bagging 的核心训练入口，是对 `sklearn.ensemble.BaggingClassifier` 的薄封装。
- `BaggingClassifier` 的核心参数是 `n_estimators`（基学习器数量）、`max_samples`（采样比例）、`bootstrap`（采样方式）、`oob_score`（OOB 估计）——四者共同决定方差缩减的程度和泛化估计的可用性。
- 基学习器 `DecisionTreeClassifier(max_depth=None)` 的配置（完全生长）是刻意选择——高方差是 Bagging 受益的前提。
- 训练完成后的核心属性：`oob_score_`（免费泛化估计）、`estimators_`（基学习器集合）——前者是 Bagging 独有的诊断工具，后者是集成学习的标志性结构。
