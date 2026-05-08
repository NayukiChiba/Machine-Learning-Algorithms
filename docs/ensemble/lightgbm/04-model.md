---
title: LightGBM — 模型构建
outline: deep
---

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `LGBMClassifier`（含可选依赖检查）。
2. 理解 `LGBMClassifier` 的核心构造器参数（`n_estimators`、`learning_rate`、`num_leaves`、`subsample`、`colsample_bytree`）及其与 GBDT 的差异。
3. 看清训练完成后最重要的模型属性——`feature_importances_`（特征重要性）、`n_estimators_`（实际树数）、`predict_proba`（概率输出）。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `lightgbm.LGBMClassifier` 模型——含可选依赖检查 |
| `LGBMClassifier(...)` | 类 | LightGBM 的 scikit-learn 兼容接口——Leaf-wise 生长 + 直方图加速的 GBDT 实现 |
| `model.fit(X_train, y_train)` | 方法 | 串行训练 300 棵 Leaf-wise 直方图回归树——支持 GOSS + 直方图分桶 |
| `model.feature_importances_` | 属性 | 特征重要性——基于分裂增益累加，和为 1.0 |
| `model.predict(X)` | 方法 | 加法模型输出——300 棵树加权累加 → softmax → argmax |
| `model.predict_proba(X)` | 方法 | 概率输出——300 棵树加权累加 → softmax |

## 1. `train_model(...)` 的函数签名

### 参数速览

适用函数：`train_model(X_train, y_train, n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1, subsample=0.9, colsample_bytree=0.9, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like`，形状 `(800, 20)` | 标准化后的训练特征矩阵，传入 `LGBMClassifier.fit()` | `X_train_s` |
| `y_train` | `array_like`，形状 `(800,)` | 训练标签 $\{0, 1, 2, 3\}$——四分类监督信息 | `y_train` |
| `n_estimators` | `int` | 弱学习器数量。当前 `300`——步数更多但每步更小（`learning_rate=0.05`） | `100`、`300`、`500` |
| `learning_rate` | `float` | 学习率（收缩因子）。`0.05`——每次只修正残差的 5% | `0.01`、`0.05`、`0.1` |
| `num_leaves` | `int` | 每棵树的最大叶子数。`31`——Leaf-wise 生长的复杂度上限 | `15`、`31`、`63`、`127` |
| `max_depth` | `int` | 树的最大深度。`-1`——不限制深度，Leaf-wise 由 `num_leaves` 控制复杂度 | `-1`、`3`、`7` |
| `subsample` | `float` | 行采样比例。`0.9`——每轮迭代随机保留 90% 的训练样本 | `0.5`、`0.9`、`1.0` |
| `colsample_bytree` | `float` | 列采样比例。`0.9`——每棵树随机选择 90% 的特征 | `0.5`、`0.9`、`1.0` |
| `random_state` | `int` | 随机种子，保证采样和训练可复现。`42` | `42` |
| 返回值 | `LGBMClassifier` | 已完成 `fit()` 的模型对象，含 `feature_importances_`、`estimators_` 等 | — |

### 示例代码

```python
from model_training.ensemble.lightgbm import train_model

model = train_model(X_train_s, y_train)
```

### 理解重点

- `train_model(...)` 是有监督训练——**必须有 `y_train` 参数**。四分类任务下，y 取值 $\{0, 1, 2, 3\}$。
- `n_estimators=300` + `learning_rate=0.05` 是教学平衡选择——总修正量 $300 \times 0.05 = 15$，比 GBDT 的 $200 \times 0.1 = 20$ 更精细。
- `train_model(...)` 内部有 `try/except ImportError` 保护——`lightgbm` 不是 sklearn 标准组件，需要单独安装。

## 2. `LGBMClassifier` 构造器参数

### 参数速览

适用 API：`LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1, subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_estimators` | `int` | 弱学习器数量。默认 `300`——核心超参数，需与 `learning_rate` 配对 | `100`、`300`、`500` |
| `learning_rate` | `float` | 学习率（收缩因子）。`0.05`——越小越需更多树，但泛化更好 | `0.01`、`0.05`、`0.1` |
| `num_leaves` | `int` | 每棵树的最大叶子数。`31`——LightGBM 独有，替代 `max_depth` 控制复杂度 | `15`、`31`、`63` |
| `max_depth` | `int` | 树的最大深度。`-1`——不限制，Leaf-wise 生长由 `num_leaves` 控制 | `-1`、`3`、`7` |
| `subsample` | `float` | 行采样比例。`0.9`——每轮迭代随机保留 90% 样本 | `0.5`、`0.8`、`0.9` |
| `colsample_bytree` | `float` | 列采样比例。`0.9`——每棵树随机选择 90% 的特征 | `0.5`、`0.8`、`0.9` |
| `subsample_freq` | `int` | 行采样频率。默认 `0`（每轮都采样） | `0`、`5` |
| `reg_alpha` | `float` | L1 正则化系数。默认 `0.0` | `0.0`、`0.1` |
| `reg_lambda` | `float` | L2 正则化系数。默认 `0.0` | `0.0`、`0.1` |
| `min_child_samples` | `int` | 叶子节点的最小样本数。默认 `20` | `5`、`20`、`50` |
| `random_state` | `int` | 随机种子。默认 `42` | `42` |
| `n_jobs` | `int` | 并行线程数。`-1` 使用所有 CPU——直方图构建和特征扫描级并行，非基学习器级并行 | `-1`、`1`、`4` |
| `verbosity` | `int` | 日志详细程度。默认 `1` | `0`、`1` |

### 示例代码

```python
try:
    from lightgbm import LGBMClassifier
except ImportError:
    raise ImportError("请先 pip install lightgbm")

model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
```

### 理解重点

- `num_leaves` 是 LightGBM 最独特的参数——它替代 GBDT 的 `max_depth` 作为复杂度控制手段。Leaf-wise 生长从根节点开始，每次选择损失下降最多的叶子分裂，直至达到 `num_leaves`。
- `max_depth=-1` 不是"无限生长"——它表示不设置硬深度上限，但 `num_leaves=31` 已经限定了最大叶子数。
- `n_jobs=-1` 利用了 LightGBM 的并行机制——在直方图构建和特征扫描层面并行，而非基学习器级并行（Boosting 仍串行）。
- 与 GBDT（sklearn）的三个关键参数差异：（1）`num_leaves` 替代 `max_depth`；（2）多了 `colsample_bytree`；（3）`max_depth=-1` 而非 `max_depth=3`。

## 3. 训练完成后的关键属性

### 参数速览

| 属性名 | 类型 | 说明 |
|---|---|---|
| `feature_importances_` | `ndarray`，形状 `(20,)` | 20 个特征的重要性分数——基于分裂增益，和为 1.0 |
| `n_estimators_` | `int` | 实际训练的树数量——等于 `n_estimators=300` |
| `classes_` | `ndarray`，形状 `(4,)` | 类别标签——四分类 `[0, 1, 2, 3]` |
| `n_features_in_` | `int` | 特征维度——当前为 `20` |
| `best_iteration_` | `int` | 早停最优迭代轮次（当启用 `early_stopping_rounds` 时） |

### 示例代码

```python
print(f"n_estimators: {n_estimators}")
print(f"learning_rate: {learning_rate}")
print(f"num_leaves: {num_leaves}")
print(f"max_depth: {max_depth}")
print(f"subsample: {subsample}")
print(f"colsample_bytree: {colsample_bytree}")
print(f"特征重要性 (前 5): {model.feature_importances_[:5]}")
```

### 理解重点

- `feature_importances_` 是 GBDT 系列共有的诊断属性——它提供"哪些特征在真正起作用"的自动诊断能力。
- 20 维特征中，预期 `x1`~`x8`（8 个有效特征）的重要性显著高于 `x9`~`x13`（5 个冗余特征）和 `x14`~`x20`（7 个纯噪声特征）。
- LightGBM 的 `feature_importances_` 默认使用 `gain`（分裂增益）——不同于 sklearn GBDT 默认的 impurity 下降量。

## 4. `predict()` 与 `predict_proba()`

### 参数速览

| 方法 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `predict(X)` | `array_like`，形状 `(n, 20)` | `ndarray`，形状 `(n,)`，取值 $\{0, 1, 2, 3\}$ | 300 棵树加权累加 → softmax → argmax |
| `predict_proba(X)` | `array_like`，形状 `(n, 20)` | `ndarray`，形状 `(n, 4)` | 300 棵树加权累加 → softmax——4 类概率和为 1.0 |

### 理解重点

- `predict()` 是加法模型输出——逐棵树累加（每棵乘以 `learning_rate`），取 softmax 概率最大的类别。
- `predict_proba()` 与 `predict()` 共享前段计算——最后一步分支（`predict` 取 argmax，`predict_proba` 返回概率分布）。
- 与 Bagging 的投票聚合不同——LightGBM 是加权累加，每棵树的权重由 `learning_rate` 隐式决定。

## 常见坑

1. `num_leaves` 设得过大（如 `num_leaves=1024`）——Leaf-wise 生长可能生成极深的树，导致单棵过拟合。
2. 把 `max_depth=-1` 当成"不设防"——`num_leaves` 才是真正的复杂度控制参数，但过深的分支仍可能只含极少数样本。
3. `learning_rate=0.05` 且 `n_estimators=100`——总修正量仅 5，可能欠拟合。
4. 混淆 LightGBM 的 `subsample` 与 Bagging 的 `max_samples`——LightGBM 是均匀随机采样（非 Bootstrap 有放回），且与 GOSS 的思想相通。
5. 忘记 `lightgbm` 是可选依赖——在新环境运行前必须 `pip install lightgbm`。

## 小结

- `train_model(...)` 是本仓库 LightGBM 的核心训练入口，是对 `lightgbm.LGBMClassifier` 的薄封装——含可选依赖检查。
- `LGBMClassifier` 的核心参数是 `n_estimators`（树数量）、`learning_rate`（学习率）、`num_leaves`（Leaf-wise 叶子数）、`subsample`（行采样）、`colsample_bytree`（列采样）——五者共同决定偏差缩减的程度和训练速度。
- 训练完成后的核心属性：`feature_importances_`（自动特征选择诊断）——20 维数据下区分有效/冗余/噪声特征的关键工具。
