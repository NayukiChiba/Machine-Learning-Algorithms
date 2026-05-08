---
title: GBDT 梯度提升树 — 模型构建
outline: deep
---

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `GradientBoostingClassifier`。
2. 理解 `GradientBoostingClassifier` 的核心构造器参数（`n_estimators`、`learning_rate`、`max_depth`、`subsample`）及其数学对应关系。
3. 看清训练完成后最重要的模型属性——`feature_importances_`（特征重要性）、`estimators_`（弱学习器列表）、`n_estimators_`（实际迭代数）。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `sklearn.ensemble.GradientBoostingClassifier` 模型，打印超参数日志 |
| `GradientBoostingClassifier(...)` | 类 | scikit-learn 提供的 GBDT 分类器——通过串行梯度提升实现偏差缩减 |
| `model.fit(X_train, y_train)` | 方法 | 串行训练 $M$ 个弱学习器——每棵新树拟合前序集成的负梯度 |
| `model.feature_importances_` | 属性 | 特征重要性——基于分裂增益的加权平均 |
| `model.estimators_` | 属性 | 弱学习器集合——$M$ 个已完成 `fit()` 的浅层 `DecisionTreeRegressor` 对象（注意是回归树） |

## 1. `train_model(...)` 的函数签名

### 参数速览

适用函数：`train_model(X_train, y_train, n_estimators=200, learning_rate=0.1, max_depth=3, subsample=1.0, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like`，形状 `(400, 8)` | 标准化后的训练特征矩阵，传入 `GradientBoostingClassifier.fit()` | `X_train_s` |
| `y_train` | `array_like`，形状 `(400,)` | 训练标签 $\{0, 1, 2\}$——三分类监督信息 | `y_train` |
| `n_estimators` | `int` | 弱学习器（提升阶段）数量。`200`——配合 `learning_rate=0.1` 的经典配置 | `50`、`100`、`200`、`500` |
| `learning_rate` | `float` | 学习率——每棵树的贡献缩放因子。`0.1` 是经验默认值 | `0.01`、`0.05`、`0.1`、`1.0` |
| `max_depth` | `int` | 基学习器的最大深度。`3`——浅层树，高偏差低方差 | `1`、`3`、`5` |
| `subsample` | `float` | 随机梯度提升的采样比例。`1.0` 表示使用全部样本 | `0.5`、`0.8`、`1.0` |
| `random_state` | `int` | 随机种子，保证训练可复现。默认 `42` | `42` |
| 返回值 | `GradientBoostingClassifier` | 已完成 `fit()` 的模型对象，含 `feature_importances_`、`estimators_` 等 | — |

### 示例代码

```python
from model_training.ensemble.gbdt import train_model

model = train_model(X_train_s, y_train)
```

### 理解重点

- `train_model(...)` 是有监督训练——**必须有 `y_train` 参数**。GBDT 比 Bagging 更依赖标签——每棵树的拟合目标（负梯度）由标签和前序预测共同决定。
- `n_estimators=200` 和 `learning_rate=0.1` 是经典组合——200 次迭代、每次修正 10%，总修正量充足但不过量。
- 与 Bagging 的 `train_model` 对比：GBDT 没有 `bootstrap` 和 `oob_score` 参数（它们属于 Bagging 独有），但有 `learning_rate`（GBDT 独有）。

## 2. `GradientBoostingClassifier` 构造器参数

### 参数速览

适用 API：`GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, subsample=1.0, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_estimators` | `int` | 提升阶段数（弱学习器数量）。`200`——GBDT 的核心参数，配合学习率决定总修正量 | `50`、`100`、`200` |
| `learning_rate` | `float` | 学习率——每棵树的贡献缩放因子。$\nu \in (0, 1]$，越小越保守 | `0.01`、`0.1`、`1.0` |
| `max_depth` | `int` | 基学习器最大深度。`3`——浅层树，GBDT 标注配置 | `1`、`3`、`5` |
| `subsample` | `float` | 每棵树使用的样本比例。`1.0`——不启用随机梯度提升 | `0.5`、`0.8`、`1.0` |
| `loss` | `str` | 损失函数。默认 `'log_loss'`——多分类对数损失（交叉熵） | `'log_loss'`、`'exponential'` |
| `min_samples_split` | `int` | 内部节点再分裂的最小样本数。默认 `2` | `2`、`5`、`10` |
| `min_samples_leaf` | `int` | 叶节点的最小样本数。默认 `1` | `1`、`5` |
| `random_state` | `int` | 随机种子——保证训练可复现 | `42` |
| `verbose` | `int` | 日志详细程度。默认 `0` | `0`、`1` |

### 示例代码

```python
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    subsample=1.0,
    random_state=42,
)
model.fit(X_train, y_train)
```

### 理解重点

- `n_estimators` 和 `learning_rate` 是 GBDT 最重要的两个参数——它们共同决定总修正量：$M \times \nu$ 约等于"有效迭代次数"。`200 × 0.1 = 20` 等效步长。
- `max_depth=3` 是 GBDT 的标志性配置——与 Bagging 的 `max_depth=None` 形成鲜明对比。浅层树的高偏差是 GBDT 降偏差的前提。
- `subsample=1.0`（默认）——当前未启用随机梯度提升。若设为 `0.8`，每棵树随机使用 80% 样本，可同时获得方差缩减的额外收益。
- 与 Bagging 的参数对比：Bagging 有 `bootstrap`、`oob_score`、`n_jobs`（并行），GBDT 有 `learning_rate`、`loss`（学习率和损失函数）。

## 3. 训练完成后的关键属性

### 参数速览

| 属性名 | 类型 | 数学含义 | 说明 |
|---|---|---|---|
| `feature_importances_` | `ndarray`，形状 `(8,)` | 基于分裂增益加权的特征重要性 | 值越大越重要——可排序绘制柱状图 |
| `estimators_` | `list`，长度 `n_estimators` × `n_classes` | 弱学习器集合（注意：每类一组树） | 三分类下有 $200 \times 3 = 600$ 个回归树对象 |
| `n_estimators_` | `int` | 实际使用的提升阶段数 | 通常等于 `n_estimators`，除非触发早停 |
| `train_score_` | `ndarray`，形状 `(n_estimators_,)` | 每轮迭代后的训练集得分 | 用于诊断是否过拟合 |
| `n_classes_` | `int` | 类别数 | 当前为 `3` |
| `classes_` | `ndarray`，形状 `(3,)` | 类别标签 | `[0, 1, 2]` |
| `n_features_in_` | `int` | 特征维度 $d$ | 当前为 `8` |

### 示例代码

```python
print(f"n_estimators: {n_estimators}")
print(f"learning_rate: {learning_rate}")
print(f"max_depth: {max_depth}")
print(f"subsample: {subsample}")

# 特征重要性（管道外部使用）
importances = model.feature_importances_
for name, imp in zip(feature_names, importances):
    print(f"  {name}: {imp:.4f}")
```

### 理解重点

- `feature_importances_` 是 GBDT 独有的诊断优势——不需要额外的排列重要性或 SHAP 计算，训练完成后直接可用。
- `estimators_` 的结构与 Bagging 不同——多分类 GBDT 内部为每个类别维护一组树，三分类下 `estimators_` 含 $200 \times 3 = 600$ 棵回归树（注意是回归树，不是分类树）。
- `train_score_` 记录了每轮迭代后的训练集得分——可用来绘制训练曲线，判断是否需要更多树或更少树。

## 4. `predict()` 与 `predict_proba()`

### 参数速览

| 方法 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `predict(X)` | `array_like`，形状 `(n, 8)` | `ndarray`，形状 `(n,)`，取值 $\{0, 1, 2\}$ | 200 棵树加权累加后取最大概率类别 |
| `predict_proba(X)` | `array_like`，形状 `(n, 8)` | `ndarray`，形状 `(n, 3)` | softmax 概率输出——用于多分类 ROC 曲线 |

### 理解重点

- `predict()` 不是投票——是 200 棵树加权累加后取 softmax 最大值。
- `predict_proba()` 是 softmax 输出——每行 3 个概率值，和为 1。
- GBDT 始终支持 `predict_proba`——当前流水线直接调用，没有条件检查（不像 Bagging 的 `hasattr` 防御）。

## 常见坑

1. 把 GBDT 的 `n_estimators` 当成 Bagging 的 `n_estimators`——GBDT 需要更多树（200+），因为每棵树只修正一点点。
2. 忽略 `learning_rate` 与 `n_estimators` 的耦合——调整学习率必须同步调整树数量。
3. 把 GBDT 的基学习器设为深层树——`max_depth=None` 会导致串行过拟合极快。
4. 忘记 GBDT 的 `estimators_` 内部是回归树而非分类树——GBDT 拟合的是连续负梯度值，不是离散标签。

## 小结

- `train_model(...)` 是本仓库 GBDT 的核心训练入口，是对 `sklearn.ensemble.GradientBoostingClassifier` 的薄封装。
- `GradientBoostingClassifier` 的核心参数是 `n_estimators`（提升阶段数）、`learning_rate`（学习率收缩）、`max_depth`（基学习器深度）——三者共同决定偏差缩减的程度和泛化能力。
- 基学习器 `max_depth=3`（浅层树）是刻意选择——高偏差是 GBDT 降偏差的前提，与 Bagging 的完全生长树形成对比。
- 训练完成后的核心属性：`feature_importances_`（特征重要性诊断）、`estimators_`（600 棵回归树的集合）、`train_score_`（训练过程得分）——第一个是 GBDT 独有的特征选择工具。
