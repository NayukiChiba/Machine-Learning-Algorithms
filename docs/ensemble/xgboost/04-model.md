---
title: XGBoost — 模型构建
outline: deep
---

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `XGBRegressor`——注意这是回归模型，非分类。
2. 理解 `XGBRegressor` 的核心构造器参数（`n_estimators`、`max_depth`、`gamma`、`reg_lambda`、`min_child_weight`）及其与 GBDT/LightGBM 的差异。
3. 看清训练完成后最重要的模型属性——`feature_importances_`（特征重要性）、`n_estimators_`（实际树数）。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `xgboost.XGBRegressor` 回归模型——含可选依赖检查 |
| `XGBRegressor(...)` | 类 | XGBoost 的 scikit-learn 兼容回归接口——二阶泰勒展开 + 显式正则化 |
| `model.fit(X_train, y_train)` | 方法 | 训练 300 棵回归树——二阶目标近似 + 加权分位数草图 + 列块并行 |
| `model.feature_importances_` | 属性 | 8 个特征的重要性分数——基于分裂增益累加 |
| `model.predict(X)` | 方法 | 300 棵树加权累加——输出连续房价预测值 |

## 1. `train_model(...)` 的函数签名

### 参数速览

适用函数：`train_model(X_train, y_train, n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=1, subsample=0.9, colsample_bytree=0.9, gamma=0.0, reg_alpha=0.0, reg_lambda=1.0, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like`，形状 `(16512, 8)` | 训练特征矩阵（**无标准化**——树模型天然尺度不敏感） | `X_train` |
| `y_train` | `array_like`，形状 `(16512,)` | 连续回归目标——房屋中位价 | `y_train` |
| `n_estimators` | `int` | 弱学习器数量。当前 `300`——与 LightGBM 一致 | `100`、`300`、`500` |
| `learning_rate` | `float` | 学习率（收缩因子）。`0.05`——每次只修正残差的 5% | `0.01`、`0.05`、`0.1` |
| `max_depth` | `int` | 树的最大深度。`6`——深于 GBDT（3），浅于完全生长 | `3`、`6`、`10` |
| `min_child_weight` | `int` | 叶子节点的最小 Hessian 和。`1`——MSE 下等价于最小样本数 | `1`、`5`、`10` |
| `subsample` | `float` | 行采样比例。`0.9`——每轮迭代随机保留 90% 训练样本 | `0.5`、`0.9`、`1.0` |
| `colsample_bytree` | `float` | 列采样比例。`0.9`——每棵树随机选择 90% 的特征（≈7/8） | `0.3`、`0.9`、`1.0` |
| `gamma` | `float` | 分裂所需的最小损失下降。`0.0`——不设最低增益门槛 | `0.0`、`0.1`、`1.0` |
| `reg_alpha` | `float` | L1 正则化系数。`0.0`——不启用 L1 稀疏 | `0.0`、`0.1`、`1.0` |
| `reg_lambda` | `float` | L2 正则化系数。`1.0`——**默认开启**，抑制叶子权重过大 | `0.0`、`1.0`、`10.0` |
| `random_state` | `int` | 随机种子。`42` | `42` |
| 返回值 | `XGBRegressor` | 已完成 `fit()` 的回归模型对象 | — |

### 示例代码

```python
from model_training.ensemble.xgboost import train_model

model = train_model(X_train, y_train)
```

### 理解重点

- `train_model(...)` 是有监督回归训练——`y_train` 是连续值房价，不是离散类别标签。
- XGBoost 的 `max_depth=6` 深于 GBDT（3）但远浅于 Bagging 的完全生长树——在偏差和方差间取平衡。
- `reg_lambda=1.0` 是 XGBoost 独有的默认值——其他 Boosting 实现默认不开启 L2 正则化。
- `min_child_weight=1` 在回归中等于"每个叶子至少 1 个样本"——因为 Hessian 恒为 1。实际上相当于 `min_samples_leaf=1`。

## 2. `XGBRegressor` 构造器参数

### 参数速览

适用 API：`XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=1, subsample=0.9, colsample_bytree=0.9, gamma=0.0, reg_alpha=0.0, reg_lambda=1.0, random_state=42, n_jobs=-1)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_estimators` | `int` | 弱学习器数量。`300`——步数更多但每步更小 | `100`、`300`、`500` |
| `learning_rate` | `float` | 学习率。`0.05`——越小越需更多树 | `0.01`、`0.05`、`0.1` |
| `max_depth` | `int` | 树的最大深度。`6`——适中深度，防止过拟合 | `3`、`6`、`10` |
| `min_child_weight` | `int` | 叶子节点的最小 Hessian 和。`1` | `1`、`5`、`10` |
| `subsample` | `float` | 行采样比例。`0.9` | `0.5`、`0.8`、`0.9` |
| `colsample_bytree` | `float` | 列采样比例。`0.9`——8 个特征中约 7 个用于每棵树 | `0.3`、`0.8`、`0.9` |
| `gamma` | `float` | 分裂最小增益。`0.0`——不设门槛 | `0.0`、`0.1`、`1.0` |
| `reg_alpha` | `float` | L1 正则化。`0.0`——不启用 L1 稀疏 | `0.0`、`0.1` |
| `reg_lambda` | `float` | L2 正则化。`1.0`——**默认开启**，抑制大权重 | `0.0`、`1.0` |
| `random_state` | `int` | 随机种子。`42` | `42` |
| `n_jobs` | `int` | 并行线程数。`-1` 使用所有 CPU——列块并行 | `-1`、`1`、`4` |
| `verbosity` | `int` | 日志级别。默认 `1`（warning） | `0`、`1`、`2` |

### 示例代码

```python
try:
    from xgboost import XGBRegressor
except ImportError:
    raise ImportError("请先 pip install xgboost")

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
```

### 理解重点

- XGBoost 的参数列表是四个集成模型中最长的——体现了它在正则化和精确控制上的设计理念。
- `gamma` 是 XGBoost 独有的预剪枝参数——区别于 `max_depth`（硬深度限制）和 `min_child_weight`（叶子样本数限制）。
- 三重正则化（gamma + reg_lambda + reg_alpha）作用于不同层级——gamma 控分裂是否发生，lambda 控叶子权重是否过大，alpha 控无关权重是否置零。

## 3. 训练完成后的关键属性

### 参数速览

| 属性名 | 类型 | 说明 |
|---|---|---|
| `feature_importances_` | `ndarray`，形状 `(8,)` | 8 个特征的重要性分数——基于分裂增益累加（`gain`） |
| `n_estimators_` | `int` | 实际训练的树数量——等于 `n_estimators=300` |
| `n_features_in_` | `int` | 特征维度——当前为 `8` |
| `best_iteration_` | `int` | 早停最优迭代轮次（启用 `early_stopping_rounds` 时可用） |

### 示例代码

```python
print(f"n_estimators: {n_estimators}")
print(f"learning_rate: {learning_rate}")
print(f"max_depth: {max_depth}")
print(f"min_child_weight: {min_child_weight}")
print(f"subsample: {subsample}")
print(f"colsample_bytree: {colsample_bytree}")
print(f"gamma: {gamma}")
print(f"reg_alpha: {reg_alpha}")
print(f"reg_lambda: {reg_lambda}")
print(f"特征重要性: {model.feature_importances_}")
```

### 理解重点

- `feature_importances_` 默认使用 `gain`（分裂增益累加）——与 LightGBM 一致，不同于 sklearn GBDT 的 impurity 下降量。
- 在加州房价数据上，`MedInc`（收入中位数）通常是最重要的特征——收入是房价的主要驱动力，符合直觉。
- XGBoost 没有 `predict_proba`——回归输出为连续值，不是概率分布。

## 4. `predict()` — 预测连续值

### 参数速览

| 方法 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `predict(X)` | `array_like`，形状 `(n, 8)` | `ndarray`，形状 `(n,)`，连续值 | 300 棵树加权累加——直接输出房价预测值 |

### 理解重点

- `predict()` 返回连续实数——即房屋中位价的预测值（单位：10 万美元）。
- 与分类集成模型不同——没有 `predict_proba`，没有 softmax，没有 argmax。
- 预测值 = $\sum_{m=1}^{300} \eta \cdot f_m(\mathbf{x})$——300 棵树的加权累加。

## 5. XGBoost vs GBDT vs LightGBM 参数对比

| 参数 | GBDT (sklearn) | LightGBM | XGBoost |
|---|---|---|---|
| 任务 | 分类 | 分类 | **回归** |
| `n_estimators` | 200 | 300 | 300 |
| `learning_rate` | 0.1 | 0.05 | 0.05 |
| 复杂度控制 | `max_depth=3` | `num_leaves=31` | **`max_depth=6`** |
| 最小叶子 | — | `min_child_samples=20` | **`min_child_weight=1`** |
| 行采样 | `subsample=1.0` | `subsample=0.9` | `subsample=0.9` |
| 列采样 | 无 | `colsample_bytree=0.9` | `colsample_bytree=0.9` |
| 分裂门槛 | — | — | **`gamma=0.0`** |
| L1 正则化 | — | — | **`reg_alpha=0.0`** |
| L2 正则化 | — | — | **`reg_lambda=1.0`** |
| 依赖 | sklearn 内置 | `pip install lightgbm` | `pip install xgboost` |

## 常见坑

1. 把 `min_child_weight=1` 理解成"最小样本数为 1"——对非 MSE 损失函数，Hessian 不是常数，两者不等价。
2. 忘记 `reg_lambda=1.0` 默认开启——如果感觉模型欠拟合，尝试降为 0.0。
3. 把 `gamma` 和 `reg_alpha` 功能混淆——gamma 做分裂级剪枝，alpha 做权重级稀疏化。
4. 在新环境中直接 `from model_training.ensemble.xgboost import train_model`——需先 `pip install xgboost`。

## 小结

- `train_model(...)` 是本仓库 XGBoost 的核心训练入口，是对 `xgboost.XGBRegressor` 的薄封装——含可选依赖检查和 12 个可配置参数。
- `XGBRegressor` 的核心参数体系是四个集成模型中最丰富的——`n_estimators`（树数量）、`learning_rate`（学习率）、`max_depth`（深度）、`min_child_weight`（最小 Hessian 和）、`gamma`（分裂门槛）、`reg_lambda`（L2）、`reg_alpha`（L1）——构成三层正则化体系。
- 训练完成后核心属性：`feature_importances_`（8 个特征按增益排序）——是回归场景下理解特征贡献的关键诊断工具。
