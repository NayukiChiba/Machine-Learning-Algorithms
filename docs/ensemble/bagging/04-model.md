---
title: Bagging 与随机森林 — 模型构建
outline: deep
---

# 模型构建

> 对应代码：`model_training/ensemble/bagging.py`
>  
> 运行方式：`python -m model_training.ensemble.bagging`

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `BaggingClassifier`。
2. 理解当前源码中基学习器和采样相关参数的默认值与作用。
3. 看清训练函数除了 `fit(...)` 之外还做了哪些工程封装和版本兼容处理。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `sklearn.ensemble.BaggingClassifier` 模型 |
| `DecisionTreeClassifier(...)` | 类 | 当前 Bagging 使用的基学习器 |
| `BaggingClassifier(...)` | 类 | scikit-learn 提供的 Bagging 分类器 |
| `model.oob_score_` | 属性 | 当前启用 OOB 时的袋外得分 |
| `estimator` / `base_estimator` | 参数 | 不同 sklearn 版本下的兼容构造方式 |

## 1. `train_model(...)` 的函数签名

### 参数速览（本节）

适用函数：`train_model(X_train, y_train, n_estimators=80, max_samples=0.8, max_features=1.0, bootstrap=True, oob_score=True, random_state=42)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_train` | 标准化后的训练特征 | 输入给 `BaggingClassifier.fit(...)` 的训练矩阵 |
| `y_train` | 训练标签 | 二分类目标 |
| `n_estimators` | `80` | 基学习器数量 |
| `max_samples` | `0.8` | 每棵基学习器的样本采样比例 |
| `max_features` | `1.0` | 每棵基学习器的特征采样比例 |
| `bootstrap` | `True` | 是否启用有放回采样 |
| `oob_score` | `True` | 是否计算 OOB 得分 |
| `random_state` | `42` | 随机种子 |
| 返回值 | `BaggingClassifier` | 已训练完成的模型对象 |

### 示例代码

```python
from model_training.ensemble.bagging import train_model

model = train_model(X_train_s, y_train)
```

### 理解重点

- 当前训练入口返回的是单个 Bagging 分类模型，而不是多个独立模型集合。
- 和 GBDT、XGBoost 不同，这里的重点是采样和并行集成，而不是串行 boosting 配置。
- 默认参数直接来自源码，是后续理解 Bootstrap 与 OOB 行为的基线。

## 2. 基学习器的实际构建方式

### 参数速览（本节）

适用 API：`DecisionTreeClassifier(...)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `max_depth` | `None` | 当前不限制树深度 |
| `min_samples_split` | `2` | 节点继续分裂的最小样本数 |
| `min_samples_leaf` | `1` | 叶子节点最小样本数 |
| `random_state` | `42` | 基学习器随机种子 |

### 示例代码

```python
base = DecisionTreeClassifier(
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=random_state,
)
```

### 理解重点

- 当前源码故意选择了一个较强、较容易过拟合的决策树作为基学习器。
- 这是为了让 Bagging 的降方差效果更明显。
- 如果基学习器本身太弱，Bagging 的优势通常就不够突出。

## 3. `BaggingClassifier(...)` 的实际构建方式

### 参数速览（本节）

适用 API（分项）：

1. `BaggingClassifier(...)`
2. `model.fit(X_train, y_train)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n_estimators` | `80` | 并行基学习器数量 |
| `max_samples` | `0.8` | 每棵树训练时可见样本比例 |
| `max_features` | `1.0` | 每棵树训练时可见特征比例 |
| `bootstrap` | `True` | 启用 Bootstrap 重采样 |
| `oob_score` | `True` | 训练后计算袋外得分 |
| `n_jobs` | `-1` | 使用全部可用 CPU 核心 |

### 示例代码

```python
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
```

### 理解重点

- 这些参数共同决定每棵基学习器到底看到多少样本、多少特征，以及是否通过有放回采样制造差异。
- 与随机森林相比，当前实现没有额外引入更强的随机特征选择逻辑，`max_features=1.0` 表示默认使用全部特征。
- 这也说明当前分册要聚焦 Bagging 本体，而不是把随机森林扩展内容误写成当前实现。

## 4. sklearn 版本兼容逻辑

### 参数速览（本节）

适用逻辑（分项）：

1. `estimator=base`
2. `base_estimator=base`

| 分支 | 当前行为 | 说明 |
|---|---|---|
| 新版本 sklearn | 使用 `estimator=base` | 当前推荐写法 |
| 旧版本 sklearn | 回退到 `base_estimator=base` | 保持兼容性 |

### 示例代码

```python
try:
    model = BaggingClassifier(estimator=base, ...)
except TypeError:
    model = BaggingClassifier(base_estimator=base, ...)
```

### 理解重点

- 当前源码不是只考虑一个 sklearn 版本，而是显式做了兼容处理。
- 这属于工程实现细节，不是算法本身差异。
- 文档需要把这点写清楚，否则读者可能误以为这是两种不同模型形式。

## 5. 训练阶段的工程封装

除了 `BaggingClassifier(...).fit(...)` 之外，`train_model(...)` 还做了多层工程包装。

### 参数速览（本节）

适用装饰与上下文（分项）：

1. `@print_func_info`
2. `@timeit`
3. `with timer(name='模型训练耗时')`

| 包装项 | 作用 |
|---|---|
| `@print_func_info` | 打印函数调用入口 |
| `@timeit` | 打印整个函数耗时 |
| `timer(...)` | 打印模型 `fit(...)` 阶段耗时 |

### 理解重点

- 当前训练函数会同时打印“模型训练耗时”和整个函数耗时，因此终端里会看到两层计时信息。
- 这些包装不改变 Bagging 训练行为，但有助于观察训练入口和耗时表现。
- 和线性回归或决策树相比，这里更强调采样配置和 OOB 结果日志。

## 6. `OOB 得分` 在当前实现里表示什么

### 参数速览（本节）

适用属性：`model.oob_score_`

| 属性名 | 当前含义 |
|---|---|
| `oob_score_` | 用袋外样本估计得到的额外参考得分 |

### 示例代码

```python
if oob_score:
    print(f"OOB 得分: {model.oob_score_:.4f}")
```

### 理解重点

- `OOB 得分` 是 Bagging 分册最有代表性的工程输出之一。
- 它的前提是启用了 `bootstrap=True` 且 `oob_score=True`。
- 当前文档要明确它是“额外参考估计”，不能把它误写成完整测试集指标替代物。

## 常见坑

1. 把当前标题中的“随机森林”误解成当前代码实现里已经训练了随机森林。\n+2. 忽略 `OOB 得分` 的前提条件，误以为任何 Bagging 配置都会自然得到这个值。
3. 把 `estimator` / `base_estimator` 的差异当成算法区别，而不是 sklearn 版本兼容问题。

## 小结

- `train_model(...)` 是本仓库 Bagging 的核心训练入口。
- 它本质上是对 `sklearn.ensemble.BaggingClassifier` 的薄封装，重点在于基学习器定义、采样配置、OOB 估计和版本兼容处理。
- 读懂这一层之后，再看流水线中的训练、预测和评估过程会更清晰。
