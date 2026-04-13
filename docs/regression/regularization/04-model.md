---
title: 正则化回归 — 模型构建
outline: deep
---

# 模型构建

> 对应代码：`model_training/regression/regularization.py`
>  
> 运行方式：`python -m model_training.regression.regularization`

## 本章目标

1. 明确 `train_model(...)` 如何一次构建并训练三种正则化模型。
2. 理解默认 `alpha`、`l1_ratio`、`feature_names` 等参数在当前源码中的作用。
3. 看清训练函数除了 `fit(...)` 之外还做了哪些工程封装和诊断输出。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练 `Lasso`、`Ridge`、`ElasticNet` |
| `Lasso(...)` | 类 | 带 L1 正则化的线性回归模型 |
| `Ridge(...)` | 类 | 带 L2 正则化的线性回归模型 |
| `ElasticNet(...)` | 类 | 同时含 L1 和 L2 的线性回归模型 |
| `@print_func_info` | 装饰器 | 打印函数调用入口，方便观察训练日志 |

## 1. `train_model(...)` 的函数签名

### 参数速览（本节）

适用函数：`train_model(X_train, y_train, feature_names=None, alphas=None, l1_ratio=0.5, random_state=42)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_train` | 标准化后的训练特征 | 输入给三个模型 `fit(...)` 的训练矩阵 |
| `y_train` | 训练标签 | 回归目标值 |
| `feature_names` | `list(X.columns)` | 用于打印各特征对应系数 |
| `alphas` | `None` | 若为空则使用源码默认正则化强度字典 |
| `l1_ratio` | `0.5` | ElasticNet 的 L1 比例 |
| `random_state` | `42` | 训练中的随机种子 |
| 返回值 | `dict` | 键为模型名，值为已训练模型对象 |

### 示例代码

```python
from model_training.regression.regularization import train_model

models = train_model(X_train_s, y_train, feature_names=feature_names)
```

### 理解重点

- 当前训练入口不是返回单个模型，而是一次返回一个 `models` 字典。
- 这个设计使得后续流水线可以用统一方式依次预测和画图。
- `feature_names` 不是训练必须参数，但对阅读日志非常重要。

## 2. 默认超参数来自哪里

### 参数速览（本节）

适用默认配置（本节）：

1. `alphas = {"lasso": 0.15, "ridge": 2.0, "elasticnet": 0.2}`
2. `l1_ratio = 0.5`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `alphas["lasso"]` | `0.15` | Lasso 的正则化强度 |
| `alphas["ridge"]` | `2.0` | Ridge 的正则化强度 |
| `alphas["elasticnet"]` | `0.2` | ElasticNet 的正则化强度 |
| `l1_ratio` | `0.5` | ElasticNet 中 L1 与 L2 的混合比例 |

### 示例代码

```python
if alphas is None:
    alphas = {"lasso": 0.15, "ridge": 2.0, "elasticnet": 0.2}
```

### 理解重点

- 这三个 `alpha` 并不相同，说明当前实现本来就是为了展示不同模型的典型行为，而不是强行用同一个正则化强度对比。
- `l1_ratio=0.5` 让 ElasticNet 处在一个比较中性的折中位置。
- 这些默认值就是当前文档讨论的基线，调参时应先与它们对照。

## 3. 三个模型的实际构建方式

### 参数速览（本节）

适用 API（分项）：

1. `Lasso(alpha=..., max_iter=10000, random_state=...)`
2. `Ridge(alpha=..., random_state=...)`
3. `ElasticNet(alpha=..., l1_ratio=..., max_iter=10000, random_state=...)`

| 模型 | 当前构建 | 说明 |
|---|---|---|
| `Lasso` | `alpha=0.15, max_iter=10000` | 强调稀疏化，需要足够迭代次数 |
| `Ridge` | `alpha=2.0` | 强调系数整体收缩 |
| `ElasticNet` | `alpha=0.2, l1_ratio=0.5, max_iter=10000` | 折中处理稀疏性与稳定性 |

### 示例代码

```python
models = {
    "Lasso": Lasso(
        alpha=alphas["lasso"], max_iter=10000, random_state=random_state
    ),
    "Ridge": Ridge(alpha=alphas["ridge"], random_state=random_state),
    "ElasticNet": ElasticNet(
        alpha=alphas["elasticnet"],
        l1_ratio=l1_ratio,
        max_iter=10000,
        random_state=random_state,
    ),
}
```

### 理解重点

- 仓库没有自己实现优化器，而是直接调用 scikit-learn 的现成模型。
- 当前文档的重点，是看这三种模型在同一份数据、同一套预处理下如何并排比较。
- `max_iter=10000` 只出现在 Lasso 和 ElasticNet 上，因为它们依赖迭代优化更明显。

## 4. `feature_names` 与日志打印的作用

### 参数速览（本节）

适用逻辑（分项）：

1. 自动生成占位特征名
2. 打印每个特征的系数

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `feature_names` | `list(X.columns)` | 与系数一一对应的特征名 |
| 自动回退值 | `Feature_0`, `Feature_1`, ... | 当传入数组而非带列名对象时使用 |

### 示例代码

```python
if feature_names is None:
    if hasattr(X_train, "shape"):
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
    else:
        feature_names = []

for f, c in zip(feature_names, coef):
    print(f"  {f}: {c:.3f}")
```

### 理解重点

- 当前实现非常强调“把系数打印出来”，因为这比分数更适合观察正则化行为。
- 如果传入真实列名，就能直接看到 `bmi_corr`、`noise_3` 这类特征是如何被不同模型处理的。
- 如果不传 `feature_names`，模型依然能训练，但文档里最重要的一部分可解释性就会下降。

## 5. 训练阶段的工程封装

除了 `model.fit(X_train, y_train)` 之外，`train_model(...)` 还做了几层日志封装。

### 参数速览（本节）

适用输出项（分项）：

1. `@print_func_info`
2. `model.alpha`
3. `model.intercept_`
4. `near_zero = np.sum(np.abs(coef) < 1e-3)`

| 输出项 | 作用 |
|---|---|
| 函数调用标题 | 帮助在终端中定位训练入口 |
| `alpha` | 回看本次模型使用的正则化强度 |
| `l1_ratio` | 仅对 ElasticNet 额外打印 |
| 截距 | 观察模型的常数项 |
| 接近 0 的系数数量 | 快速判断稀疏化程度 |

### 示例代码

```python
for name, model in models.items():
    model.fit(X_train, y_train)
    coef = model.coef_
    near_zero = np.sum(np.abs(coef) < 1e-3)

    print(f"\n{name} 训练完成")
    print(f"alpha: {model.alpha}")
    if name == "ElasticNet":
        print(f"l1_ratio: {model.l1_ratio}")
    print(f"截距: {model.intercept_:.3f}")
    print(f"接近 0 的系数数量: {near_zero}/{len(coef)}")
```

### 理解重点

- `接近 0 的系数数量` 是当前分册非常关键的工程化诊断项。
- 它不是严格统计“等于 0”的系数，而是统计绝对值小于 `1e-3` 的系数数量，更适合统一比较三种模型。
- ElasticNet 会额外打印 `l1_ratio`，因为这是它区别于 Lasso 和 Ridge 的核心超参数。

## 常见坑

1. 误以为 `train_model(...)` 只训练一个模型，实际它返回的是三模型字典。
2. 只关注 `alpha`，忽略 ElasticNet 的 `l1_ratio` 对行为差异的影响。
3. 训练后只看是否成功运行，不看“接近 0 的系数数量”和具体系数分布。

## 小结

- `train_model(...)` 是本仓库正则化回归的核心训练入口。
- 它本质上是对 `Lasso`、`Ridge`、`ElasticNet` 的并行式薄封装，重点在统一构建、统一训练和统一打印日志。
- 读懂这一层之后，再看流水线中的训练与预测过程会更清晰。
