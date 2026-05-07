---
title: LogisticRegression 逻辑回归分类 — 模型构建
outline: deep
---

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `LogisticRegression`。
2. 理解每个构造器参数的数学含义与调参方向——特别是 `C`（$\lambda = 1/C$）和 `penalty` 的关系。
3. 理解 `coef_`、`intercept_`、`classes_` 在当前源码中的作用。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练 `LogisticRegression`，返回已训练模型 |
| `LogisticRegression(...)` | 构造器 | 创建逻辑回归分类器，通过超参数控制正则化、优化器与收敛条件 |
| `model.fit(X_train, y_train)` | 方法 | 使用 `lbfgs` 优化器最小化 L2 正则化交叉熵损失 |
| `model.classes_` | 属性 | 返回模型识别到的类别标签数组，形状 `(n_classes,)` |
| `model.intercept_` | 属性 | 返回逻辑回归截距 $b$，决定边界偏移量 |
| `model.coef_` | 属性 | 返回各特征对应系数 $\mathbf{w}$，反映特征对正类倾向的影响方向与强弱 |

## 1. `train_model(...)` 的函数签名

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like` | 标准化后的训练特征矩阵，形状 `(n_samples, n_features)`。传入 `model.fit()` | `X_train_s` |
| `y_train` | `array_like` | 训练标签向量，形状 `(n_samples,)`。二分类标签取值为 $\{0, 1\}$ | `y_train` |
| `penalty` | `str` | 正则化类型。`"l2"` 对 $\|\mathbf{w}\|_2^2$ 惩罚；`"l1"` 对 $\|\mathbf{w}\|_1$ 惩罚（稀疏解）；`"elasticnet"` 混合；`None` 不惩罚。当前默认 `"l2"` | `"l2"`、`"l1"`、`None` |
| `C` | `float` | 正则化强度倒数，$\lambda = 1/C$。$C$ 越大 → 正则越弱 → 系数越自由。当前默认 `1.0` | `0.01`、`1.0`、`100.0` |
| `solver` | `str` | 优化器。`"lbfgs"` 拟牛顿法（默认，适合小中型数据）；`"liblinear"` 坐标下降（适合小数据）；`"saga"` 支持 L1 + 弹性网络 + 大数据。当前默认 `"lbfgs"` | `"lbfgs"`、`"liblinear"`、`"saga"` |
| `max_iter` | `int` | 优化器最大迭代次数。默认 `100`，当前取 `1000`——给的比较宽裕，防止未收敛就停止 | `100`、`1000` |
| `class_weight` | `dict`、`str` 或 `None` | 类别权重。`"balanced"` 自动按 $w_k = n / (K \cdot n_k)$ 加权；`None` 各类等权。当前默认 `None` | `None`、`"balanced"`、`{0:0.5, 1:2.0}` |
| `random_state` | `int` | 随机种子，保证数据打乱与优化器初始化的可复现性。当前取 `42` | `42` |
| 返回值 | `LogisticRegression` | 已训练完成的模型对象，含 `coef_`、`intercept_`、`classes_` 等属性 | — |

### 示例代码

```python
from model_training.classification.logistic_regression import train_model

model = train_model(X_train_s, y_train)
```

### 理解重点

- 当前训练入口很直接，只负责训练一个 `LogisticRegression` 模型。
- 和部分实验型代码不同，这里没有参数搜索逻辑，也没有多模型对比。
- 所有默认超参数都写在函数签名里，阅读成本较低，适合作为源码入口。

## 2. `LogisticRegression(...)` 的完整参数

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `penalty` | `str` 或 `None` | 正则化类型。`"l2"` = $\frac{1}{2C}\|\mathbf{w}\|_2^2$（均匀收缩）；`"l1"` = $\frac{1}{C}\|\mathbf{w}\|_1$（稀疏解，部分系数压为 0）；`"elasticnet"` = $\frac{1}{C}(\rho\|\mathbf{w}\|_1 + (1-\rho)\|\mathbf{w}\|_2^2)$；`None` 不惩罚。默认为 `"l2"` | `"l2"`、`"l1"`、`"elasticnet"`、`None` |
| `dual` | `bool` | 对偶或原始形式。仅 `solver='liblinear'` 且 `penalty='l2'` 时对偶形式可用。$n_{\text{samples}} > n_{\text{features}}$ 时应设为 `False`。默认为 `False` | `False`、`True` |
| `tol` | `float` | 优化收敛容忍度。优化器在两次迭代损失变化小于此值时停止。默认为 `1e-4` | `1e-4`、`1e-6` |
| `C` | `float` | 正则化强度倒数，数学上 $\lambda = 1/C$。$\lambda$ 越大 → 系数收缩越强 → 过拟合风险越低。默认为 `1.0` | `0.01`、`1.0`、`100.0` |
| `fit_intercept` | `bool` | 是否计算截距 $b$。`False` 时强制 $b=0$，决策边界过原点。默认为 `True` | `True`、`False` |
| `intercept_scaling` | `float` | 截距缩放因子。仅 `solver='liblinear'` 且 `fit_intercept=True` 时有效。值越大截距的正则化越小。默认为 `1.0` | `1.0`、`10.0` |
| `class_weight` | `dict`、`str` 或 `None` | 类别权重。`None` 各类等权；`"balanced"` 各样本权重 $w_k = n / (K \cdot n_k)$，$n_k$ 为类别 $k$ 的样本数。不均衡数据应关注。默认为 `None` | `None`、`"balanced"`、`{0:0.5, 1:2.0}` |
| `random_state` | `int` | 随机种子，控制数据打乱与优化器初始化。保证相同数据下结果可复现。默认为 `None` | `42` |
| `solver` | `str` | 优化器选择。`"lbfgs"` 拟牛顿法（稳健默认）；`"liblinear"` 坐标下降法（小数据快）；`"newton-cg"` 牛顿法；`"sag"` 随机平均梯度（大数据快）；`"saga"` sag 的改进版（支持稀疏 + elasticnet）。默认为 `"lbfgs"` | `"lbfgs"`、`"liblinear"`、`"saga"` |
| `max_iter` | `int` | 优化器最大迭代次数。当前取 `1000`——比默认 `100` 高很多，因为高维 + 正则化场景可能收敛慢。未收敛时会有 `ConvergenceWarning` | `100`、`500`、`1000` |
| `multi_class` | `str` | 多分类策略。`"auto"` 根据数据自动选择（二分类选 `"ovr"`）；`"ovr"` 一对多；`"multinomial"` 多项逻辑回归（Softmax 交叉熵）。默认为 `"auto"` | `"auto"`、`"ovr"`、`"multinomial"` |
| `warm_start` | `bool` | 是否复用上一次 `fit()` 的解作为初始化。`True` 时适合连续调参。默认为 `False` | `False`、`True` |
| `n_jobs` | `int` 或 `None` | 并行作业数。仅 `multi_class='ovr'` 时有效。`-1` 用全部核心。默认为 `None` | `None`、`-1`、`4` |
| `l1_ratio` | `float` | L1 在 elasticnet 中的混合比例。仅 `penalty='elasticnet'` 时生效。默认为 `None` | `None`、`0.15`、`0.5` |

### 示例代码

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    max_iter=1000,
    class_weight=None,
    random_state=42,
)
model.fit(X_train_s, y_train)
```

### 理解重点

- 仓库没有自己实现交叉熵优化过程，而是直接调用 scikit-learn 的成熟实现。
- **`C` 是正则化强度的倒数**（$\lambda = 1/C$）——$C$ 越大正则越弱，这是逻辑回归文档最容易出错的地方。
- 最值得关注的核心参数：`penalty`、`C`、`solver`、`max_iter`——它们决定"怎么限制模型"和"怎么优化"。
- 当前 `max_iter=1000` 是教科书的保守设置，确保 `lbfgs` 在高维 + 正则化场景下有足够迭代次数收敛。

## 3. 训练完成后最重要的模型属性

### 属性表

| 属性 | 类型 | 数学含义 |
|---|---|---|
| `classes_` | `ndarray` | 模型识别到的类别标签，形状 `(n_classes,)`。当前二分类为 `[0, 1]` |
| `coef_` | `ndarray`，形状 `(1, d)` | 权重系数 $\mathbf{w}$。$w_j > 0$ 表示特征 $j$ 增大推高正类概率，$w_j < 0$ 表示压低正类概率。标准化后各系数可比 |
| `intercept_` | `ndarray`，形状 `(1,)` | 截距 $b$。$b > 0$ 表示在没有特征信息时（$\mathbf{x} = \mathbf{0}$）模型倾向正类 |
| `n_features_in_` | `int` | 训练时的特征维度 $d$。当前为 `6` |
| `n_iter_` | `ndarray`，形状 `(n_classes,)` | 优化器实际迭代次数。如果接近 `max_iter=1000`，说明可能未收敛 |
| `C_` | `float` | 实际使用（而非用户传入）的 $C$ 值。当 `C` 传入 0 或负数时，sklearn 会修正为很小的正数 |
| `penalty_` | `str` | 实际使用的正则化类型。当 `penalty='elasticnet'` 但 `l1_ratio=0` 时会修正为 `'l2'` |

### 示例代码

```python
print(f"类别: {model.classes_.tolist()}")
print(f"截距: {model.intercept_.round(4)}")
print(f"系数: {model.coef_.round(4)}")
print(f"实际迭代次数: {model.n_iter_}")
```

### 理解重点

- `coef_` 和 `intercept_` 是逻辑回归最有价值的训练结果——它们把"线性边界 $\mathbf{w}^T\mathbf{x} + b = 0$"映射成可直接观察的数值。
- 在标准化后的特征空间中，$w_j$ 的大小可以粗略反映特征 $j$ 的相对重要性。
- `n_iter_` 值得关注：如果它等于 `max_iter`，说明优化器在到达最大迭代次数时可能尚未收敛——此时应考虑增大 `max_iter` 或调整 `solver`。

## 4. 训练阶段的工程封装

除了 `LogisticRegression(...).fit(...)` 之外，`train_model(...)` 还做了几层工程包装：

| 输出项 | 作用 |
|---|---|
| 函数调用标题（`@print_func_info`） | 帮助在终端中定位训练入口 |
| 训练耗时（`@timeit`） | 观察 `lbfgs` 优化器的拟合时间 |
| 超参数日志（`penalty`、`C`、`solver`、`max_iter`） | 确认当前训练配置 |
| 类别、截距与系数日志 | 把线性边界参数映射为源码里可直接观察的输出 |

### 理解重点

- 当前封装强调的是教学型可读性，而不是复杂训练框架。
- 与 KNN 的 `fit()`（只建索引）和决策树的 `fit()`（递归划分）不同，逻辑回归的 `fit()` 本质上是迭代优化交叉熵损失——这是训练耗时的主要来源。
- 这一层封装把"构建模型""训练模型""打印结果"收在一个函数里，方便文档和流水线复用。

## 常见坑

1. 把 `C` 的含义写反——`C` 是正则化强度的倒数，$C$ 越大正则越弱，不是"正则化系数"。
2. 只知道可以 `predict(...)`，却忽略 `coef_` 和 `intercept_` 才是理解逻辑回归行为的重要线索。
3. 忘记当前 `X_train` 应该是标准化后的训练特征——原始特征会让系数不可比，且优化收敛困难。
4. 忽略 `n_iter_` 的值——如果它等于 `max_iter`，模型可能未收敛，预测结果不可靠。

## 小结

- `train_model(...)` 是本仓库 Logistic Regression 的核心训练入口，本质上是对 `sklearn.linear_model.LogisticRegression` 的薄封装。
- `LogisticRegression` 的 14 个构造器参数中，`penalty`、`C`、`solver`、`max_iter` 是最核心的四个——它们决定正则化方式和优化行为。
- **核心公式记忆**：损失 = 交叉熵 + $\frac{1}{2C}\|\mathbf{w}\|_2^2$（当 `penalty='l2'`），$\lambda = 1/C$。
- 训练后属性 `coef_`、`intercept_`、`n_iter_` 是后续模型解释与调参的直接数据来源。
