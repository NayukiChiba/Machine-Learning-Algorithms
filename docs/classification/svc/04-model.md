---
title: SVC 支持向量分类 — 模型构建
outline: deep
---

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `SVC`。
2. 理解 `SVC` 的核心构造器参数（`C`、`kernel`、`gamma`）及其数学对应关系。
3. 看清训练完成后最重要的模型属性——`n_support_`、`support_vectors_`、`dual_coef_`、`intercept_`。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `sklearn.svm.SVC` 模型，打印训练日志 |
| `SVC(...)` | 类 | scikit-learn 提供的 C-Support Vector Classification——基于 `libsvm` 的成熟实现 |
| `model.fit(X_train, y_train)` | 方法 | 求解对偶优化问题，找出支持向量和决策函数参数 |
| `model.n_support_` | 属性 | 各类别的支持向量数量——量化模型依赖的关键样本规模 |
| `model.support_vectors_` | 属性 | 支持向量的特征矩阵 |
| `model.dual_coef_` | 属性 | 对偶系数与标签的乘积 $\alpha_i y_i$ |

## 1. `train_model(...)` 的函数签名

### 参数速览

适用函数：`train_model(X_train, y_train, C=1.0, kernel='rbf', gamma='scale', random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like` | 标准化后的训练特征矩阵，形状 $(320, 2)$，传入 `SVC.fit()` | `X_train_s` |
| `y_train` | `array_like` | 训练标签向量，二分类取值 $\{0, 1\}$ | `y_train` |
| `C` | `float` | 正则化参数（误分类惩罚系数）。$C$ 越大，间隔越窄、越不容忍误分类。默认 `1.0` | `0.1`、`1.0`、`10.0` |
| `kernel` | `str` | 核函数类型。默认 `'rbf'`，当前同心圆数据的最优选择 | `'linear'`、`'rbf'`、`'poly'` |
| `gamma` | `float` 或 `str` | RBF 核系数。`'scale'`（默认）时 $\gamma = 1/(d \cdot X.var())$；`'auto'` 时 $\gamma = 1/d$ | `'scale'`、`'auto'`、`0.1`、`1.0` |
| `random_state` | `int` | 随机种子，保证概率估计等随机过程可复现。默认 `42` | `42` |
| 返回值 | `SVC` | 已完成 `fit()` 的模型对象，含 `n_support_`、`support_vectors_` 等属性 | — |

### 示例代码

```python
from model_training.classification.svc import train_model

model = train_model(X_train_s, y_train)
```

### 理解重点

- 当前入口很直接：只负责构建一个 RBF 核 `SVC` 并 `fit`，没有多核并行对比或网格搜索。
- 所有默认超参数（`C=1.0`、`kernel='rbf'`、`gamma='scale'`）都写在函数签名里，阅读成本低。
- `train_model(...)` 是对 `sklearn.svm.SVC` 的薄封装——算法本体是 scikit-learn 基于 `libsvm` 的 C++ 实现。

## 2. `SVC` 构造器核心参数

### 参数速览

适用 API：`SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `C` | `float` | 软间隔惩罚系数，对应目标函数 $C\sum\xi_i$。$C \uparrow$ → 间隔变窄、更关注训练精度；$C \downarrow$ → 间隔变宽、更关注泛化。默认 `1.0` | `0.1`、`1.0`、`10.0`、`100.0` |
| `kernel` | `str` | 核函数类型。`'linear'`、`'poly'`、`'rbf'`、`'sigmoid'` 或 `'precomputed'`。默认 `'rbf'` | `'rbf'`、`'linear'`、`'poly'` |
| `degree` | `int` | 多项式核的次数 $d$，仅当 `kernel='poly'` 时生效。默认 `3` | `2`、`3`、`4` |
| `gamma` | `float` 或 `str` | 核系数，控制单个训练样本的影响半径。`'scale'`（默认）时 $\gamma = 1/(n\_features \cdot X.var())$；`'auto'` 时 $\gamma = 1/n\_features$；传入 `float` 直接使用。$\gamma \uparrow$ → 影响半径缩小、边界更精细弯曲 | `'scale'`、`'auto'`、`0.01`、`1.0`、`10.0` |
| `coef0` | `float` | 核函数中的独立项 $r$，仅对 `'poly'` 和 `'sigmoid'` 核生效。默认 `0.0` | `0.0`、`1.0` |
| `probability` | `bool` | 是否启用概率估计。`True` 时会在训练后额外做 5 折交叉验证 Platt scaling，显著增加训练耗时。默认 `False` | `False`、`True` |
| `shrinking` | `bool` | 是否使用收缩启发式加速优化。默认 `True` | `True` |
| `tol` | `float` | 优化停止容差。默认 `1e-3` | `1e-3`、`1e-4` |
| `cache_size` | `float` | 核矩阵缓存大小（MB）。默认 `200` | `200`、`500` |
| `max_iter` | `int` | 求解器最大迭代次数。`-1` 表示无限制。默认 `-1` | `-1`、`1000` |
| `decision_function_shape` | `str` | 多分类决策函数形状。`'ovr'`（One-vs-Rest）或 `'ovo'`（One-vs-One）。默认 `'ovr'` | `'ovr'`、`'ovo'` |
| `random_state` | `int` | 随机种子，控制概率估计等随机过程。当前设为 `42` | `42` |

### 示例代码

```python
model = SVC(C=1.0, kernel="rbf", gamma="scale", random_state=42)
model.fit(X_train, y_train)
```

### 理解重点

- SVC 的参数主要集中在核函数配置上——`kernel`、`gamma`、`degree`、`coef0` 都与非线性映射相关。
- `C` 和 `gamma` 是最需要关注的超参数组合：$C$ 控制容错，$\gamma$ 控制核局部性——两者共同决定模型复杂度。
- `probability=False`（默认）意味着当前流水线不使用 `predict_proba` 也不画 ROC 曲线——这是 SVC 与其他分类算法分册在评估体系上的重要差异。
- SVC 的 `fit()` 是迭代优化——求解对偶二次规划问题（`libsvm` 的 SMO 算法），这与 GaussianNB（解析解）和 KNN（无训练）在计算特征上完全不同。

## 3. 训练完成后的关键属性

### 参数速览

| 属性名 | 类型 | 数学含义 | 说明 |
|---|---|---|---|
| `support_vectors_` | `ndarray`，形状 `(n_sv, n_features)` | $SV = \{\mathbf{x}_i \mid \alpha_i > 0\}$ | 所有支持向量的特征矩阵 |
| `n_support_` | `ndarray`，形状 `(n_classes,)` | — | 每个类别的支持向量数量，二分类返回 `[n_sv_class0, n_sv_class1]` |
| `support_` | `ndarray`，形状 `(n_sv,)` | — | 支持向量在训练集中对应的索引 |
| `dual_coef_` | `ndarray`，形状 `(n_classes-1, n_sv)` | $\alpha_i y_i$ | 对偶系数与标签的乘积——非支持向量的项为 0 |
| `intercept_` | `ndarray`，形状 `(n_classes*(n_classes-1)/2,)` | $b$ | 决策函数的偏置项 |
| `classes_` | `ndarray`，形状 `(n_classes,)` | — | 模型识别到的类别标签列表 |
| `shape_fit_` | `tuple` | — | 训练数据特征维度 $d$，当前为 `(2,)` |

### 示例代码

```python
print(f"支持向量总数: {model.n_support_.sum()}")
print(f"各类别支持向量数: {model.n_support_.tolist()}")
print(f"截距: {model.intercept_}")
```

### 理解重点

- `n_support_` 是 SVC 最有教学意义的属性——它直接将"支持向量决定边界"这一理论概念量化为可观察的数字。
- `dual_coef_` 和 `support_vectors_` 组合起来完整定义了决策函数 $f(\mathbf{x}) = \sum \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b$。
- 支持向量通常只占训练样本的 20%~40%——这是 SVC 稀疏性的直接体现，也是它在内存效率上优于 KNN 的原因之一。

## 4. 训练阶段的工程封装

除了 `SVC(...).fit(...)` 之外，`train_model(...)` 还做了几层工程包装：

| 输出项 | 作用 |
|---|---|
| `@print_func_info` 标题 | 帮助在终端中定位训练入口 |
| `@timeit` 训练耗时 | 观察当前模型拟合时间——SVC 的二次规划迭代比 GaussianNB 慢但比深度学习快 |
| `支持向量总数` 日志 | 快速查看模型依赖的关键样本规模 |
| `各类别支持向量数` 日志 | 观察两类样本对边界的贡献差异 |

### 理解重点

- 当前封装强调教学型可读性——通过装饰器打印函数信息和耗时，通过 `print` 输出 `n_support_`。
- 支持向量数量是最重要的日志输出——它直接反映了分类任务的难度和模型的稀疏程度。
- 这一层把"构建模型""训练模型""打印结果"收在一个函数里，方便流水线和文档复用。

## 常见坑

1. 误以为当前实现默认是线性核——源码明确使用 `kernel='rbf'`，是对同心圆数据的直接回应。
2. 只知道 `predict(...)`，却忽略 `n_support_` 和 `support_vectors_` 才是理解 SVC 行为的关键属性。
3. 把 `C` 当成"越大模型越强"的参数——$C \uparrow$ 容易过拟合，需要结合数据噪声水平调整。
4. 忘记 `probability=False` 的默认值——当前流水线不产生概率输出、不画 ROC 曲线，这是与逻辑回归等分册的评估差异。
5. 把训练函数和后续评估逻辑混在一起理解——`train_model` 只负责训练主模型，不负责混淆矩阵等诊断。

## 小结

- `train_model(...)` 是本仓库 SVC 的核心训练入口，是对 `sklearn.svm.SVC` 的薄封装。
- `SVC` 的关键参数是 `C`（软间隔容错）、`kernel`（核函数类型）和 `gamma`（RBF 核局部半径）。
- 训练完成后的核心属性：`n_support_`（支持向量数）、`support_vectors_`（支持向量特征）、`dual_coef_`（$\alpha_i y_i$）、`intercept_`（$b$）——四者共同定义决策函数。
- SVC 的 `fit()` 是真正的迭代优化（二次规划 SMO 算法），在训练效率上介于解析解模型和深度学习之间。
