---
title: PCA 主成分分析 — 模型构建
outline: deep
---

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `PCA`。
2. 理解 `PCA` 的核心构造器参数（`n_components`、`svd_solver`）及其数学对应关系。
3. 看清训练完成后最重要的模型属性——`components_`（主成分方向）、`explained_variance_ratio_`（解释方差比）、`singular_values_`（奇异值）。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `sklearn.decomposition.PCA` 模型，打印解释方差比日志 |
| `PCA(...)` | 类 | scikit-learn 提供的主成分分析器——通过 SVD 寻找方差最大的正交方向 |
| `model.fit(X_train)` | 方法 | 学习主成分方向——无监督，不接收标签 |
| `model.components_` | 属性 | 主成分方向矩阵——将原始特征空间映射到主成分空间的线性变换 |
| `model.explained_variance_ratio_` | 属性 | 各主成分的解释方差占比——反映每个方向的重要性 |
| `model.transform(X)` | 方法 | 将数据投影到主成分空间——生成降维后的坐标 |

## 1. `train_model(...)` 的函数签名

### 参数速览

适用函数：`train_model(X_train, n_components=2, svd_solver='auto', random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like`，形状 $(400, 10)$ | 标准化后的特征矩阵，传入 `PCA.fit()` | `X_scaled` |
| `n_components` | `int` | 保留的主成分数。当前 2D 模型为 `2`，3D 模型为 `3` | `2`、`3`、`5` |
| `svd_solver` | `str` | SVD 求解器。`'auto'`（默认）自适应选择最优实现：小数据用 `'full'`，大数据用 `'randomized'` | `'auto'`、`'full'`、`'randomized'`、`'arpack'` |
| `random_state` | `int` | 随机种子——`'randomized'` 求解器时需要，保证结果可复现。默认 `42` | `42` |
| 返回值 | `PCA` | 已完成 `fit()` 的模型对象，含 `components_`、`explained_variance_ratio_` 等 | — |

### 示例代码

```python
from model_training.dimensionality.pca import train_model

model = train_model(X_scaled, n_components=2)
```

### 理解重点

- 和 LDA 分册不同，`train_model(...)` **没有 `y_train` 参数**——PCA 是无监督算法，不需要标签。
- `n_components=2` 和 `n_components=3` 分别在流水线中用于训练两个独立模型——这在所有算法分册中独一无二。
- `train_model(...)` 是对 `sklearn.decomposition.PCA` 的薄封装——算法本体是 scikit-learn 基于 SVD 的高效实现。

## 2. `PCA` 构造器参数

### 参数速览

适用 API：`PCA(n_components=2, svd_solver='auto', random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_components` | `int` | 保留的主成分数。无类别约束（与 LDA 的 $K-1$ 上限不同），可从 1 到 $\min(d, N)$ 自由选择 | `2`、`3`、`5` |
| `svd_solver` | `str` | SVD 求解器。`'auto'`（默认）自适应选择；`'full'` 完整 SVD（精确但慢）；`'randomized'` 随机化 SVD（大数据快）；`'arpack'` 只求前 $q$ 个特征对 | `'auto'`、`'full'`、`'randomized'`、`'arpack'` |
| `random_state` | `int` | 随机种子——`'randomized'` 和 `'arpack'` 求解器需要。默认 `42` | `42` |
| `whiten` | `bool` | 是否白化——使各主成分方差归一化。默认 `False` | `False`、`True` |
| `tol` | `float` | `'arpack'` 求解器的收敛容忍度。默认 `0.0` | `0.0`、`1e-6` |
| `iterated_power` | `int` 或 `'auto'` | `'randomized'` 求解器的幂迭代次数。默认 `'auto'` | `'auto'`、`5` |
| `n_oversamples` | `int` | `'randomized'` 求解器的过采样数。默认 `10` | `10`、`20` |
| `power_iteration_normalizer` | `str` | `'randomized'` 求解器的幂迭代归一化方式。默认 `'auto'` | `'auto'`、`'QR'`、`'LU'`、`'none'` |

### 示例代码

```python
model = PCA(
    n_components=2,
    svd_solver="auto",
    random_state=42,
)
model.fit(X_train)
```

### 理解重点

- PCA 的核心参数是 `n_components`——它直接决定降维后的维数和保留的信息量。与 LDA 不同，PCA 没有 $K-1$ 上限。
- `svd_solver='auto'` 是大多数情况下的最佳选择——scikit-learn 会根据数据大小自动选择 full / randomized / arpack。
- PCA 的 `fit()` 是解析求解（SVD）——与 KMeans（迭代优化）和 DBSCAN（密度扩展）在计算特征上截然不同，与 LDA 类似（都是特征分解家族）。

## 3. 训练完成后的关键属性

### 参数速览

| 属性名 | 类型 | 数学含义 | 说明 |
|---|---|---|---|
| `components_` | `ndarray`，形状 `(n_components, n_features)` | 主成分方向 $\mathbf{u}_1, \dots, \mathbf{u}_q$ | 将 10 维特征映射到主成分空间的线性变换——每行是一个主成分方向 |
| `explained_variance_` | `ndarray`，形状 `(n_components,)` | $\lambda_k$ | 各主成分的方差（特征值）——反映每个方向捕获的绝对变化量 |
| `explained_variance_ratio_` | `ndarray`，形状 `(n_components,)` | $\lambda_k / \sum_j \lambda_j$ | 各主成分的解释方差占比——反映每个方向的相对重要性 |
| `singular_values_` | `ndarray`，形状 `(n_components,)` | $\sigma_k$ | 各主成分对应的奇异值——$\sigma_k^2/N = \lambda_k$ |
| `mean_` | `ndarray`，形状 `(n_features,)` | $\bar{\mathbf{x}}$ | 训练数据的均值向量——`transform()` 时用于中心化 |
| `n_features_in_` | `int` | 特征维度 $d$ | 训练时输入的特征维数，当前为 `10` |
| `n_samples_` | `int` | 样本数 $N$ | 训练时的样本数，当前为 `400` |

### 示例代码

```python
print(f"n_components: {n_components}")
print(f"解释方差比: {model.explained_variance_ratio_.round(4)}")
print(f"累计解释方差: {model.explained_variance_ratio_.sum():.4f}")
```

### 理解重点

- `components_` 是 PCA 最有教学意义的属性——它把"主成分方向"这一概念直接映射为可观察的线性变换矩阵。
- `explained_variance_ratio_` 是 PCA 的核心量化输出——它直接告诉你每个主成分有多重要。当前源码打印到 4 位小数。
- 与 LDA 的关键对比：PCA 有 `components_`（主成分方向）、`singular_values_`（奇异值），LDA 有 `scalings_`（判别方向）、`means_`（类均值）。名称不同，数学含义也不同。

## 4. `transform()` ：从模型训练到降维输出的桥梁

### 参数速览

适用方法：`model.transform(X)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `array_like`，形状 `(n, 10)` | 经过同一 `scaler` 标准化后的特征矩阵 | `X_scaled` |
| 返回值 | `ndarray`，形状 `(n, n_components)` | 投影到主成分空间后的坐标——$\mathbf{X} \cdot \text{components\_}^T$ | `X_transformed` |

### 示例代码

```python
X_transformed = model.transform(X_scaled)
```

### 理解重点

- `fit()` 学习主成分方向（`components_`），`transform()` 执行投影——两者分离的设计使模型可以对新数据重复投影。
- 流水线中 `X_transformed` 是 `plot_dimensionality(...)` 的直接输入——它是训练和可视化的桥梁。
- 与 LDA 的 `transform()` 语法相同、语义不同——PCA 投影到方差最大方向，LDA 投影到类别最可分方向。

## 5. 训练阶段的工程封装

除了 `PCA(...).fit(...)` 之外，`train_model(...)` 还做了几层工程包装：

| 输出项 | 作用 |
|---|---|
| `@print_func_info` 标题 | 帮助在终端中定位训练入口 |
| `@timeit` 训练耗时 | 观察 SVD 求解耗时——通常极快（毫秒级） |
| `n_components` 日志 | 确认当前保留的主成分数 |
| `explained_variance_ratio_` 日志 | 打印各主成分解释占比（4 位小数）和累计值 |
| `timer(...)` 上下文 | 单独测量 `fit()` 阶段的耗时 |

### 理解重点

- 当前封装强调教学型可读性——通过装饰器打印函数信息和耗时，通过 `print` 输出关键统计量。
- `explained_variance_ratio_` 是 PCA 独有的日志输出——它直接反映方差保留比例，是选择 `n_components` 的基础。
- PCA 与 LDA 在日志输出结构上相似（都有解释方差比 + 累计值），但数值含义不同——当前数据前 3 个主成分累计方差应接近但小于 100%（有噪声），LDA 的累计必然是 100%（$K-1$ 上限）。

## 常见坑

1. 误以为 `train_model(...)` 需要传入 `y_train`——PCA 是无监督算法，不接受标签。
2. 忽略 `n_components` 与数据固有秩的关系——当 `n_components` 超过固有秩时，多余的主成分只贡献噪声方差。
3. 把 `components_` 当成 LDA 的 `scalings_`——两者优化目标不同，方向含义不同。
4. 忘记 PCA 的 `explained_variance_ratio_` 总和必然 < 100%（有噪声时），而 LDA 的累计值在 `n_components=K-1` 时必为 100%。

## 小结

- `train_model(...)` 是本仓库 PCA 的核心训练入口，是对 `sklearn.decomposition.PCA` 的薄封装。
- `PCA` 的核心参数是 `n_components`（保留主成分数）和 `svd_solver`（SVD 求解器）——前者决定信息保留量，后者决定计算路径。
- 训练完成后的核心属性：`components_`（主成分方向）、`explained_variance_ratio_`（解释方差比）、`singular_values_`（奇异值）——三者分别回答了"哪个方向""多重要""多大变化"。
- PCA 有 `components_`、有 `explained_variance_ratio_`、无监督、维数自由选择——这四点构成了它与 LDA 最核心的工程差异。
