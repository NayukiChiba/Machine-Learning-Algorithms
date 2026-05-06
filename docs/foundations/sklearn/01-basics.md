---
title: sklearn 入门
outline: deep
---

# Scikit-learn 入门

## 本章目标

1. 掌握 sklearn 内置数据集的加载方式与返回结构
2. 学会使用 `make_*` 系列函数生成人工数据集
3. 理解 `train_test_split` 的分层抽样机制
4. 走通 KNN 模型的完整流程：创建 → 训练 → 预测 → 评估
5. 熟悉 sklearn 估计器的通用方法与属性命名约定

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `datasets.load_iris()` | 函数 | 加载鸢尾花数据集，返回 `Bunch` 对象 |
| `datasets.make_classification(...)` | 函数 | 生成分类人工数据集 |
| `datasets.make_regression(...)` | 函数 | 生成回归人工数据集 |
| `datasets.make_blobs(...)` | 函数 | 生成聚类人工数据集 |
| `train_test_split(...)` | 函数 | 按比例划分训练/测试集 |
| `KNeighborsClassifier(n_neighbors)` | 构造器 | K 近邻分类器 |
| `estimator.fit(X, y)` | 方法 | 训练模型 |
| `estimator.predict(X)` | 方法 | 预测标签 |
| `estimator.get_params()` / `.set_params()` | 方法 | 获取/设置超参数 |
| `clone(estimator)` | 函数 | 克隆模型（不复制训练状态） |

## 1. 加载内置数据集

### `datasets.load_iris`

#### 作用

sklearn 提供多个经典数据集，通过 `datasets.load_*()` 直接加载。返回值是 `Bunch` 对象（类字典），包含 `data`、`target`、`feature_names`、`target_names` 属性。`return_X_y=True` 直接返回 `(X, y)`，`as_frame=True` 返回 Pandas DataFrame。

#### 重点方法

```python
datasets.load_iris(*, return_X_y=False, as_frame=False)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `return_X_y` | `bool` | `True` 时直接返回 `(X, y)` 元组，默认为 `False` | `True` |
| `as_frame` | `bool` | `True` 时返回 DataFrame 格式，默认为 `False` | `True` |

内置数据集速览：

| 数据集 | 函数 | 类型 | 样本/特征 |
|---|---|---|---|
| 鸢尾花 | `load_iris()` | 分类 (3 类) | 150 / 4 |
| 乳腺癌 | `load_breast_cancer()` | 二分类 | 569 / 30 |
| 手写数字 | `load_digits()` | 分类 (10 类) | 1797 / 64 |
| 糖尿病 | `load_diabetes()` | 回归 | 442 / 10 |

#### 示例代码

```python
from sklearn import datasets

iris = datasets.load_iris()
print(f"特征矩阵形状: {iris.data.shape}")
print(f"目标向量形状: {iris.target.shape}")
print(f"特征名称: {iris.feature_names}")

# 直接返回 X, y
X, y = datasets.load_iris(return_X_y=True)
print(f"X={X.shape}, y={y.shape}")

# DataFrame 格式
iris_df = datasets.load_iris(as_frame=True)
print(f"\n{iris_df.frame.head()}")
```

#### 输出

```text
特征矩阵形状: (150, 4)
目标向量形状: (150,)
特征名称: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

X=(150, 4), y=(150,)

   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
0                5.1               3.5                1.4               0.2       0
1                4.9               3.0                1.4               0.2       0
```

#### 理解重点

- `Bunch` 对象可像字典一样访问：`iris['data']` 等价于 `iris.data`
- `data` 形状 `(n_samples, n_features)`，`target` 形状 `(n_samples,)`
- `return_X_y=True` 是最简洁的加载方式
- `as_frame=True` 在数据探索阶段非常方便——列名自动对应特征名称

## 2. 生成人工数据集

### `datasets.make_classification` / `make_regression` / `make_blobs`

#### 作用

`make_*` 系列函数用于生成可控的人工数据集，常用于算法验证和教学。`make_classification` 生成分类数据，`make_regression` 生成回归数据，`make_blobs` 生成聚类数据。

#### 重点方法

```python
datasets.make_classification(n_samples=100, n_features=20, n_informative=2, ...)
datasets.make_regression(n_samples=100, n_features=100, ...)
datasets.make_blobs(n_samples=100, n_features=2, centers=None, ...)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_samples` | `int` | 样本数量 | `1000` |
| `n_features` | `int` | 总特征数 | `20` |
| `n_informative` | `int` | 有信息量的特征数（`make_classification`） | `10` |
| `n_redundant` | `int` | 冗余特征数（`make_classification`） | `5` |
| `n_classes` | `int` | 类别数（`make_classification`） | `3` |
| `centers` | `int` | 聚类中心数（`make_blobs`） | `4` |
| `cluster_std` | `float` | 簇的标准差（`make_blobs`） | `1.0` |
| `noise` | `float` | 噪声标准差（`make_regression`） | `10` |
| `random_state` | `int` | 随机种子，保证可复现 | `42` |

#### 示例代码

```python
import numpy as np
from sklearn.datasets import (
    make_classification, make_regression, make_blobs
)

X_clf, y_clf = make_classification(
    n_samples=1000, n_features=20, n_informative=10,
    n_redundant=5, n_classes=3, random_state=42
)
print(f"分类数据: X={X_clf.shape}, 各类别数量={np.bincount(y_clf)}")

X_reg, y_reg = make_regression(
    n_samples=1000, n_features=10, noise=10, random_state=42
)
print(f"回归数据: X={X_reg.shape}, y 范围=[{y_reg.min():.1f}, {y_reg.max():.1f}]")

X_blob, y_blob = make_blobs(
    n_samples=500, centers=4, cluster_std=1.0, random_state=42
)
print(f"聚类数据: X={X_blob.shape}, y={np.unique(y_blob)}")
```

#### 输出

```text
分类数据: X=(1000, 20), 各类别数量=[334 333 333]
回归数据: X=(1000, 10), y 范围=[-609.8, 571.4]
聚类数据: X=(500, 2), y=[0 1 2 3]
```

#### 理解重点

- `n_informative + n_redundant <= n_features`——剩余特征为随机噪声
- `make_moons` 和 `make_circles` 生成线性不可分数据——适合验证非线性模型
- `random_state` 保证每次生成相同数据——实验可复现
- 人工数据的优势：已知 ground truth，便于验证模型行为

## 3. 数据划分

### `train_test_split`

#### 作用

将数据划分为训练集和测试集。`stratify=y` 进行分层抽样——确保训练集和测试集的类别比例与原数据一致。`random_state` 保证每次划分结果相同。

#### 重点方法

```python
train_test_split(*arrays, test_size=None, train_size=None, random_state=None,
                 shuffle=True, stratify=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `*arrays` | `array_like` | 待划分的数据（可多个） | `X, y` |
| `test_size` | `float` 或 `int` | 测试集比例（0~1）或样本数 | `0.3` |
| `train_size` | `float` 或 `int` | 训练集比例，默认 `1 - test_size` | `0.7` |
| `random_state` | `int` | 随机种子 | `42` |
| `shuffle` | `bool` | 划分前是否打乱，默认为 `True` | `True` |
| `stratify` | `array_like` | 按此数组类别比例分层抽样 | `y` |

#### 示例代码

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"训练集: {X_train.shape[0]} 样本, 类别分布: {np.bincount(y_train)}")
print(f"测试集: {X_test.shape[0]} 样本, 类别分布: {np.bincount(y_test)}")
```

#### 输出

```text
训练集: 105 样本, 类别分布: [35 35 35]
测试集: 45 样本, 类别分布: [15 15 15]
```

#### 理解重点

- `stratify=y` 在类别不平衡时尤为重要——避免某些类别在测试集中缺失
- 返回值顺序：`X_train, X_test, y_train, y_test`（先 X 后 y，先 train 后 test）
- 同一个 `random_state` 保证实验可复现

## 4. 第一个模型（KNN）

### `KNeighborsClassifier`

#### 作用

sklearn 所有模型遵循统一 API 流程：**创建 → fit → predict → score**。KNeighborsClassifier 通过 k 个最近邻投票决定类别——是最简单直观的分类算法。

#### 重点方法

```python
KNeighborsClassifier(n_neighbors=5, *, weights='uniform', metric='minkowski', p=2)
# 核心方法：fit(X, y) → predict(X) → score(X, y)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_neighbors` | `int` | 邻居数量 k，默认为 `5` | `3` |
| `weights` | `str` | `'uniform'` 等权投票 / `'distance'` 距离加权，默认为 `'uniform'` | `'distance'` |
| `metric` | `str` | 距离度量方式，默认为 `'minkowski'` | `'euclidean'` |
| `p` | `int` | Minkowski 距离的 p 值，`2` = 欧氏距离，默认为 `2` | `1` |

#### 示例代码

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(f"准确率: {knn.score(X_test, y_test):.4f}")
print(f"前三样本预测概率:\n{knn.predict_proba(X_test[:3])}")
```

#### 输出

```text
准确率: 0.9778
前三样本预测概率:
[[0.  1.  0. ]
 [0.  1.  0. ]
 [0.  0.8 0.2]]
```

#### 理解重点

- sklearn 统一 API 核心三步：`fit` → `predict` → `score`
- `score` 方法内部调用 `predict` 再计算指标——是一个便捷方法
- `predict_proba()` 返回每行概率之和为 1，列顺序对应 `classes_`
- KNN 的 k 值关键：k 太小过拟合，k 太大欠拟合

## 5. 估计器通用方法

### `get_params` / `set_params` / `clone`

#### 作用

sklearn 所有估计器共享一套通用方法和属性命名约定。`get_params()` 查看超参数，`set_params()` 修改超参数，`clone()` 克隆模型结构但不复制训练状态。训练后产生的属性以下划线 `_` 结尾（`classes_`、`n_features_in_` 等）。

#### 重点方法

```python
estimator.get_params(deep=True)
estimator.set_params(**params)
clone(estimator)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `deep` | `bool` | `True` 时递归获取嵌套对象的参数，默认为 `True` | `True` |
| `**params` | `dict` | 要修改的参数名=值 | `n_neighbors=3` |
| `estimator` | `estimator` | 待克隆的估计器对象 | `knn` |

训练后关键属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `classes_` | `ndarray` | 训练后类别标签 |
| `n_features_in_` | `int` | 训练时输入的特征数 |
| `feature_names_in_` | `ndarray` | 训练时输入的特征名称 |

#### 示例代码

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# get_params() — 获取全部超参数
print(f"n_neighbors: {knn.get_params()['n_neighbors']}")

# set_params() — 修改后需重新 fit
knn.set_params(n_neighbors=3, weights="distance")
knn.fit(X_train, y_train)
print(f"修改后: n_neighbors={knn.n_neighbors}, weights={knn.weights}")

# clone() — 克隆超参数，不复制训练状态
knnClone = clone(knn)
print(f"克隆模型已训练: {hasattr(knnClone, 'classes_')}")  # False

# 训练后属性（带 _ 后缀）
print(f"classes_: {knn.classes_}, n_features_in_: {knn.n_features_in_}")
```

#### 输出

```text
n_neighbors: 5
修改后: n_neighbors=3, weights=distance
克隆模型已训练: False
classes_: [0 1 2], n_features_in_: 4
```

#### 理解重点

- **超参数**（创建时传入）vs **训练后属性**（带 `_` 后缀）：这是 sklearn 的核心命名约定
- `get_params()` 返回完整参数字典——包括默认值
- `set_params()` 修改后需重新 `fit` 才生效
- `clone()` 常用于交叉验证——每折需要一个"干净"的模型

## 常见坑

1. `train_test_split` 返回值顺序是 `X_train, X_test, y_train, y_test`——不是 train 全部在前
2. 忘记 `stratify=y`——类别不平衡时不分层可能导致测试集缺少某些类别
3. `set_params()` 后未重新训练——修改参数不会自动重新 fit
4. `clone()` vs 直接赋值——`clone()` 只复制超参数，直接赋值是引用同一对象
5. `predict_proba()` 不是所有模型都有——如 `LinearSVC` 不直接支持
6. `load_boston()` 已在 sklearn 1.2+ 中移除——使用 `fetch_openml` 替代

## 小结

- sklearn 内置数据集通过 `datasets.load_*()` 加载——`return_X_y=True` 最简洁
- `make_*` 系列生成可控人工数据——适合算法验证和教学
- `train_test_split` 是数据划分标准方法——务必使用 `stratify` 保持类别比例
- sklearn 统一 API：`fit` → `predict` → `score`——所有模型通用
- 超参数用 `get_params()` / `set_params()` 管理——训练后属性以 `_` 结尾
