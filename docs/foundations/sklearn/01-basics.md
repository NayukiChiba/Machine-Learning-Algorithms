---
title: sklearn 入门
outline: deep
---

# Scikit-learn 入门

> 对应脚本：`Basic/ScikitLearn/01_basics.py`
> 运行方式：`python Basic/ScikitLearn/01_basics.py`（仓库根目录）

## 本章目标

1. 掌握 sklearn 内置数据集的加载方式与返回结构。
2. 学会使用 `make_*` 系列函数生成人工数据集。
3. 理解 `train_test_split` 的分层抽样机制。
4. 走通 KNN 模型的完整流程：创建 → 训练 → 预测 → 评估。
5. 熟悉 sklearn 估计器的通用方法与属性命名约定。

## 重点方法速览

| 方法 | 作用 | 本章位置 |
|---|---|---|
| `datasets.load_iris()` | 加载鸢尾花数据集 | `demo_load_datasets` |
| `datasets.load_iris(return_X_y=True)` | 直接返回 X, y | `demo_load_datasets` |
| `datasets.load_iris(as_frame=True)` | 返回 DataFrame | `demo_load_datasets` |
| `make_classification()` | 生成分类数据集 | `demo_generate_data` |
| `make_regression()` | 生成回归数据集 | `demo_generate_data` |
| `make_blobs()` | 生成聚类数据集 | `demo_generate_data` |
| `make_moons()` / `make_circles()` | 生成非线性可分数据 | `demo_generate_data` |
| `train_test_split(X, y, ...)` | 划分训练/测试集 | `demo_train_test_split` |
| `KNeighborsClassifier(n_neighbors)` | K 近邻分类器 | `demo_first_model` |
| `estimator.get_params()` | 获取模型参数 | `demo_estimator_methods` |
| `estimator.set_params()` | 设置模型参数 | `demo_estimator_methods` |
| `clone(estimator)` | 克隆模型（不复制训练状态） | `demo_estimator_methods` |

## 1. 加载内置数据集

### 方法重点

- sklearn 提供多个经典数据集，可通过 `datasets.load_*()` 直接加载。
- 返回值是一个 `Bunch` 对象（类似字典），包含 `data`、`target`、`feature_names`、`target_names` 等属性。
- `return_X_y=True` 直接返回 `(X, y)` 元组，省去属性访问。
- `as_frame=True` 返回 Pandas DataFrame 格式，便于 EDA。

### 参数速览（本节）

适用 API：`sklearn.datasets.load_iris(**kwargs)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `return_X_y` | `True` / `False`（默认） | `True` 直接返回 `(X, y)` 元组 |
| `as_frame` | `True` / `False`（默认） | `True` 返回 DataFrame 格式 |

常用内置数据集：

| 数据集 | 函数 | 类型 | 样本/特征 |
|---|---|---|---|
| 鸢尾花 | `load_iris()` | 分类 (3 类) | 150 / 4 |
| 乳腺癌 | `load_breast_cancer()` | 二分类 | 569 / 30 |
| 手写数字 | `load_digits()` | 分类 (10 类) | 1797 / 64 |
| 糖尿病 | `load_diabetes()` | 回归 | 442 / 10 |

### 示例代码

```python
from sklearn import datasets

# 加载鸢尾花数据集
iris = datasets.load_iris()

print(f"特征矩阵形状: {iris.data.shape}")
print(f"目标向量形状: {iris.target.shape}")
print(f"特征名称: {iris.feature_names}")
print(f"类别名称: {iris.target_names}")

# 直接返回 X, y
X, y = datasets.load_iris(return_X_y=True)
print(f"\nreturn_X_y=True: X={X.shape}, y={y.shape}")

# 返回 DataFrame 格式
iris_df = datasets.load_iris(as_frame=True)
print(f"\nas_frame=True:\n{iris_df.frame.head()}")
```

### 结果输出

```text
特征矩阵形状: (150, 4)
目标向量形状: (150,)
特征名称: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
类别名称: ['setosa' 'versicolor' 'virginica']

return_X_y=True: X=(150, 4), y=(150,)

as_frame=True:
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
0                5.1               3.5                1.4               0.2       0
1                4.9               3.0                1.4               0.2       0
2                4.7               3.2                1.3               0.2       0
3                4.6               3.1                1.5               0.2       0
4                5.0               3.6                1.4               0.2       0
```

### 理解重点

- `Bunch` 对象可像字典一样访问：`iris['data']` 等价于 `iris.data`。
- `data` 是特征矩阵 `(n_samples, n_features)`，`target` 是目标向量 `(n_samples,)`。
- `return_X_y=True` 是最常用的加载方式，代码更简洁。
- `as_frame=True` 在数据探索阶段非常方便，列名自动对应特征名称。

## 2. 生成人工数据集

### 方法重点

- `make_*` 系列函数用于生成可控的人工数据集，常用于算法验证和教学。
- `make_classification`：生成分类数据，可控制有信息量的特征数、冗余特征数。
- `make_regression`：生成回归数据，可控制噪声大小。
- `make_blobs`：生成聚类数据，指定簇的数量和标准差。
- `make_moons` / `make_circles`：生成非线性可分的二分类数据。

### 参数速览（本节）

适用 API：`sklearn.datasets.make_classification(**kwargs)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n_samples` | `1000` | 样本数量 |
| `n_features` | `20` | 总特征数 |
| `n_informative` | `10` | 有信息量的特征数 |
| `n_redundant` | `5` | 冗余特征数（由有信息特征线性组合生成） |
| `n_classes` | `3` | 类别数 |
| `random_state` | `42` | 随机种子 |

适用 API：`sklearn.datasets.make_blobs(**kwargs)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n_samples` | `500` | 样本数量 |
| `centers` | `4` | 聚类中心数 |
| `cluster_std` | `1.0` | 簇的标准差 |

### 示例代码

```python
import numpy as np
from sklearn.datasets import (
    make_classification, make_regression,
    make_blobs, make_moons, make_circles,
)

# 分类数据
X_clf, y_clf = make_classification(
    n_samples=1000, n_features=20,
    n_informative=10, n_redundant=5,
    n_classes=3, random_state=42,
)
print(f"分类数据: X={X_clf.shape}, y 各类别数量={np.bincount(y_clf)}")

# 回归数据
X_reg, y_reg = make_regression(
    n_samples=1000, n_features=10,
    noise=10, random_state=42,
)
print(f"回归数据: X={X_reg.shape}, y 范围=[{y_reg.min():.1f}, {y_reg.max():.1f}]")

# 聚类数据
X_blob, y_blob = make_blobs(
    n_samples=500, centers=4,
    cluster_std=1.0, random_state=42,
)
print(f"聚类数据: X={X_blob.shape}, y={np.unique(y_blob)}")
```

### 结果输出

```text
分类数据: X=(1000, 20), y 各类别数量=[334 333 333]
回归数据: X=(1000, 10), y 范围=[-609.8, 571.4]
聚类数据: X=(500, 2), y=[0 1 2 3]
```

### 理解重点

- `n_informative + n_redundant <= n_features`，剩余特征为随机噪声。
- `make_moons` 和 `make_circles` 生成的数据线性不可分，适合验证非线性模型（如 SVM RBF 核）。
- `random_state` 保证每次生成相同的数据，实验可复现。
- 人工数据的优势：已知"真相"（ground truth），便于验证模型行为。

## 3. 数据划分

### 方法重点

- `train_test_split` 将数据划分为训练集和测试集。
- `stratify=y` 进行分层抽样，确保训练集和测试集的类别比例与原数据一致。
- `random_state` 保证每次划分结果相同。

### 参数速览（本节）

适用 API：`sklearn.model_selection.train_test_split(*arrays, **kwargs)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `test_size` | `0.3` | 测试集比例（0~1）或样本数（整数） |
| `train_size` | `None` | 训练集比例，默认 `1 - test_size` |
| `random_state` | `42` | 随机种子 |
| `stratify` | `y` | 按此数组的类别比例进行分层抽样 |
| `shuffle` | `True`（默认） | 划分前是否打乱数据 |

### 示例代码

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

print(f"训练集: {X_train.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")
print(f"训练集类别分布: {np.bincount(y_train)}")
print(f"测试集类别分布: {np.bincount(y_test)}")
```

### 结果输出

```text
训练集: 105 样本
测试集: 45 样本
训练集类别分布: [35 35 35]
测试集类别分布: [15 15 15]
```

### 理解重点

- `stratify=y` 在类别不平衡时尤为重要，避免某些类别在测试集中缺失。
- 返回值顺序：`X_train, X_test, y_train, y_test`（先 X 后 y，先 train 后 test）。
- `test_size=0.3` 是常用默认值，实际中根据数据量调整（数据量大时可缩小测试集比例）。
- 同一个 `random_state` 保证实验可复现。

## 4. 第一个模型（KNN）

### 方法重点

- sklearn 所有模型遵循统一的 API 流程：**创建 → fit → predict → score**。
- `KNeighborsClassifier` 是最简单直观的分类算法，通过 k 个最近邻投票决定类别。
- `fit(X, y)` 训练模型，`predict(X)` 预测标签，`score(X, y)` 计算准确率。

### 参数速览（本节）

适用 API：`sklearn.neighbors.KNeighborsClassifier(**kwargs)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n_neighbors` | `5` | 邻居数量 k |
| `weights` | `'uniform'`（默认） | `'uniform'` 等权投票，`'distance'` 距离加权 |
| `metric` | `'minkowski'`（默认） | 距离度量方式 |
| `p` | `2`（默认） | Minkowski 距离的 p 值，`2` 为欧氏距离 |

### 示例代码

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets

# 加载和划分数据
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 创建模型
knn = KNeighborsClassifier(n_neighbors=5)

# 训练模型
knn.fit(X_train, y_train)

# 预测
y_pred = knn.predict(X_test)

# 评估
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"score方法: {knn.score(X_test, y_test):.4f}")
```

### 结果输出

```text
准确率: 0.9778
score方法: 0.9778
```

### 理解重点

- sklearn 统一 API 的核心三步：`fit` → `predict` → `score`。
- `accuracy_score(y_test, y_pred)` 和 `knn.score(X_test, y_test)` 结果完全一致。
- `score` 方法内部调用 `predict` 再计算指标，是一个便捷方法。
- KNN 的 k 值是关键超参数：k 太小容易过拟合，k 太大容易欠拟合。

## 5. 估计器通用方法

### 方法重点

- sklearn 所有估计器（estimator）共享一套通用方法和属性命名约定。
- `get_params()` / `set_params()` 用于查看和修改超参数。
- `predict_proba()` 返回每个类别的预测概率（部分模型支持）。
- `clone()` 克隆模型结构和超参数，但不复制训练状态。
- 训练后产生的属性以下划线 `_` 结尾（如 `classes_`、`n_features_in_`）。

### 参数速览（本节）

1. `get_params()`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 返回值 | `dict` | 获取所有超参数 |

2. `set_params(**params)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n_neighbors` | `3` | 修改近邻数 |
| `weights` | `'distance'` | 修改加权方式 |
| 返回值 | `estimator` | 返回修改后的模型对象 |

3. `predict_proba(X)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X` | `X_test[:3]` | 预测输入样本 |
| 返回值 | `(n_samples, n_classes)` | 预测类别概率矩阵 |

4. `clone(estimator)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `estimator` | `knn` | 待克隆模型 |
| 返回值 | 新估计器对象 | 仅复制超参数，不复制训练状态 |

5. `classes_`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 返回值 | `[0, 1, 2]` | 训练后类别标签 |

6. `n_features_in_`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 返回值 | `4` | 训练后输入特征数 |

### 示例代码

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

# get_params() - 获取模型参数
params = knn.get_params()
for key, value in params.items():
    print(f"  {key}: {value}")

# set_params() - 设置模型参数
knn.set_params(n_neighbors=3, weights="distance")
print(f"\nset_params() 后: n_neighbors={knn.n_neighbors}, weights={knn.weights}")

# predict_proba() - 预测概率
knn.fit(X_train, y_train)
proba = knn.predict_proba(X_test[:3])
print(f"\npredict_proba() 前3个样本:\n{proba.round(3)}")

# clone() - 克隆模型
knn_clone = clone(knn)
print(f"\nclone():")
print(f"  原模型已训练: {hasattr(knn, 'classes_')}")
print(f"  克隆模型已训练: {hasattr(knn_clone, 'classes_')}")

# 训练后属性
print(f"\n训练后属性:")
print(f"  classes_: {knn.classes_}")
print(f"  n_features_in_: {knn.n_features_in_}")
```

### 结果输出

```text
  algorithm: auto
  leaf_size: 30
  metric: minkowski
  metric_params: None
  n_jobs: None
  n_neighbors: 5
  p: 2
  weights: uniform

set_params() 后: n_neighbors=3, weights=distance

predict_proba() 前3个样本:
[[0.    0.378 0.622]
 [0.    0.76  0.24 ]
 [1.    0.    0.   ]]

clone():
  原模型已训练: True
  克隆模型已训练: False

训练后属性:
  classes_: [0 1 2]
  n_features_in_: 4
```

### 理解重点

- **超参数**（创建时传入）vs **训练后属性**（带 `_` 后缀）：这是 sklearn 的命名约定。
- `get_params()` 返回完整参数字典，包括默认值。
- `set_params()` 修改后需要重新 `fit` 才能生效。
- `clone()` 常用于交叉验证场景：每折需要一个"干净"的模型。
- `predict_proba()` 的每行概率之和为 1，列顺序对应 `classes_`。

## 常见坑

| 坑 | 说明 |
|---|---|
| `train_test_split` 返回值顺序 | 是 `X_train, X_test, y_train, y_test`，不是 train 全部在前 |
| 忘记 `stratify=y` | 在类别不平衡数据上，不分层可能导致测试集缺少某些类别 |
| `set_params()` 后未重新训练 | 修改参数不会自动重新 fit，必须手动调用 `fit` |
| `clone()` vs 直接赋值 | `clone()` 只复制超参数，直接赋值是引用同一对象 |
| `predict_proba()` 不是所有模型都有 | 如 `LinearSVC` 不直接支持，需要用 `CalibratedClassifierCV` 包装 |
| `load_boston()` 已弃用 | sklearn 1.2+ 中已移除，使用 `fetch_openml` 替代 |

## 小结

- sklearn 内置数据集通过 `datasets.load_*()` 加载，`return_X_y=True` 是最简洁的方式。
- `make_*` 系列函数生成可控的人工数据，适合算法验证和教学。
- `train_test_split` 是数据划分的标准方法，务必使用 `stratify` 保持类别比例。
- sklearn 统一 API：`fit` → `predict` → `score`，所有模型通用。
- 估计器超参数用 `get_params()` / `set_params()` 管理，训练后属性以 `_` 结尾。
