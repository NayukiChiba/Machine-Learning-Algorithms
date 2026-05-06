---
title: sklearn 预处理
outline: deep
---

# sklearn 预处理

## 本章目标

1. 理解四种常见缩放器在异常值场景下的行为差异
2. 掌握 `StandardScaler` 与 `MinMaxScaler` 的核心参数与逆变换
3. 理解 `PowerTransformer` 处理偏态分布的作用与限制
4. 学会区分 `LabelEncoder` 与 `OneHotEncoder` 的适用场景
5. 掌握缺失值填充与 `ColumnTransformer` 组合预处理流程

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `StandardScaler()` | 构造器 | 标准化到均值 0、方差 1 |
| `MinMaxScaler(feature_range)` | 构造器 | 线性缩放到指定区间 |
| `RobustScaler()` | 构造器 | 使用中位数与 IQR，抗异常值 |
| `PowerTransformer(method)` | 构造器 | 幂变换改善偏态分布 |
| `LabelEncoder()` | 构造器 | 一维标签编码 |
| `OneHotEncoder(...)` | 构造器 | 类别特征独热编码 |
| `SimpleImputer(strategy)` | 构造器 | 统计量或常量填充缺失值 |
| `KNNImputer(n_neighbors)` | 构造器 | 基于近邻估计缺失值 |
| `ColumnTransformer(transformers)` | 构造器 | 按列类型组合预处理流水线 |

## 1. 缩放器对比

### StandardScaler / MinMaxScaler / RobustScaler / MaxAbsScaler

#### 作用

同一组数据在不同缩放器下的分布明显不同——尤其存在异常值时。`StandardScaler` 与 `MinMaxScaler` 对异常值更敏感，`RobustScaler` 使用中位数和 IQR 更稳健，`MaxAbsScaler` 不平移数据中心适合保持稀疏结构。

#### 重点方法

```python
StandardScaler(*, copy=True, with_mean=True, with_std=True)
MinMaxScaler(feature_range=(0, 1), *, copy=True, clip=False)
RobustScaler(*, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
MaxAbsScaler(*, copy=True)
# 核心方法：fit(X) → transform(X) / fit_transform(X) → inverse_transform(X)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `copy` | `bool` | 是否复制输入数据，默认为 `True` | `True` |
| `with_mean` | `bool` | 对特征做中心化（StandardScaler），默认为 `True` | `True` |
| `with_std` | `bool` | 按标准差缩放（StandardScaler），默认为 `True` | `True` |
| `feature_range` | `tuple[float, float]` | 目标缩放区间（MinMaxScaler），默认为 `(0, 1)` | `(-1, 1)` |
| `clip` | `bool` | 推理阶段是否截断超范围值（MinMaxScaler），默认为 `False` | `True` |
| `with_centering` | `bool` | 使用中位数做中心化（RobustScaler），默认为 `True` | `True` |
| `with_scaling` | `bool` | 按分位间距缩放（RobustScaler），默认为 `True` | `True` |
| `quantile_range` | `tuple[float, float]` | IQR 分位区间（RobustScaler），默认为 `(25.0, 75.0)` | `(10.0, 90.0)` |

#### 示例代码

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

np.random.seed(42)
X = np.random.randn(100, 2) * 10 + 50
X[0] = [200, 200]  # 注入异常值

scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler(),
    "MaxAbsScaler": MaxAbsScaler(),
}

for name, scaler in scalers.items():
    Xs = scaler.fit_transform(X)
    print(f"{name}: 范围=[{Xs.min():.2f}, {Xs.max():.2f}]")
```

#### 输出

```text
StandardScaler: 范围=[-1.76, 9.59]
MinMaxScaler: 范围=[0.00, 1.00]
RobustScaler: 范围=[-2.57, 11.58]
MaxAbsScaler: 范围=[0.17, 1.00]
```

#### 理解重点

- 若后续模型依赖距离（KNN、SVM），缩放是必要步骤
- 存在强异常值时优先比较 `RobustScaler` 与其他方案
- 缩放器选择本质是分布假设选择——不是固定套路

## 2. StandardScaler 详解

### `StandardScaler`

#### 作用

学习训练集的 `mean_` 与 `scale_`，对每列做 $z = (x - \mu) / \sigma$。`fit_transform` 用于训练阶段，测试集应使用 `transform`。`inverse_transform` 可将标准化结果还原到原始尺度。

#### 重点方法

```python
StandardScaler(*, copy=True, with_mean=True, with_std=True)
# fit(X) → transform(X) / fit_transform(X) → inverse_transform(X)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `copy` | `bool` | 是否复制输入数据，默认为 `True` | `True` |
| `with_mean` | `bool` | 对每列减去均值，默认为 `True` | `True` |
| `with_std` | `bool` | 对每列除以标准差，默认为 `True` | `True` |

训练后属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `mean_` | `ndarray` | 每列均值 |
| `scale_` | `ndarray` | 每列标准差 |
| `var_` | `ndarray` | 每列方差 |

#### 示例代码

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

X = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])

scaler = StandardScaler()
XScaled = scaler.fit_transform(X)
XBack = scaler.inverse_transform(XScaled)

print(f"mean_: {scaler.mean_}")
print(f"scale_: {scaler.scale_}")
print(f"逆变换还原:\n{XBack}")
```

#### 输出

```text
mean_: [ 3. 30.]
scale_: [ 1.4142 14.1421]
逆变换还原:
[[ 1. 10.]
 [ 2. 20.]
 [ 3. 30.]
 [ 4. 40.]
 [ 5. 50.]]
```

#### 理解重点

- 标准化参数必须来自训练集——避免数据泄露
- `mean_` 和 `scale_` 也是可解释信息——可用于排查异常列
- 稀疏矩阵通常谨慎使用中心化——会破坏稀疏结构

## 3. MinMaxScaler 详解

### `MinMaxScaler`

#### 作用

将每列线性映射到指定区间 $[min, max]$，保持原始排序关系。常见区间为 `(0, 1)` 或 `(-1, 1)`。对异常值敏感——极端值会压缩其余样本的有效分辨率。

#### 重点方法

```python
MinMaxScaler(feature_range=(0, 1), *, copy=True, clip=False)
# fit(X) → transform(X) / fit_transform(X) → inverse_transform(X)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `feature_range` | `tuple[float, float]` | 目标区间，默认为 `(0, 1)` | `(-1, 1)` |
| `copy` | `bool` | 是否复制输入数组，默认为 `True` | `True` |
| `clip` | `bool` | 推理阶段是否截断超范围值，默认为 `False` | `True` |

#### 示例代码

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

X = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])

s1 = MinMaxScaler(feature_range=(0, 1))
s2 = MinMaxScaler(feature_range=(-1, 1))

print(f"feature_range=(0,1):\n{s1.fit_transform(X)}")
print(f"\nfeature_range=(-1,1):\n{s2.fit_transform(X)}")
```

#### 输出

```text
feature_range=(0,1):
[[0.   0.  ]
 [0.25 0.25]
 [0.5  0.5 ]
 [0.75 0.75]
 [1.   1.  ]]

feature_range=(-1,1):
[[-1.  -1. ]
 [-0.5 -0.5]
 [ 0.   0. ]
 [ 0.5  0.5]
 [ 1.   1. ]]
```

#### 理解重点

- 区间缩放不会让分布接近正态——仅改变取值区间
- 对树模型通常不是必须——但对基于距离或梯度的模型常常有帮助
- 当特征天然有上下界时 MinMax 缩放更直观

## 4. PowerTransformer 幂变换

### `PowerTransformer`

#### 作用

缓解偏态分布，使数据更接近对称分布。`yeo-johnson` 可处理非正数，`box-cox` 仅支持严格正数。变换后通常更利于线性模型和基于方差假设的方法。

#### 重点方法

```python
PowerTransformer(method='yeo-johnson', *, standardize=True)
# fit(X) → transform(X) / fit_transform(X)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `method` | `str` | `'yeo-johnson'` 支持非正数 / `'box-cox'` 仅正数，默认为 `'yeo-johnson'` | `'box-cox'` |
| `standardize` | `bool` | 变换后是否再做标准化，默认为 `True` | `True` |

训练后属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `lambdas_` | `ndarray` | 每列的变换参数，反映分布拉伸程度 |

#### 示例代码

```python
import numpy as np
from sklearn.preprocessing import PowerTransformer

np.random.seed(42)
XSkewed = np.random.exponential(scale=2, size=(500, 1))

ptYj = PowerTransformer(method="yeo-johnson")
ptBc = PowerTransformer(method="box-cox")

Xyj = ptYj.fit_transform(XSkewed)
Xbc = ptBc.fit_transform(XSkewed)

print(f"原始: 均值={XSkewed.mean():.2f}, 偏度={np.mean((XSkewed - XSkewed.mean())**3) / XSkewed.std()**3:.2f}")
print(f"Yeo-Johnson lambda: {ptYj.lambdas_[0]:.3f}")
print(f"Box-Cox lambda: {ptBc.lambdas_[0]:.3f}")
```

#### 输出

```text
原始: 均值=2.01, 偏度=1.92
Yeo-Johnson lambda: -0.412
Box-Cox lambda: 0.264
```

#### 理解重点

- 偏态修正不等于信息增强——目标是改善建模假设匹配度
- 数据含 0 或负值优先使用 Yeo-Johnson
- 配合可视化直方图判断变换效果

## 5. 类别编码

### `LabelEncoder` / `OneHotEncoder`

#### 作用

`LabelEncoder` 适合目标标签编码——将类别映射为整数。`OneHotEncoder` 将类别映射为哑变量——更适合线性模型与距离模型。处理未知类别时 `handle_unknown='ignore'` 更稳妥。

#### 重点方法

```python
LabelEncoder()
OneHotEncoder(*, categories='auto', drop=None, sparse_output=True,
              handle_unknown='error')
# fit(y) → transform(y) / fit_transform(y)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `sparse_output` | `bool` | OneHotEncoder：`False` 返回稠密数组，默认为 `True` | `False` |
| `handle_unknown` | `str` | OneHotEncoder：`'error'` 报错 / `'ignore'` 忽略，默认为 `'error'` | `'ignore'` |
| `drop` | `str` 或 `None` | OneHotEncoder：`'first'` 丢弃第一类避免共线性，默认为 `None` | `'first'` |

编码后属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `classes_` | `ndarray` | 编码前后的类别映射 |
| `categories_` | `list[ndarray]` | OneHotEncoder：每列的类别列表 |

#### 示例代码

```python
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

colors = np.array([["红"], ["绿"], ["蓝"], ["红"], ["绿"]])

le = LabelEncoder()
colorsLe = le.fit_transform(colors.ravel())

ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
colorsOhe = ohe.fit_transform(colors)

print(f"LabelEncoder 类别: {le.classes_}")
print(f"编码结果: {colorsLe}")
print(f"OneHotEncoder 特征名: {ohe.get_feature_names_out()}")
print(f"编码结果:\n{colorsOhe}")
```

#### 输出

```text
LabelEncoder 类别: ['蓝' '绿' '红']
编码结果: [2 1 0 2 1]
OneHotEncoder 特征名: ['x0_红' 'x0_绿' 'x0_蓝']
编码结果:
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]]
```

#### 理解重点

- `LabelEncoder` 产生整数顺序——可能引入虚假大小关系，仅适合目标变量
- `OneHotEncoder` 增加维度——需关注稀疏性与内存开销
- 生产环境必须提前设计未知类别处理策略（`handle_unknown='ignore'`）
- `get_feature_names_out()` 返回编码后列名——便于调试和特征分析

## 6. 缺失值处理

### `SimpleImputer` / `KNNImputer`

#### 作用

`SimpleImputer` 提供均值、中位数、众数、常量等规则化填充。`KNNImputer` 利用样本相似性推断缺失值——通常更平滑但更耗时。填充策略应与特征分布和业务语义一致。

#### 重点方法

```python
SimpleImputer(*, missing_values=nan, strategy='mean', fill_value=None)
KNNImputer(*, n_neighbors=5, weights='uniform')
# fit(X) → transform(X) / fit_transform(X)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `strategy` | `str` | SimpleImputer 填充策略：`'mean'` / `'median'` / `'most_frequent'` / `'constant'`，默认为 `'mean'` | `'median'` |
| `fill_value` | `str` 或 `float` | `strategy='constant'` 时的填充值，默认为 `None`（数值列填 0，字符串列填 `'missing_value'`） | `0` |
| `missing_values` | `int`、`float`、`str` | 缺失值标记符，默认为 `np.nan` | `np.nan` |
| `n_neighbors` | `int` | KNNImputer 参考邻居数量，默认为 `5` | `3` |
| `weights` | `str` | KNNImputer 权重策略：`'uniform'` / `'distance'`，默认为 `'uniform'` | `'distance'` |

#### 示例代码

```python
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

X = np.array([[1, 2, np.nan], [3, np.nan, 6], [7, 8, 9], [np.nan, 5, 3]])

print("mean 填充:\n", SimpleImputer(strategy="mean").fit_transform(X))
print("\nmedian 填充:\n", SimpleImputer(strategy="median").fit_transform(X))
print("\nconstant=0 填充:\n", SimpleImputer(strategy="constant", fill_value=0).fit_transform(X))
print("\nKNN 填充:\n", KNNImputer(n_neighbors=2).fit_transform(X))
```

#### 输出

```text
mean 填充:
[[1.   2.   6.  ]
 [3.   5.   6.  ]
 [7.   8.   9.  ]
 [3.67 5.   3.  ]]

median 填充:
[[1. 2. 6.]
 [3. 5. 6.]
 [7. 8. 9.]
 [3. 5. 3.]]

constant=0 填充:
[[1. 2. 0.]
 [3. 0. 6.]
 [7. 8. 9.]
 [0. 5. 3.]]

KNN 填充:
[[1.  2.  6. ]
 [3.  5.  6. ]
 [7.  8.  9. ]
 [4.  5.  3. ]]
```

#### 理解重点

- 均值/中位数填充简单稳定——多数任务的首选基线
- KNN 填充更依赖特征尺度——通常应先做合理缩放
- 缺失值机制（MCAR/MAR/MNAR）影响填充有效性——需结合业务判断

## 7. ColumnTransformer 组合预处理

### `ColumnTransformer`

#### 作用

混合类型数据应采用分列处理：数值列与类别列使用不同流水线。`ColumnTransformer` 将多条子流水线拼接为统一特征空间——这是生产级预处理的核心模式，后续可直接接入模型。

#### 重点方法

```python
ColumnTransformer(transformers, *, remainder='drop', n_jobs=None,
                 verbose_feature_names_out=True)
# fit(X) → transform(X) / fit_transform(X) → get_feature_names_out()
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `transformers` | `list[tuple[str, estimator, columns]]` | 列分组与对应处理器列表 | `[("num", numPipe, numCols), ("cat", catPipe, catCols)]` |
| `remainder` | `str` 或 `estimator` | 未指定列的处理方式：`'drop'` 丢弃 / `'passthrough'` 保留，默认为 `'drop'` | `'passthrough'` |
| `verbose_feature_names_out` | `bool` | 特征名是否加前缀，默认为 `True` | `False` |

快捷列选择器：

```python
from sklearn.compose import make_column_selector
make_column_selector(dtype_include='number')  # 按类型自动选择列
```

#### 示例代码

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.DataFrame({
    "年龄": [25, 30, np.nan, 40, 35],
    "收入": [50000, 60000, 55000, np.nan, 70000],
    "性别": ["男", "女", "男", "女", "男"],
    "城市": ["北京", "上海", "北京", "广州", "上海"],
})

numPipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

catPipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numPipe, selector(dtype_include="number")),
    ("cat", catPipe, selector(dtype_include="object")),
])

XProc = preprocessor.fit_transform(df)
print(f"处理后形状: {XProc.shape}")
print(f"特征名称: {preprocessor.get_feature_names_out()}")
```

#### 输出

```text
处理后形状: (5, 7)
特征名称: ['num__年龄' 'num__收入' 'cat__性别_女' 'cat__性别_男' 'cat__城市_上海' 'cat__城市_北京' 'cat__城市_广州']
```

#### 理解重点

- 列级流水线能把预处理逻辑完全纳入模型训练过程——减少线上线下不一致
- 该模式可直接嵌入 `Pipeline` 做联合调参与部署
- `make_column_selector` 按 dtype 自动选列——避免手动列名硬编码
- 当类别空间较大时应关注 One-Hot 维度膨胀问题

## 常见坑

1. 在训练前先对全量数据 `fit` 缩放器或填充器——导致数据泄露
2. 直接把 `LabelEncoder` 用在普通类别特征上——引入错误顺序关系
3. 在 `ColumnTransformer` 中忘记统一缺失值策略——导致推理阶段报错
4. 测试集使用 `fit_transform` 而非 `transform`——破坏了训练/测试隔离

## 小结

- 预处理不是独立步骤——而是模型流程的一部分
- 推荐将缩放、编码、填充封装进流水线并与模型共同训练
- 先建立稳定可复现的预处理基线——再做策略替换和调优
- `ColumnTransformer` + `Pipeline` 是生产级预处理的标准模式
