---
title: 正则化回归 — 数据构成
outline: deep
---

# 数据构成

> 对应代码：`data_generation/regression.py`、`data_generation/__init__.py`、`pipelines/regression/regularization.py`
>  
> 相关对象：`RegressionData.regularization()`、`regularization_data`

## 本章目标

1. 明确本仓库正则化回归数据来自 `RegressionData.regularization()` 的构造逻辑。
2. 明确原始医学特征、人工相关特征、纯噪声特征分别是什么。
3. 明确训练集/测试集切分与标准化的顺序和边界。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `RegressionData.regularization()` | 方法 | 生成正则化回归使用的数据表 |
| `load_diabetes(as_frame=True)` | 函数 | scikit-learn 提供的 diabetes 回归数据集加载器 |
| `regularization_data` | 变量 | 在 `data_generation/__init__.py` 中导出的数据对象 |
| `bmi_corr` / `bp_corr` / `s5_corr` | 列名 | 为了制造多重共线性而追加的相关特征 |
| `noise_1` ~ `noise_8` | 列名 | 为了观察稀疏化效果而追加的纯噪声特征 |
| `price` | 列名 | 当前流水线中的回归目标列 |

## 1. 本仓库数据入口

- 数据变量：`data_generation/__init__.py` 中导出的 `regularization_data`
- 生成来源：`data_generation/regression.py` 中的 `RegressionData.regularization()`
- 流水线使用：`pipelines/regression/regularization.py` 中的 `data = regularization_data.copy()`

### 理解重点

- `regularization_data` 在导入时就已经生成完成，因此流水线里直接 `.copy()` 使用即可。
- 使用 `.copy()` 的目的，是避免后续切分或特征处理误改原始数据对象。

## 2. 数据生成函数 `RegressionData.regularization()`

### 参数速览（本节）

适用 API（分项）：

1. `RegressionData.regularization()`
2. `load_diabetes(as_frame=True)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `as_frame` | `True` | 直接返回带列名的 `DataFrame` |
| `random_state` | `42` | 用于生成相关特征噪声和纯噪声特征 |
| `reg_add_corr_features` | `True` | 是否添加 3 个高度相关特征 |
| `reg_add_noise_features` | `8` | 额外追加的纯噪声特征数量 |
| 返回值 | `DataFrame` | 含原始特征、人工特征与 `price` 的数据表 |

### 示例代码

```python
rng = np.random.RandomState(self.random_state)

data = load_diabetes(as_frame=True)
df = data.frame.copy().rename(columns={"target": "price"})

if self.reg_add_corr_features:
    df["bmi_corr"] = df["bmi"] * 0.9 + rng.normal(scale=0.02, size=len(df))
    df["bp_corr"] = df["bp"] * 0.9 + rng.normal(scale=0.02, size=len(df))
    df["s5_corr"] = df["s5"] * 0.9 + rng.normal(scale=0.02, size=len(df))

for i in range(self.reg_add_noise_features):
    df[f"noise_{i + 1}"] = rng.normal(size=len(df))
```

### 理解重点

- 基础数据不是手工合成，而是 scikit-learn 自带的 diabetes 真实回归数据集。
- 文档里关注的重点不是 diabetes 本身，而是源码在其基础上额外构造的共线性和噪声特征。
- 这些人工特征正是当前分册能够展示 Ridge、Lasso、ElasticNet 差异的关键。

## 3. 特征列的三层结构

当前数据表不是单一来源，而是三部分特征拼接而成。

### 参数速览（本节）

适用列组（本节）：

1. diabetes 原始特征
2. 共线性特征
3. 纯噪声特征

| 列组 | 当前内容 | 作用 |
|---|---|---|
| 原始医学特征 | `age`、`sex`、`bmi`、`bp`、`s1` ~ `s6` | 提供真实回归信号 |
| 相关特征 | `bmi_corr`、`bp_corr`、`s5_corr` | 人为制造多重共线性 |
| 噪声特征 | `noise_1` ~ `noise_8` | 用于观察稀疏化与抗过拟合能力 |
| 标签列 | `price` | 回归目标值 |

### 示例代码

```python
X = data.drop(columns=["price"])
y = data["price"]
feature_names = list(X.columns)
```

### 理解重点

- `bmi_corr`、`bp_corr`、`s5_corr` 与对应原始列高度相关，适合观察共线性下系数如何分配。
- `noise_1` ~ `noise_8` 理论上不该提供稳定预测信息，因此很适合观察 L1 稀疏化是否把它们压到接近 0。
- 当前流水线统一使用 `price` 作为标签列名，因此训练代码只按表字段拆分，不需要关心原始数据集里的 `target` 名称。

## 4. 切分与标准化

### 参数速览（本节）

适用 API（分项）：

1. `train_test_split(X, y, test_size=0.2, random_state=42)`
2. `StandardScaler().fit_transform(X_train)`
3. `StandardScaler().transform(X_test)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `test_size` | `0.2` | 测试集占比 |
| `random_state` | `42` | 保证可复现划分 |
| `X_train` | 训练特征 | 只在训练集上拟合标准化统计量 |
| `X_test` | 测试特征 | 使用训练集统计量做变换 |
| 返回值 | `X_train_s`、`X_test_s` | 标准化后的训练和测试特征 |

### 示例代码

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

### 理解重点

- 正则化回归非常依赖特征量纲一致性，因此标准化在这里不是可有可无，而是默认流程的一部分。
- 标准化必须发生在切分之后，否则会把测试集信息泄露到训练过程里。
- 当前文档中的 `X_train_s`、`X_test_s` 都是源码里真实使用的变量名。

## 常见坑

1. 忘记把 `target` 已被重命名为 `price`，导致拆分标签列时写错字段名。
2. 只看到 diabetes 原始特征，忽略了源码额外追加的 `*_corr` 与 `noise_*` 特征。
3. 在切分之前就对全量数据做标准化，造成数据泄露。

## 小结

- 当前正则化回归数据来自 `RegressionData.regularization()`，底层使用的是 `load_diabetes(as_frame=True)`。
- 数据表由原始医学特征、3 个相关特征、8 个纯噪声特征和标签列 `price` 组成。
- 读懂数据构成，才能真正理解后续章节里三种正则化模型为什么会表现出不同的系数形态。
