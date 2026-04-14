---
title: Bagging 与随机森林 — 数据构成
outline: deep
---

# 数据构成

> 对应代码：`data_generation/ensemble.py`、`data_generation/__init__.py`、`pipelines/ensemble/bagging.py`
>  
> 相关对象：`EnsembleData.bagging()`、`bagging_data`

## 本章目标

1. 明确本仓库 Bagging 数据来自 `EnsembleData.bagging()` 的双月牙构造逻辑。
2. 明确特征列、标签列与高噪声设计在当前任务中的作用。
3. 明确分层切分与标准化的顺序和边界。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `EnsembleData.bagging()` | 方法 | 生成 Bagging 分类使用的双月牙数据 |
| `make_moons(...)` | 函数 | scikit-learn 提供的双月牙数据生成器 |
| `bagging_data` | 变量 | 在 `data_generation/__init__.py` 中导出的数据对象 |
| `label` | 列名 | 当前流水线中的分类目标列 |

## 1. 本仓库数据入口

- 数据变量：`data_generation/__init__.py` 中导出的 `bagging_data`
- 生成来源：`data_generation/ensemble.py` 中的 `EnsembleData.bagging()`
- 流水线使用：`pipelines/ensemble/bagging.py` 中的 `data = bagging_data.copy()`

### 理解重点

- `bagging_data` 在导入时就已经生成完成，因此流水线里直接 `.copy()` 使用即可。
- 使用 `.copy()` 的目的，是避免后续切分或调试过程意外修改原始数据对象。

## 2. 数据生成函数 `EnsembleData.bagging()`

### 参数速览（本节）

适用 API（分项）：

1. `EnsembleData.bagging()`
2. `make_moons(...)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n_samples` | `500` | 样本总数 |
| `noise` | `0.35` | 双月牙噪声强度 |
| `random_state` | `42` | 随机种子，保证可复现 |
| 返回值 | `DataFrame` | 含 `x1`、`x2` 与 `label` 的数据表 |

### 示例代码

```python
X, y = make_moons(
    n_samples=self.n_samples,
    noise=self.bagging_noise,
    random_state=self.random_state,
)
return DataFrame({"x1": X[:, 0], "x2": X[:, 1], "label": y})
```

### 理解重点

- 当前数据不是线性可分的规则点集，而是带较高噪声的双月牙二分类数据。
- 高噪声会让单棵深树更容易过拟合，因此更能体现 Bagging 的降方差价值。
- 这也是源码注释里特别强调 `noise=0.35` 是有意设计的原因。

## 3. 特征列与标签列

当前数据表结构如下：

- 特征列：`x1`、`x2`
- 标签列：`label`

### 参数速览（本节）

适用列组（本节）：

1. 二维特征列
2. 二分类标签列

| 列名 | 当前作用 |
|---|---|
| `x1`、`x2` | 描述双月牙二维几何结构 |
| `label` | 0/1 二分类目标 |

### 示例代码

```python
X = data.drop(columns=["label"])
y = data["label"]
```

### 理解重点

- 当前 Bagging 任务非常适合做分类边界可视化理解，虽然当前流水线没有直接画决策边界图。
- 二维特征结构能帮助你在脑中建立“噪声双月牙 + 高方差树模型”的直觉。
- 这和 GBDT、LightGBM 那类高维分类任务的阅读重点不同。

## 4. 分层切分与标准化

### 参数速览（本节）

适用 API（分项）：

1. `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`
2. `StandardScaler().fit_transform(X_train)`
3. `StandardScaler().transform(X_test)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `test_size` | `0.2` | 测试集占比 |
| `random_state` | `42` | 保证可复现划分 |
| `stratify` | `y` | 保持训练集和测试集类别比例一致 |
| `X_train_s` | 标准化训练特征 | 供 Bagging 训练使用 |
| `X_test_s` | 标准化测试特征 | 供 Bagging 预测使用 |

### 示例代码

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

### 理解重点

- 当前 Bagging 流水线真实包含标准化步骤，文档必须如实写清楚。
- `stratify=y` 对二分类任务依然重要，它能让训练集和测试集的类别比例更稳定。
- 当前文档中的 `X_train_s`、`X_test_s` 都是源码里真实使用的变量名。

## 常见坑

1. 把当前 Bagging 数据误写成低噪声或线性可分数据，忽略它其实是高噪声双月牙。
2. 忽略 `stratify=y`，把当前二分类切分流程写成普通随机切分。
3. 把“当前是二维边界任务”误扩展成“当前流水线已经实现了边界可视化”。

## 小结

- 当前 Bagging 数据来自 `EnsembleData.bagging()`，底层使用的是 `make_moons(...)`。
- 数据表由二维特征 `x1`、`x2` 和标签列 `label` 组成，并故意加入较高噪声。
- 读懂数据结构、噪声设计、分层切分和标准化顺序，是理解后续 Bagging 训练与分类评估的前提。
