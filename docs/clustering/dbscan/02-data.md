---
title: DBSCAN 密度聚类 — 数据构成
outline: deep
---

# 数据构成

> 对应代码：`data_generation/clustering.py`、`data_generation/__init__.py`、`pipelines/clustering/dbscan.py`
>
> 相关对象：`ClusteringData.dbscan()`、`dbscan_data`

## 本章目标

1. 明确本仓库 DBSCAN 数据来自 `ClusteringData.dbscan()` 的双月牙生成逻辑。
2. 明确特征列与 `true_label` 在当前流水线中的角色差异。
3. 明确标准化发生在什么位置，以及它为什么对 DBSCAN 很重要。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `ClusteringData.dbscan()` | 方法 | 生成 DBSCAN 使用的二维聚类数据 |
| `make_moons(...)` | 函数 | scikit-learn 提供的双月牙数据生成器 |
| `dbscan_data` | 变量 | 在 `data_generation/__init__.py` 中导出的数据对象 |
| `true_label` | 列名 | 真实簇标签，仅用于结果对照，不参与训练 |
| `StandardScaler` | 类 | 对特征做标准化，避免距离尺度影响邻域判定 |

## 1. 本仓库数据入口

- 数据变量：`data_generation/__init__.py` 中导出的 `dbscan_data`
- 生成来源：`data_generation/clustering.py` 中的 `ClusteringData.dbscan()`
- 流水线使用：`pipelines/clustering/dbscan.py` 中的 `data = dbscan_data.copy()`

### 理解重点

- `dbscan_data` 在导入时就已经生成完成，因此流水线里直接 `.copy()` 使用即可。
- 用 `.copy()` 的目的，是避免后续处理意外修改原始数据对象。
- 当前数据是为 DBSCAN 教学场景专门构造的，因此能突出密度聚类的优势。

## 2. 数据生成函数 `ClusteringData.dbscan()`

### 参数速览（本节）

适用 API（分项）：

1. `ClusteringData.dbscan()`
2. `make_moons(n_samples=self.n_samples, noise=self.dbscan_noise, random_state=self.random_state)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n_samples` | `400` | 样本数，来自 `ClusteringData` 默认属性 |
| `noise` | `0.08` | 月牙边界的随机噪声强度 |
| `random_state` | `42` | 随机种子，保证数据可复现 |
| 返回值 | `DataFrame` | 含 `x1`、`x2` 与 `true_label` 的数据表 |

### 示例代码

```python
X, y = make_moons(
    n_samples=self.n_samples,
    noise=self.dbscan_noise,
    random_state=self.random_state,
)
return DataFrame({"x1": X[:, 0], "x2": X[:, 1], "true_label": y})
```

### 理解重点

- `make_moons(...)` 生成的是两个弯曲的月牙形簇，这类结构很难用“离哪个中心更近”来描述。
- 当前数据只有两个特征维度，因此聚类结果可以直接画成二维散点图。
- `true_label` 是数据生成器自带的参考答案，方便后续和预测簇结果做视觉对照。

## 3. 特征列与 `true_label` 的角色

当前数据表结构如下：

- 特征列：`x1`、`x2`
- 参考标签列：`true_label`

### 示例代码

```python
y_true = data["true_label"].values
X = data.drop(columns=["true_label"])
```

### 理解重点

- `true_label` 不是监督学习里的训练标签，不能直接送进 DBSCAN。
- 当前流水线先把 `true_label` 单独保存到 `y_true`，只用于绘图对照。
- 如果把 `true_label` 当成普通特征一起输入模型，就会破坏无监督聚类的教学意义。

## 4. 标准化为什么放在训练前

### 参数速览（本节）

适用 API：`StandardScaler().fit_transform(X)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X` | 去掉 `true_label` 后的二维特征 | DBSCAN 的实际输入 |
| 返回值 | `X_scaled` | 标准化后的特征矩阵 |

### 示例代码

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 理解重点

- DBSCAN 用 `eps` 定义邻域半径，而邻域本质上也是基于距离计算的。
- 如果特征尺度差异明显，同样的 `eps` 在不同维度上的意义会被扭曲。
- 因此即使当前示例只有二维，流水线仍保留标准化步骤，体现规范做法。

## 常见坑

1. 把 `true_label` 当成普通特征一起送进模型。
2. 误以为 `true_label` 是当前 DBSCAN 的训练目标。
3. 忽略标准化，让 `eps` 的距离含义失真。
4. 看到双月牙效果很好，就误以为 DBSCAN 在所有密度分布上都同样稳定。

## 小结

- 当前 DBSCAN 数据来自 `ClusteringData.dbscan()`，底层使用的是 `make_moons(...)`。
- 数据表结构清晰：`x1`、`x2` 是特征，`true_label` 是仅用于对照的真实簇标签。
- 读懂数据来源、标签角色和标准化位置，是理解后续模型构建与训练流程的前提。
