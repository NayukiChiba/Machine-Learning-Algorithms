---
title: EM 与 GMM — 数据构成
outline: deep
---

# 数据构成

> 对应代码：`data_generation/probabilistic.py`、`data_generation/__init__.py`、`pipelines/probabilistic/em.py`
>  
> 相关对象：`ProbabilisticData.em()`、`em_data`

## 本章目标

1. 明确本仓库 EM 数据来自 `ProbabilisticData.em()` 的二维高斯混合构造逻辑。
2. 明确观测特征列、对比标签列以及它们在流水线中的边界。
3. 明确当前流程的标准化顺序，以及当前实现没有 train/test split 这一事实。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `ProbabilisticData.em()` | 方法 | 生成 EM / GMM 使用的二维混合高斯数据 |
| `em_data` | 变量 | 在 `data_generation/__init__.py` 中导出的数据对象 |
| `x1`、`x2` | 列名 | 当前流水线中的二维观测特征 |
| `true_label` | 列名 | 样本所属真实高斯分量，仅用于训练后对比 |

## 1. 本仓库数据入口

- 数据变量：`data_generation/__init__.py` 中导出的 `em_data`
- 生成来源：`data_generation/probabilistic.py` 中的 `ProbabilisticData.em()`
- 流水线使用：`pipelines/probabilistic/em.py` 中的 `data = em_data.copy()`

### 理解重点

- `em_data` 在导入时就已经生成完成，因此流水线里直接 `.copy()` 使用即可。
- 使用 `.copy()` 的目的，是避免后续标准化或调试过程意外修改原始数据对象。

## 2. 数据生成函数 `ProbabilisticData.em()`

### 参数速览（本节）

适用参数（本节）：

1. `n_samples`
2. `em_n_components`
3. `em_means`
4. `em_stds`
5. `em_weights`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n_samples` | `500` | 样本总数 |
| `random_state` | `42` | 随机种子，保证可复现 |
| `em_n_components` | `3` | 高斯分量数量 |
| `em_means` | `[[0.0, 0.0], [4.0, 4.0], [-3.0, 4.0]]` | 各分量中心位置 |
| `em_stds` | `[[0.8, 0.5], [0.6, 1.0], [1.2, 0.7]]` | 各分量在两个维度上的标准差 |
| `em_weights` | `[0.5, 0.3, 0.2]` | 各分量混合权重 |
| 返回值 | `DataFrame` | 含 `x1`、`x2`、`true_label` 的数据表 |

### 示例代码

```python
counts = rng.multinomial(self.n_samples, weights)

for k in range(n_components):
    mean = np.array(self.em_means[k])
    std = np.array(self.em_stds[k])
    X_k = rng.randn(counts[k], 2) * std + mean
    X_list.append(X_k)
    y_list.extend([k] * counts[k])
```

### 理解重点

- 当前数据不是通用 `make_blobs`，而是手工指定均值、方差和混合权重的二维高斯混合数据。
- 这样做的好处是更容易展示 GMM 对椭圆形簇和不均匀权重的建模能力。
- 这份数据是当前 EM 分册理解“软聚类”最直接的实验载体。

## 3. `true_label` 的语义与边界

当前数据表结构如下：

- 特征列：`x1`、`x2`
- 对比标签列：`true_label`

### 参数速览（本节）

适用列组（本节）：

1. 观测特征
2. 真实分量标签

| 列名 | 当前作用 |
|---|---|
| `x1`、`x2` | 训练 GMM 的观测特征 |
| `true_label` | 仅用于训练后与预测簇标签做可视化对比 |

### 示例代码

```python
y_true = data["true_label"].values
X = data.drop(columns=["true_label"])
```

### 理解重点

- `true_label` 不是监督学习里的训练标签，它只用于训练后对比聚类结果。
- 当前训练函数 `train_model(...)` 只接收 `X_train`，并不会使用 `true_label`。
- 这一点必须反复区分清楚，否则很容易把当前 EM 流程误读成监督学习。

## 4. 为什么当前数据必须是二维

### 参数速览（本节）

适用可视化函数：`plot_clusters(...)`

| 条件 | 当前要求 |
|---|---|
| 输入维度 | 只能是 2 维 |
| 当前特征列 | `x1`、`x2` |

### 示例代码

```python
if X.shape[1] != 2:
    raise ValueError(
        f"聚类散点绘制仅支持二维特征，当前为 {X.shape[1]} 维。"
    )
```

### 理解重点

- 当前聚类可视化函数只支持二维输入，因此 EM 数据也被设计成了二维。
- 这让训练后可以直接把预测簇和真实分量可视化对比出来。
- 如果后续想扩展到高维数据，就需要先降维再画图，或者改造当前可视化工具。

## 5. 标准化与当前流程边界

### 参数速览（本节）

适用 API（分项）：

1. `StandardScaler().fit_transform(X)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X` | `x1`、`x2` 组成的特征矩阵 | 参与标准化的观测数据 |
| 返回值 | `X_scaled` | 标准化后的二维特征 |

### 示例代码

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 理解重点

- 当前 EM 流水线会先对全量观测特征做标准化，再训练 GMM。
- 与前面几个监督学习分册不同，这里没有 train/test split，因此也不存在“只在训练集拟合标准化器”这一流程。
- 文档必须如实描述当前实现边界，不能套用监督学习分册的切分逻辑。

## 常见坑

1. 把 `true_label` 当成训练输入，误以为当前 EM 是监督学习流程。
2. 忽略当前数据是二维设计，直接把更高维特征送给 `plot_clusters(...)`。
3. 把监督学习里的 train/test split 惯性写进当前 EM 流程，和源码不一致。

## 小结

- 当前 EM 数据来自 `ProbabilisticData.em()`，底层是手工合成的二维高斯混合数据。
- 数据表结构很清晰：`x1`、`x2` 是观测特征，`true_label` 只用于训练后对比评估。
- 读懂数据来源、标准化顺序和标签边界，是理解后续 EM 训练与聚类可视化的前提。
