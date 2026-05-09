---
title: EM 与 GMM — 数据构成
outline: deep
---

# 数据构成

## 本章目标

1. 明确本仓库 EM 数据来自 `ProbabilisticData.em()` 手动合成的 3 分量高斯混合数据。
2. 理解为何手动合成非球形数据——各分量具有不同的均值和标准差，充分展示 GMM 全协方差的建模能力。
3. 明确当前流程中 `true_label` 的角色——仅用于评估对比，**不参与 EM 训练**。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `ProbabilisticData.em()` | 方法 | 手动合成 3 分量非球形 GMM 数据 |
| `numpy.random.RandomState` | 类 | 种子随机数生成——保证数据可复现 |
| `em_means` | 属性 | 3 个分量的均值——$\{[0,0], [4,4], [-3,4]\}$ |
| `em_stds` | 属性 | 3 个分量的标准差——各维度不同，生成非球形簇 |
| `em_weights` | 属性 | 3 个分量的混合权重——$[0.5, 0.3, 0.2]$ |
| `true_label` | 列 | 真实分量标签——仅用于评估对比，EM 训练时不可见 |

## 1. 数据生成：`ProbabilisticData.em()`

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_samples` | `int` | 总样本数。`500`——适中规模，EM 可在秒级收敛 | `500`、`1000` |
| `em_n_components` | `int` | 高斯分量数。`3`——充分展示 GMM 的多分量建模 | `2`、`3`、`5` |
| `em_means` | `list[list]` | 各分量的 2 维均值。$\{[0,0], [4,4], [-3,4]\}$——分量间有显著间距 | 任意 `list[list[float]]` |
| `em_stds` | `list[list]` | 各分量的 2 维标准差。$\{[0.8,0.5], [0.6,1.0], [1.2,0.7]\}$——各维度不同，生成非球形簇 | 任意 `list[list[float]]` |
| `em_weights` | `list[float]` | 混合权重。$[0.5, 0.3, 0.2]$——分量不等权，更贴近实际情况 | 任意和为 1 的 `list[float]` |
| `random_state` | `int` | 随机种子。`42` | `42` |
| 返回值 | `DataFrame` | 含 `x1`、`x2`、`true_label` 三列 | — |

### 示例代码

```python
from data_generation.probabilistic import ProbabilisticData

data = ProbabilisticData().em()
# data.columns = ["x1", "x2", "true_label"]
# data.shape = (500, 3)
```

### 生成流程

```python
# 1. 按混合权重分配各分量样本数
counts = rng.multinomial(500, [0.5, 0.3, 0.2])
# counts ≈ [250, 150, 100]

# 2. 各分量独立生成样本
for k in range(3):
    X_k = rng.randn(counts[k], 2) * stds[k] + means[k]

# 3. 合并后随机打乱
X = np.vstack([X_0, X_1, X_2])
idx = rng.permutation(500)
```

### 理解重点

- 这是**手动合成**的数据——不是 `make_blobs` 生成的等权球形簇。每个分量有独立的均值和各向异性的标准差。
- 分量 1（`[0,0]`，权重 0.5）是最大最密的分量——标准差 $[0.8, 0.5]$，沿 $x_1$ 方向更宽。
- 分量 2（`[4,4]`，权重 0.3）较紧凑——标准差 $[0.6, 1.0]$，沿 $x_2$ 方向更宽。
- 分量 3（`[-3,4]`，权重 0.2）最小最散——标准差 $[1.2, 0.7]$，沿 $x_1$ 方向最分散。
- 这种非球形设计使得 `covariance_type="full"` 的 GMM 能正确建模椭圆簇——而 KMeans（球面聚类）则无法精确刻画。

## 2. 特征列与标签列

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame`，形状 `(500, 2)` | 含 2 个连续特征的特征矩阵，列名 `x1`、`x2` | `data.drop(columns=["true_label"])` |
| `true_label` | `Series`，形状 `(500,)` | 真实分量标签 $\{0, 1, 2\}$——**仅用于评估对比**，不参与训练 | `data["true_label"].values` |

### 理解重点

- `true_label` 是生成数据时记录的真实分量标号——EM 算法**完全不使用**这一列。
- 在流水线中，`y_true = data["true_label"].values` 在标准化前即被提取——它不参与后续任何计算，仅传入 `plot_clusters` 做可视化对比。
- 这与 KMeans/DBSCAN 分册中的 `true_label` 角色完全一致——聚类算法对标签"视而不见"。

## 3. 标准化

### 参数速览

适用 API：`StandardScaler().fit_transform(X)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame`，形状 `(500, 2)` | 全量特征矩阵 | `X` |
| 返回值 | `ndarray`，形状 `(500, 2)` | Z-score 标准化后的特征 | `X_scaled` |

### 示例代码

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 理解重点

- 当前流水线使用 `fit_transform` 在全量数据上做标准化——**不分训练/测试集**。这是聚类的标准做法。
- 与集成分类（有 `fit_transform`/`transform` 分离）形成对比——聚类没有"将训练集的统计量应用于测试集"的概念。
- 标准化对 EM 至关重要——高斯密度中的马氏距离 $(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})$ 对特征尺度敏感。未标准化的特征可能导致某个维度主导协方差矩阵。

## 4. 数据设计意图：与 KMeans/DBSCAN 的对比

| 数据维度 | KMeans | DBSCAN | EM (GMM) |
|---|---|---|---|
| 生成方式 | `make_blobs`（等权球形簇） | `make_blobs` + 均匀噪声 | **手动合成（不等权非球形）** |
| 簇形状 | 球形——适合 KMeans | 球形 + 噪声——适合 DBSCAN | **非球形椭圆——适合 GMM** |
| 样本数 | 500 | 500 | 500 |
| 特征维度 | 2 | 2 | 2 |
| 簇数 | 3 | 3 | 3 |
| 噪声点 | 无 | 有（均匀噪声） | 无 |
| 混合权重 | 等权 | 等权 | **不等权 $[0.5, 0.3, 0.2]$** |

### 理解重点

- EM 数据**刻意不球对称**——每个分量的 $x_1$ 和 $x_2$ 标准差各不相同，簇形状为拉伸的椭圆。
- 这种设计是为了展示 GMM 相对于 KMeans 的核心优势——`covariance_type="full"` 能正确建模椭圆形簇，而 KMeans 的球面假设无法处理。
- 不等权设计也更贴近实际数据——真实数据中的"簇"通常大小不匀。

## 数据可视化

![聚类分布图](../../../outputs/em/cluster_distribution.png)

## 常见坑

1. 不标准化就直接调用 `GaussianMixture`——不同维度尺度差异会导致某个维度主导协方差估计。
2. 把 `true_label` 当成训练标签——EM 是无监督算法，标签只用于评估。
3. 在球形数据（`make_blobs`）上对 GMM 使用 `covariance_type="full"`——参数过多可能过拟合，`spherical` 更合适。
4. 忽略数据打乱步骤——本数据集在生成后已随机打乱，但若自行构造数据，不打乱会影响可视化判断。

## 小结

- 当前 EM 数据来自手动合成的 3 分量 GMM：均值 $\{[0,0], [4,4], [-3,4]\}$，标准差各异，权重 $[0.5, 0.3, 0.2]$——非球形、不等权。
- 数据流为：手动采样 → DataFrame（`x1`、`x2` + `true_label`）→ 提取 `true_label` 用于评估 → 全量标准化。
- 非球形不等权的设计意图是展示 GMM 全协方差建模相对于 KMeans 球面聚类的核心优势。
