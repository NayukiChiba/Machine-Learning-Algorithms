---
title: DBSCAN 密度聚类 — 训练与预测
outline: deep
---

# 训练与预测

## 本章目标

1. 按源码顺序看清当前 DBSCAN 流水线从数据复制到聚类输出的完整步骤。
2. 理解 DBSCAN 无训练/测试切分、无 `predict()` 的工程特征——与分类分册有本质差异。
3. 理解 `fit()` 即聚类、`labels_` 即输出的流程特征。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `dbscan_data.copy()` | 方法 | 复制原始数据，避免修改源对象 |
| `StandardScaler` | 类 | 对全量特征做一致性标准化——`eps` 邻域判定的前置条件 |
| `train_model(...)` | 函数 | 调用 `DBSCAN.fit()` 执行密度聚类，返回模型对象 |
| `model.fit(X_scaled)` | 方法 | 在标准化特征上执行密度聚类——标签生成、簇分配一步到位 |
| `model.labels_` | 属性 | 聚类结果的唯一输出——每个样本被分配到簇编号或 $-1$（噪声） |

## 1. 流水线起点：复制数据并拆出特征与对照标签

### 示例代码

```python
data = dbscan_data.copy()
y_true = data["true_label"].values
X = data.drop(columns=["true_label"])
```

### 理解重点

- `.copy()` 确保后续处理不修改全局 `dbscan_data`。
- `true_label` 被单独保存为 `y_true`——它**不参与**后续的 `fit()`，只在最后的 `plot_clusters(...)` 中作为对照显示。
- 这是聚类与分类最根本的工程差异——没有"标签=训练目标"的概念。

## 2. 标准化

### 参数速览

适用 API：`StandardScaler().fit_transform(X)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame`，形状 $(400, 2)$ | 去掉 `true_label` 后的全量特征矩阵 | `X` |
| 输出 | `ndarray` | $z_{ij} = (x_{ij} - \mu_j) / \sigma_j$，均值为 0 标准差为 1 | `X_scaled` |

### 示例代码

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 理解重点

- DBSCAN 流水线**没有**训练/测试切分——无监督聚类不需要验证集。
- `fit_transform` 直接在全量数据上计算统计量并变换——不存在测试集数据泄露的风险。
- 标准化是必须的——`eps=0.3` 作为绝对距离阈值，其几何意义依赖于各维度尺度一致。

## 3. 密度聚类：`fit()` 即训练 + 预测

### 参数速览

适用 API：`train_model(X_scaled)` → `model.fit(X_scaled)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_scaled` | `ndarray`，形状 $(400, 2)$ | 标准化后的全量特征矩阵——DBSCAN 的唯一输入 | `X_scaled` |
| 无 `y` 参数 | — | DBSCAN 是无监督算法——`fit(X)` 不接受标签 | — |

### 示例代码

```python
model = train_model(X_scaled)
```

### 理解重点

- `DBSCAN.fit(X_scaled)` 内部流程：遍历所有点 → 对每个点计算 $\epsilon$ 邻域 → 判定核心/边界/噪声 → 沿密度可达关系 BFS/DFS 扩展簇 → 生成 `labels_`。
- 这**既是训练也是预测**——DBSCAN 没有分离的 `fit()` + `predict()` 两阶段。对于新样本，sklearn 的 DBSCAN 无法直接预测簇归属。
- 与分类模型比较：分类流程是 `fit(X, y)` → `predict(X_test)`，DBSCAN 流程是 `fit(X)` → `labels_`。

## 4. 获取聚类结果

### 参数速览

| 属性名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `labels_` | `ndarray`，形状 $(400,)$ | 每个样本的簇分配标签，$\{-1, 0, 1, \dots, k-1\}$ | `model.labels_` |
| `n_clusters` | `int` | 排除 $-1$ 后的簇数量 | 当前期望为 `2` |
| `n_noise` | `int` | 标签为 $-1$ 的噪声点数量 | 取决于 `noise=0.08` 下落入月牙间隙的样本数 |

### 示例代码

```python
labels = model.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
```

### 理解重点

- `model.labels_` 是 DBSCAN 的唯一输出——它就是这个聚类算法的"预测结果"。
- `-1` 标签是 DBSCAN 特定的噪声标记——与分类模型中的 `predict` 输出不同，噪声点不属于任何类别。
- 没有 `model.predict(X_test)`——这是 DBSCAN 在预测能力上的天然限制（sklearn 实现）。新样本的簇归属需通过其他方式推断（如最近邻搜索）。

## 5. 聚类结果可视化

### 参数速览

适用函数：`plot_clusters(X_scaled, labels_pred=model.labels_, labels_true=y_true, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_scaled` | `ndarray`，形状 $(400, 2)$ | 标准化后的全量特征，用于散点图的坐标 | `X_scaled` |
| `labels_pred` | `ndarray`，形状 $(400,)$ | DBSCAN 的预测簇标签（含 $-1$ 噪声） | `model.labels_` |
| `labels_true` | `ndarray`，形状 $(400,)$ | 真实簇标签（仅用于视觉对照） | `y_true` |

### 示例代码

```python
plot_clusters(
    X_scaled,
    labels_pred=model.labels_,
    labels_true=y_true,
    title="DBSCAN 聚类分布",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- `plot_clusters(...)` 是当前 DBSCAN 分册唯一的可视化函数——与分类分册的四类评估（混淆矩阵+ROC+决策边界+学习曲线）完全不同。
- 双侧对照布局：左侧显示 `labels_pred`（算法聚类结果），右侧显示 `labels_true`（真实簇标签）——帮助读者直观判断算法是否恢复了真实结构。
- `labels_pred` 中的噪声点（$-1$）通常以特殊颜色（如黑色或灰色）标记，便于识别。

## 常见坑

1. 期望当前流水线有训练/测试切分——无监督聚类不划分训练集和验证集。
2. 误以为 `model.predict(X_new)` 可用——sklearn 的 DBSCAN 不支持 `predict()`，`fit()` 即得到全部聚类结果。
3. 把 `true_label` 当成 `fit()` 的输入——它仅用于最终的可视化对照。
4. 把 DBSCAN 的训练流程理解成"先生成模型，再用于预测新数据"——它是直接对输入数据进行标记，没有分离的训练和推理阶段。

## 小结

- 当前 DBSCAN 流水线极为简洁：复制数据 → 剥离 `true_label` → 全量标准化 → `fit(X)` 密度聚类 → `labels_` 直接作为聚类输出 → 可视化对照。
- 与分类分册的核心差异：无切分（无 `train_test_split`）、无监督标签、无 `predict()`（`fit()` 即输出）、无概率输出、无混淆矩阵/ROC/学习曲线。
- 这种简洁性源于 DBSCAN 的算法特性——它是直接对数据点做密度连通分析，而非训练一个可泛化的判别函数。
