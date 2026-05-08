---
title: PCA 主成分分析 — 训练与预测
outline: deep
---

# 训练与预测

## 本章目标

1. 按源码顺序看清当前 PCA 流水线从数据复制到 2D/3D 降维图输出的完整步骤。
2. 理解 PCA 无监督训练、无切分、`transform()` 为输出的工程特征——与 LDA 既有相似又有本质差异。
3. 理解"双模型"设计（分别训练 2D 和 3D PCA）的教学意图——对比不同降维程度下的信息保留。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `pca_data.copy()` | 方法 | 复制原始数据，避免修改源对象 |
| `data.drop(columns=["label"])` | 操作 | 去掉伪标签列，保留 10 个特征作为训练输入 |
| `StandardScaler().fit_transform(X)` | 方法 | 对全量特征做一致性标准化——协方差矩阵计算的前置条件 |
| `train_model(X_scaled, n_components=2)` | 函数 | 训练 2D PCA 模型——无监督，不传标签 |
| `train_model(X_scaled, n_components=3)` | 函数 | 训练 3D PCA 模型——第二个独立模型，与 2D 模型互不依赖 |
| `model.transform(X_scaled)` | 方法 | 将 10 维特征投影到主成分空间——生成降维坐标 |
| `plot_dimensionality(...)` | 函数 | 绘制降维后的散点图——2D 和 3D 分别调用 |

## 1. 流水线起点：复制数据并拆出特征与伪标签

### 示例代码

```python
data = pca_data.copy()
X = data.drop(columns=["label"])
y = data["label"].values
```

### 理解重点

- `.copy()` 确保后续处理不修改全局 `pca_data`。
- `label` 被单独保存为 `y`——它**仅用于**最终的 `plot_dimensionality(...)` 着色，不参与 `train_model()`。
- 与 LDA 流水线最关键的区别：`y` 在这里不流入 `train_model()`，PCA 的训练完全不接触标签。

## 2. 标准化

### 参数速览

适用 API：`StandardScaler().fit_transform(X)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame`，形状 $(400, 10)$ | 去掉 `label` 后的全量特征矩阵 | `X` |
| 输出 | `ndarray` | $z_{ij} = (x_{ij} - \mu_j) / \sigma_j$，均值为 0 标准差为 1 | `X_scaled` |

### 示例代码

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 理解重点

- PCA 流水线**没有**训练/测试切分——当前实现为教学型简化，直接在全量数据上训练和投影。
- `fit_transform` 直接在全量数据上计算统计量并变换——目标是展示整体数据结构而非评估泛化能力。
- 标准化是必须的——协方差矩阵对特征尺度高度敏感。不标准化意味着"尺度最大的特征主导所有主成分方向"。

## 3. 第一阶段：训练 2D PCA 并生成 2D 图

### 参数速览

适用函数：`train_model(X_scaled, n_components=2)` → `model.transform(X_scaled)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_scaled` | `ndarray`，形状 $(400, 10)$ | 标准化后的特征矩阵——2D PCA 的输入 | `X_scaled` |
| `n_components` | `int` | 保留的主成分数。当前为 `2` | `2` |
| `X_transformed` | `ndarray`，形状 $(400, 2)$ | 投影到 2 维主成分空间后的坐标 | — |

### 示例代码

```python
# 训练 2D PCA
model = train_model(X_scaled, n_components=2)

# 投影到 2D 主成分空间
X_transformed = model.transform(X_scaled)

# 绘制 2D 降维图
plot_dimensionality(
    X_transformed,
    y=y,
    explained_variance_ratio=model.explained_variance_ratio_,
    title="PCA 降维 (2D)",
    dataset_name=DATASET,
    model_name=MODEL,
    mode="2d",
)
```

### 理解重点

- `PCA.fit(X_scaled)` 内部执行 SVD 分解 $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$，取 $\mathbf{V}$ 的前 2 行作为 `components_`。
- 2D 投影图展示的是数据方差最大的两个方向——对于当前低秩数据（3 个真实方向），它能展示约 2/3 的真实结构。
- `plot_dimensionality` 的 `mode='2d'` 参数决定输出 2D 散点图，轴标签为 `PC1 (xx.x%)` 和 `PC2 (xx.x%)`。

## 4. 第二阶段：训练 3D PCA 并生成 3D 图

### 参数速览

适用函数：`train_model(X_scaled, n_components=3)` → `model_3d.transform(X_scaled)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_scaled` | `ndarray`，形状 $(400, 10)$ | 标准化后的特征矩阵——3D PCA 的输入（与 2D PCA 共享同一份标准化数据） | `X_scaled` |
| `n_components` | `int` | 保留的主成分数。当前为 `3` | `3` |
| `X_3d` | `ndarray`，形状 $(400, 3)$ | 投影到 3 维主成分空间后的坐标 | — |

### 示例代码

```python
# 训练 3D PCA（第二个独立模型）
model_3d = train_model(X_scaled, n_components=3)

# 投影到 3D 主成分空间
X_3d = model_3d.transform(X_scaled)

# 绘制 3D 降维图
plot_dimensionality(
    X_3d,
    y=y,
    explained_variance_ratio=model_3d.explained_variance_ratio_,
    title="PCA 降维 (3D)",
    dataset_name=DATASET,
    model_name=MODEL,
    mode="3d",
)
```

### 理解重点

- **`model_3d` 是第二个独立模型**——它不是从 `model`（2D PCA）复用或扩展而来的，而是重新创建 `PCA(n_components=3)` 并重新 `fit()`。
- 3D 投影图展示数据方差最大的三个方向——对于当前低秩数据（3 个真实方向），它能几乎完整展示全部真实结构。
- 这种"先训练 2D、再训练 3D"的双模型设计，在 PCA 分册中旨在展示：**增加一个主成分（从 2D 到 3D）新增了多少结构信息？**

## 5. 2D 与 3D 模型的关系

### 理解重点

- 两个模型分别 `fit()` 了两次——从数学上讲，2D PCA 的前两个主成分与 3D PCA 的前两个主成分完全相同（都是相同的 $\mathbf{V}$ 的前两行）。工程上分开训练只是为了代码清晰。
- `X_transformed`（2D）的列是 `X_3d`（3D）的前两列——3D 投影完全包含 2D 投影的信息。
- 2D 累计解释方差 < 3D 累计解释方差——增加第 3 个主成分必然带来信息增益，但增益的幅度取决于数据的固有秩。

## 6. 用伪代码看完整流程

```python
data = pca_data.copy()
X = data.drop(columns=["label"])
y = data["label"].values

X_scaled = StandardScaler().fit_transform(X)

# 第一阶段：2D PCA
model = train_model(X_scaled, n_components=2)
X_2d = model.transform(X_scaled)
plot_dimensionality(X_2d, y=y, explained_variance_ratio=..., mode="2d")

# 第二阶段：3D PCA
model_3d = train_model(X_scaled, n_components=3)
X_3d = model_3d.transform(X_scaled)
plot_dimensionality(X_3d, y=y, explained_variance_ratio=..., mode="3d")
```

### 理解重点

- 流水线的主线非常清楚：取数 → 标准化 → 2D PCA 训练+投影+画图 → 3D PCA 训练+投影+画图。
- 这条链路里最关键的中间变量是：`X_scaled`（标准化特征）、2D PCA `model`（含 `components_` 和 `explained_variance_ratio_`）、3D PCA `model_3d`、两组投影结果和伪标签 `y`。
- 与 LDA 流水线最直观的区别：PCA 只画降维图（2D+3D），LDA 只画 2D 图——流程结构相似但输出维度不同。

## 常见坑

1. 以为 2D 和 3D 是同一个模型的不同 `n_components` 调用——实际上是两个独立 `PCA` 实例分别 `fit()`。
2. 忘记 `label` 在 PCA 中只用于着色——与 LDA 流水线中将 `y` 传入 `train_model()` 形成鲜明对比。
3. 误以为 PCA 流水线有分类评估步骤——PCA 是无监督降维，输出只有降维散点图。
4. 把 `PC1`、`PC2`、`PC3` 的标签当成分类标签——它们是主成分编号，不是类别编号。

## 小结

- 当前 PCA 流水线分为两阶段：2D PCA（训练+投影+画图）→ 3D PCA（训练+投影+画图）。这是所有算法分册中独一无二的双模型设计。
- `train_model` 在两次调用中分别传入 `n_components=2` 和 `n_components=3`——两次都只传 `X_scaled`（无监督）。
- 与 LDA 流水线的核心差异：无监督（无 `y` 入 `fit`）、维数自由（不受 $K-1$ 约束）、两阶段双模型、2D+3D 双图输出。
- 与分类分册的核心差异：输出是低维坐标而非类别预测、可视化是降维散点图而非混淆矩阵/ROC。
