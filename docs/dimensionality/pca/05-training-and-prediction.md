---
title: PCA — 训练与预测
outline: deep
---

# 训练与预测

> 对应代码：`pipelines/dimensionality/pca.py`、`model_training/dimensionality/pca.py`
>  
> 运行方式：`python -m pipelines.dimensionality.pca`

## 本章目标

1. 明确当前流水线从取数到生成 2D / 3D 降维图的完整执行顺序。
2. 理解训练阶段、投影阶段和可视化阶段分别由哪个函数负责。
3. 明确当前 PCA 实现没有 train/test split，而是对全量数据标准化后直接训练和投影。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `run()` | 函数 | PCA 端到端流水线入口 |
| `StandardScaler` | 类 | 对全量特征做标准化 |
| `train_model(...)` | 函数 | 训练 PCA 模型 |
| `model.transform(X_scaled)` | 方法 | 生成低维主成分坐标 |
| `plot_dimensionality(...)` | 函数 | 绘制 2D 或 3D 降维图 |

## 1. 端到端入口 `run()`

### 参数速览（本节）

适用函数：`run()`

| 项目 | 当前实现 |
|---|---|
| 数据源 | `pca_data.copy()` |
| 标签列 | `label`（仅用于着色） |
| 训练入口 | `train_model(X_scaled, n_components=2)` 与 `train_model(X_scaled, n_components=3)` |
| 投影入口 | `model.transform(X_scaled)` |
| 可视化入口 | `plot_dimensionality(...)` |

### 示例代码

```python
def run():
    data = pca_data.copy()
    X = data.drop(columns=["label"])
    y = data["label"].values
```

### 理解重点

- 整个分册的运行入口就是 `pipelines/dimensionality/pca.py` 里的 `run()`。\n+- 这个函数不负责推导 PCA 数学原理，而是把取数、标准化、训练、投影和画图串成一条完整流程。\n+- 这里的 `y` 只用于图像着色，不参与训练。

## 2. 训练前的数据准备顺序

### 参数速览（本节）

适用流程（分项）：

1. 去掉 `label` 得到 `X`
2. 保留 `label` 得到 `y`
3. 全量标准化 `X`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X` | `data.drop(columns=["label"])` | 真正参与 PCA 训练的特征矩阵 |
| `y` | `data["label"].values` | 仅用于可视化着色 |
| `X_scaled` | `StandardScaler().fit_transform(X)` | 标准化后的高维特征 |

### 示例代码

```python
X = data.drop(columns=["label"])
y = data["label"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 理解重点

- 当前实现没有 train/test split，而是直接对全量特征做标准化。\n+- 这和监督学习流程不同，因为当前目标不是泛化预测，而是学习主成分结构。\n+- 标准化在这里尤其重要，否则不同量纲会直接影响主成分方向。

## 3. 训练阶段：先训练 2D PCA

### 参数速览（本节）

适用函数：`train_model(X_scaled, n_components=2)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_scaled` | 标准化后的高维特征 | 当前 PCA 训练输入 |
| `n_components` | `2` | 保留两个主成分 |
| 返回值 | `model` | 已训练好的 2D PCA 模型 |

### 示例代码

```python
model = train_model(X_scaled, n_components=2)
X_transformed = model.transform(X_scaled)
```

### 理解重点

- 当前实现第一步先训练 2D PCA，用于最基础的二维降维展示。\n+- 这里训练和投影是两步：先 `fit` 学方向，再 `transform` 得到坐标。\n+- `X_transformed` 才是后续 2D 可视化真正使用的数据。

## 4. 2D 可视化是如何接入流水线的

### 参数速览（本节）

适用函数：`plot_dimensionality(..., mode='2d')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_transformed` | 2D 投影结果 | 降维后的二维坐标 |
| `y` | `label` 数组 | 只用于着色 |
| `explained_variance_ratio` | `model.explained_variance_ratio_` | 用于轴标签展示解释方差比 |
| `mode` | `'2d'` | 输出二维图 |

### 示例代码

```python
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

- 当前 2D 图的轴标签会直接带上 `PC1`、`PC2` 的解释方差比。\n+- 这让图像不仅是“降维后坐标图”，也是解释方差信息的直观展示。\n+- `y` 只用于着色，不参与任何降维计算。

## 5. 为什么当前实现还要单独训练 3D PCA

### 参数速览（本节）

适用函数：`train_model(X_scaled, n_components=3)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n_components` | `3` | 保留三个主成分 |
| 返回值 | `model_3d` | 已训练好的 3D PCA 模型 |

### 示例代码

```python
model_3d = train_model(X_scaled, n_components=3)
X_3d = model_3d.transform(X_scaled)
```

### 理解重点

- 当前实现不是复用 2D 模型去拼出 3D 图，而是重新训练一个 `n_components=3` 的 PCA 模型。\n+- 这样做可以更自然地对应“保留 3 个主成分”的建模设定。\n+- 这也是当前分册一个很重要的工程细节，文档必须如实说明。

## 6. 3D 可视化是如何接入流水线的

### 参数速览（本节）

适用函数：`plot_dimensionality(..., mode='3d')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_3d` | 3D 投影结果 | 降维后的三维坐标 |
| `y` | `label` 数组 | 只用于着色 |
| `explained_variance_ratio` | `model_3d.explained_variance_ratio_` | 用于轴标签展示解释方差比 |
| `mode` | `'3d'` | 输出三维图 |

### 示例代码

```python
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

- 当前 3D 图会额外在 Z 轴上标注 `PC3` 及其解释方差比。\n+- 它的价值在于帮助读者比较“保留 2 个主成分”和“保留 3 个主成分”时的结构差异。\n+- 这也是当前流水线为什么会明确写出两套训练与投影流程。

## 7. 用伪代码看完整流程

### 示例代码

```python
data = pca_data.copy()
X = data.drop(columns=["label"])
y = data["label"].values

X_scaled = StandardScaler().fit_transform(X)

model = train_model(X_scaled, n_components=2)
X_transformed = model.transform(X_scaled)
plot_dimensionality(..., mode="2d")

model_3d = train_model(X_scaled, n_components=3)
X_3d = model_3d.transform(X_scaled)
plot_dimensionality(..., mode="3d")
```

### 理解重点

- 当前 PCA 流水线的主线非常清楚：取数、标准化、训练 2D PCA、投影、画图，再训练 3D PCA、投影、画图。\n+- 这条链路里最关键的中间变量是 `X_scaled`、2D 模型及其投影结果、3D 模型及其投影结果。\n+- 只要把这条流程走清楚，整个 pca 分册的工程部分就基本读懂了。

## 常见坑

1. 把 PCA 当前流程误写成监督学习流程，忽略它没有 train/test split。\n+2. 忘记 `transform(...)` 才是真正得到低维坐标的步骤。\n+3. 把 2D 与 3D 结果误写成同一个模型的不同展示，而忽略当前源码实际上训练了两个模型。

## 小结

- 当前流水线把数据准备、标准化、2D/3D PCA 训练、投影和可视化串成了一条完整路径。\n+- 训练函数负责“得到 PCA 模型”，流水线函数负责“组织执行和产出结果”。\n+- 把这一层执行顺序读清楚，后续看评估与工程实现章节就会更顺。
