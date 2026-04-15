---
title: LDA — 训练与预测
outline: deep
---

# 训练与预测

> 对应代码：`pipelines/dimensionality/lda.py`、`model_training/dimensionality/lda.py`
>  
> 运行方式：`python -m pipelines.dimensionality.lda`

## 本章目标

1. 明确当前流水线从取数到生成 2D 判别图的完整执行顺序。
2. 理解训练阶段、投影阶段和可视化阶段分别由哪个函数负责。
3. 明确当前 LDA 实现没有 train/test split，而是对全量数据标准化后直接训练和投影。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `run()` | 函数 | LDA 端到端流水线入口 |
| `StandardScaler` | 类 | 对全量特征做标准化 |
| `train_model(...)` | 函数 | 训练 LDA 模型 |
| `model.transform(X_scaled)` | 方法 | 生成判别子空间坐标 |
| `plot_dimensionality(...)` | 函数 | 绘制 2D 降维图 |

## 1. 端到端入口 `run()`

### 参数速览（本节）

适用函数：`run()`

| 项目 | 当前实现 |
|---|---|
| 数据源 | `lda_data.copy()` |
| 标签列 | `label`（参与训练，也用于着色） |
| 训练入口 | `train_model(X_scaled, y, n_components=2)` |
| 投影入口 | `model.transform(X_scaled)` |
| 可视化入口 | `plot_dimensionality(...)` |

### 示例代码

```python
def run():
    data = lda_data.copy()
    X = data.drop(columns=["label"])
    y = data["label"].values
```

### 理解重点

- 整个分册的运行入口就是 `pipelines/dimensionality/lda.py` 里的 `run()`。
- 这个函数不负责推导 Fisher 判别准则，而是把取数、标准化、训练、投影和画图串成一条完整流程。
- 和 PCA 最大不同之处，是这里的 `y` 真正参与训练。

## 2. 训练前的数据准备顺序

### 参数速览（本节）

适用流程（分项）：

1. 拆分 `X` 与 `label`
2. 全量标准化 `X`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X` | `data.drop(columns=["label"])` | 真正参与 LDA 训练的特征矩阵 |
| `y` | `data["label"].values` | LDA 训练所需监督标签 |
| `X_scaled` | `StandardScaler().fit_transform(X)` | 标准化后的高维特征 |

### 示例代码

```python
X = data.drop(columns=["label"])
y = data["label"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 理解重点

- 当前实现没有 train/test split，而是直接对全量特征做标准化。
- 这和监督分类流程不同，因为当前目标不是评估泛化预测，而是学习判别投影结构。
- 但与 PCA 不同的是，这里 `y` 不只是展示辅助信息，而是训练所必需的输入。

## 3. 训练阶段：调用 `train_model(...)`

### 参数速览（本节）

适用函数：`train_model(X_scaled, y, n_components=2)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_scaled` | 标准化后的特征 | 当前 LDA 训练输入 |
| `y` | 标签数组 | 当前 LDA 训练必须使用的类别信息 |
| `n_components` | `2` | 保留两个判别方向 |
| 返回值 | `model` | 已训练好的 LDA 模型 |

### 示例代码

```python
model = train_model(X_scaled, y, n_components=2)
```

### 理解重点

- 当前实现只训练一个 2D LDA 模型。
- 这不是随意选择，而是因为当前 Wine 数据有 3 个类别，理论上最多只能降到 `K - 1 = 2` 维。
- 当前训练阶段最关键的差异点，是标签在这里真实参与了判别方向学习。

## 4. 投影阶段：`transform(...)` 如何接入流程

### 参数速览（本节）

适用方法：`model.transform(X_scaled)`

| 方法 | 当前作用 |
|---|---|
| `fit(...)` | 学习判别方向 |
| `transform(...)` | 将原始高维数据投影到判别子空间 |

### 示例代码

```python
X_transformed = model.transform(X_scaled)
```

### 理解重点

- 当前训练并不会自动给出最终低维坐标结果，它先学习判别方向，再由 `transform(...)` 生成投影坐标。
- 当前 2D 图的输入，正是这一步之后得到的结果。
- 这使 `transform(...)` 成为连接“模型训练”和“可视化输出”的关键步骤。

## 5. 2D 可视化是如何接入流水线的

### 参数速览（本节）

适用函数：`plot_dimensionality(..., mode='2d')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_transformed` | 2D 投影结果 | 降维后的二维坐标 |
| `y` | `label` 数组 | 用于着色与图例 |
| `explained_variance_ratio` | `evr` | 若存在则用于轴标签展示 |
| `mode` | `'2d'` | 输出二维图 |

### 示例代码

```python
plot_dimensionality(
    X_transformed,
    y=y,
    explained_variance_ratio=evr,
    title="LDA 降维 (2D)",
    dataset_name=DATASET,
    model_name=MODEL,
    mode="2d",
)
```

### 理解重点

- 当前 LDA 分册只有 2D 图，不会像 PCA 那样再额外训练 3D 模型。
- 图中的 `y` 既是监督训练标签，也是着色依据，因此它在当前分册中有双重作用。
- 这也是当前 LDA 工程流程与 PCA 最容易混淆但最需要区分的地方。

## 6. 用伪代码看完整流程

### 示例代码

```python
data = lda_data.copy()
X = data.drop(columns=["label"])
y = data["label"].values

X_scaled = StandardScaler().fit_transform(X)

model = train_model(X_scaled, y, n_components=2)
X_transformed = model.transform(X_scaled)

plot_dimensionality(..., mode="2d")
```

### 理解重点

- 当前 LDA 流水线的主线非常清楚：取数、标准化、训练、投影、画图。
- 这条链路里最关键的中间变量是 `X_scaled`、训练后的 `model`、二维投影结果 `X_transformed` 和标签 `y`。
- 只要把这条流程走清楚，整个 lda 分册的工程部分就基本读懂了。

## 常见坑

1. 把 LDA 当前流程误写成 PCA 那种“标签只用于着色”的无监督流程。
2. 忘记 `transform(...)` 才是真正得到判别子空间坐标的步骤。
3. 误以为当前还会训练 3D LDA，而忽略这里受 `K - 1` 维上限约束。

## 小结

- 当前流水线把数据准备、标准化、2D LDA 训练、投影和可视化串成了一条完整路径。
- 训练函数负责“得到 LDA 模型”，流水线函数负责“组织执行和产出结果”。
- 把这一层执行顺序读清楚，后续看评估与工程实现章节就会更顺。
