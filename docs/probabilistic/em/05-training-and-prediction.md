---
title: EM 与 GMM — 训练与预测
outline: deep
---

# 训练与预测

> 对应代码：`pipelines/probabilistic/em.py`、`model_training/probabilistic/em.py`
>  
> 运行方式：`python -m pipelines.probabilistic.em`

## 本章目标

1. 明确当前流水线从取数到生成聚类对比图的完整执行顺序。
2. 理解训练阶段、预测阶段和聚类可视化分别由哪个函数负责。
3. 明确当前 EM 实现没有 train/test split，而是在全量标准化后直接训练和预测。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `run()` | 函数 | EM / GMM 端到端流水线入口 |
| `StandardScaler` | 类 | 对观测特征做标准化 |
| `train_model(...)` | 函数 | 训练 GMM 模型 |
| `model.predict(X_scaled)` | 方法 | 输出每个样本的预测簇标签 |
| `plot_clusters(...)` | 函数 | 绘制预测标签和真实标签对比图 |

## 1. 端到端入口 `run()`

### 参数速览（本节）

适用函数：`run()`

| 项目 | 当前实现 |
|---|---|
| 数据源 | `em_data.copy()` |
| 对比标签列 | `true_label` |
| 特征列 | `x1`、`x2` |
| 训练入口 | `train_model(X_scaled)` |
| 预测入口 | `model.predict(X_scaled)` |
| 可视化入口 | `plot_clusters(...)` |

### 示例代码

```python
def run():
    data = em_data.copy()
    y_true = data["true_label"].values
    X = data.drop(columns=["true_label"])
```

### 理解重点

- 整个分册的运行入口就是 `pipelines/probabilistic/em.py` 里的 `run()`。
- 这个函数不负责实现 EM 迭代本身，而是把拆标签、标准化、训练、预测和画图串成一条流程。
- 相比监督学习流水线，这里的重点不在 train/test 切分，而在观测特征与隐变量视角的分离。

## 2. 训练前的数据准备顺序

### 参数速览（本节）

适用流程（分项）：

1. 取出 `true_label`
2. 删除 `true_label` 得到 `X`
3. 标准化 `X`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `y_true` | `data["true_label"].values` | 仅用于训练后对比 |
| `X` | `data.drop(columns=["true_label"])` | 真正参与训练的观测特征 |
| `X_scaled` | `StandardScaler().fit_transform(X)` | 标准化后的二维特征 |

### 示例代码

```python
y_true = data["true_label"].values
X = data.drop(columns=["true_label"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 理解重点

- 当前实现先明确切出 `true_label`，再把它从训练输入里删除，强调这不是监督学习标签。
- 当前分册会对全量特征直接做标准化，因为这里没有 train/test split。
- 这一步和回归分册的标准化流程不同，文档中必须明确区分。

## 3. 训练阶段：调用 `train_model(...)`

### 参数速览（本节）

适用函数：`train_model(X_scaled)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_scaled` | 标准化后的二维观测特征 | 当前直接传入 GMM 训练函数 |
| 返回值 | `model` | 已训练好的 `GaussianMixture` 模型 |

### 示例代码

```python
model = train_model(X_scaled)
```

### 理解重点

- 当前实现没有显式手写 E 步和 M 步循环，而是把这部分交给 `GaussianMixture.fit(...)`。
- 训练阶段最重要的副产物，不只是 `model` 对象，还有控制台里的分量数、协方差类型和 `log-likelihood`。
- 这些训练日志更接近“模型设定与收敛情况”，而不是最终聚类质量本身。

## 4. 预测阶段：从软聚类到硬标签输出

### 参数速览（本节）

适用流程（分项）：

1. `labels_pred = model.predict(X_scaled)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `model` | 已训练完成 GMM 模型 | 来自 `train_model(...)` 返回值 |
| `X_scaled` | 标准化后的二维特征 | 与训练时相同的输入 |
| `labels_pred` | 预测簇标签数组 | 用于聚类对比图 |

### 示例代码

```python
labels_pred = model.predict(X_scaled)
```

### 理解重点

- 当前流水线最终输出的是硬标签 `labels_pred`，方便做可视化对比。
- 但训练过程的核心仍然是软责任度更新，而不是一开始就硬分配簇标签。
- 也就是说，`predict(...)` 是一个方便展示的结果接口，不是 EM 本质的全部。

## 5. 预测后的聚类分布图输出

### 参数速览（本节）

适用函数：`plot_clusters(X_scaled, labels_pred=labels_pred, labels_true=y_true, title=..., dataset_name=..., model_name=...)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_scaled` | 标准化后的二维特征 | 聚类散点图输入 |
| `labels_pred` | 预测簇标签 | 左图着色依据 |
| `labels_true` | `y_true` | 右图对比用真实分量标签 |
| `title` | `"EM (GMM) 聚类分布"` | 图标题 |
| `dataset_name` | `"em"` | 输出目录名 |
| `model_name` | `"gmm"` | 输出文件名前缀 |

### 示例代码

```python
plot_clusters(
    X_scaled,
    labels_pred=labels_pred,
    labels_true=y_true,
    title="EM (GMM) 聚类分布",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 当前聚类图会同时展示“预测标签”和“真实标签”两张并排散点图。
- 这让你可以直观看到 GMM 学到的簇结构是否大致贴近真实分量。
- 这一步是当前 EM 分册的核心结果展示方式。

## 6. 用伪代码看完整流程

### 示例代码

```python
data = em_data.copy()
y_true = data["true_label"].values
X = data.drop(columns=["true_label"])

X_scaled = StandardScaler().fit_transform(X)

model = train_model(X_scaled)
labels_pred = model.predict(X_scaled)

plot_clusters(X_scaled, labels_pred=labels_pred, labels_true=y_true, ...)
```

### 理解重点

- 当前 EM 流水线的主线非常清楚：拆标签、标准化、训练、预测、画聚类对比图。
- 这条链路里最关键的中间变量是 `X_scaled`、训练后的 `model` 和预测簇标签 `labels_pred`。
- 只要把这条流程读清楚，整个 EM 分册的工程部分就基本串起来了。

## 常见坑

1. 把监督学习里的 train/test split 惯性套进当前 EM 流程，和源码不一致。
2. 只看到 `predict(...)` 的硬标签输出，就忽略了 EM 训练本质上是软聚类。
3. 把 `true_label` 当成训练标签，而不是训练后可视化对比用的参考列。

## 小结

- 当前流水线把标签拆分、全量标准化、GMM 训练、簇标签预测和聚类对比图输出串成了一条完整路径。
- 训练函数负责“得到 GMM 模型”，流水线函数负责“组织执行和产出结果”。
- 把这一层执行顺序读清楚，后续看评估与工程实现章节就会更顺。
