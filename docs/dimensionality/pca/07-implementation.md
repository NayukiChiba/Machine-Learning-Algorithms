---
title: PCA — 工程实现
outline: deep
---

# 工程实现

> 对应代码：`data_generation/dimensionality.py`、`model_training/dimensionality/pca.py`、`pipelines/dimensionality/pca.py`、`result_visualization/dimensionality_plot.py`
>  
> 运行方式：`python -m pipelines.dimensionality.pca`

## 本章目标

1. 看清当前 PCA 分册在仓库中的模块分层与调用关系。
2. 理解从命令行入口到 2D / 3D 降维图落盘，中间依次发生了什么。
3. 明确哪些逻辑属于数据层、训练层、流水线层和可视化层。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成层 | `data_generation/dimensionality.py` | `DimensionalityData.pca()` 构造低秩高维数据 |
| 数据导出层 | `data_generation/__init__.py` | 提供 `pca_data` 给外部导入 |
| 训练层 | `model_training/dimensionality/pca.py` | 定义 `train_model(...)` 并训练 PCA 模型 |
| 流水线层 | `pipelines/dimensionality/pca.py` | 负责标准化、训练、投影、画图 |
| 可视化层 | `result_visualization/dimensionality_plot.py` | 负责 2D / 3D 降维图绘制与保存 |

## 1. 入口命令如何触发整条链路

### 示例代码

```bash
python -m pipelines.dimensionality.pca
```

### 理解重点

- 这个命令会执行 `pipelines/dimensionality/pca.py` 中的 `run()`。
- `run()` 是真正的工程入口，其他模块都被它按顺序调用。
- 所以理解工程实现时，最清晰的方式也是先从入口脚本往下追踪。

## 2. 模块之间的调用关系

### 示例代码

```python
from data_generation import pca_data
from model_training.dimensionality.pca import train_model
from result_visualization.dimensionality_plot import plot_dimensionality
```

### 理解重点

- `pipelines` 层不自己造数据、不自己实现 PCA，也不自己画图，而是扮演调度者角色。
- 这种分层使每个文件职责单一：数据文件只关心数据，训练文件只关心模型，画图文件只关心结果展示。
- 当前 PCA 分册虽然没有学习曲线或分类指标，但工程层次依然很清晰。

## 3. 流水线层真正负责什么

### 参数速览（本节）

适用逻辑（分项）：

1. 复制数据
2. 拆分特征与着色标签
3. 全量标准化
4. 训练 2D PCA
5. 投影并画 2D 图
6. 训练 3D PCA
7. 投影并画 3D 图

| 步骤 | 所在文件 | 当前职责 |
|---|---|---|
| 读取 `pca_data` | `pipelines/dimensionality/pca.py` | 拿到统一数据入口 |
| `X` / `label` 拆分 | `pipelines/dimensionality/pca.py` | 区分训练输入与着色标签 |
| 标准化 `X` | `pipelines/dimensionality/pca.py` | 生成 PCA 训练输入 |
| 调用 `train_model(...)` | `pipelines/dimensionality/pca.py` | 获得 2D / 3D PCA 模型 |
| `transform(...)` + `plot_dimensionality(...)` | `pipelines/dimensionality/pca.py` | 完成投影与结果输出 |

### 理解重点

- 当前仓库没有使用 `Pipeline` 类，也没有把 2D / 3D 输出封装进一个统一循环。
- 这种显式写法更适合教学，因为每一步都能直接看到变量名和执行顺序。
- PCA 分册最容易被误读的地方，是把 `label` 当成训练输入，因此这里要特别明确它只用于着色。

## 4. 为什么这里没有 train/test split

### 理解重点

- 当前 PCA 分册属于无监督降维流程，不像监督学习那样围绕训练/测试标签误差展开。
- 当前实现选择直接在全量数据上标准化、训练和投影，以便更直观看到整体数据结构。
- 这是一种教学型简化实现，文档需要如实说明，而不是套用监督学习默认结构。

## 5. 训练层真正负责什么

### 参数速览（本节）

适用函数：`train_model(...)`

| 输出项 | 作用 |
|---|---|
| `model` | 返回已训练好的 `PCA` 模型 |
| 控制台日志 | 打印 `n_components`、解释方差比、累计解释方差和训练耗时 |

### 理解重点

- 训练层并不负责标准化数据，也不负责画降维图。
- 它的核心任务是学习主成分方向，并输出解释方差相关日志。
- 和监督学习分册相比，这里日志的重点从“预测表现”转成了“信息保留结构”。

## 6. 为什么当前会分别训练 2D 和 3D 两个 PCA 模型

### 示例代码

```python
model = train_model(X_scaled, n_components=2)
model_3d = train_model(X_scaled, n_components=3)
```

### 理解重点

- 当前实现不是拿一个 3D 模型随便截断出 2D 结果，也不是拿一个 2D 模型扩展出 3D 结果。
- 它明确训练了两个不同的 PCA 模型，用于两个不同的可视化场景。
- 这能让 2D 和 3D 各自的解释方差输出与投影图完全对应当前设定。

## 7. 常量 `DATASET` 和 `MODEL` 的作用

### 参数速览（本节）

适用常量：

1. `DATASET = "pca"`
2. `MODEL = "pca"`

| 常量 | 当前作用 |
|---|---|
| `DATASET` | 决定图片输出的上层目录 |
| `MODEL` | 决定图片文件名前缀 |

### 理解重点

- 这两个常量的作用，不是影响模型训练，而是统一结果文件的命名和归档。
- 这样当前 PCA 分册生成的图像会被稳定保存到固定位置。
- 这也是为什么当前工程结构适合后续继续扩展更多降维图表。

## 8. 从命令到结果图的执行链

### 示例代码

```python
python -m pipelines.dimensionality.pca
    -> run()
    -> pca_data.copy()
    -> data.drop(columns=["label"])
    -> StandardScaler().fit_transform(...)
    -> train_model(..., n_components=2)
    -> model.transform(...)
    -> plot_dimensionality(..., mode="2d")
    -> train_model(..., n_components=3)
    -> model_3d.transform(...)
    -> plot_dimensionality(..., mode="3d")
    -> savefig(...)
```

### 理解重点

- 这条链里最关键的中间产物有四个：`X_scaled`、2D 模型及其投影结果、3D 模型及其投影结果、着色标签 `y`。
- 一旦这些中间变量理解清楚，整个 pca 分册的代码结构就基本串起来了。
- 文档中的各章节，其实就是在拆解这条执行链上的不同环节。

## 常见坑

1. 把 `pipelines` 层和 `model_training` 层职责混在一起，误以为训练函数负责全部工程流程。
2. 不理解为什么当前分册没有分类指标图，从而误读当前实现目标。
3. 忽略 `label`、2D / 3D 双模型和解释方差日志这些关键工程细节。

## 小结

- 当前 PCA 实现采用了清晰的分层结构：数据层、训练层、流水线层、可视化层各司其职。
- 入口脚本负责调度，训练模块负责模型，画图模块负责结果呈现。
- 这种结构既方便阅读，也方便后续继续补累计解释方差曲线、重构误差或更多降维方法对比实验。
