---
title: EM 与 GMM — 工程实现
outline: deep
---

# 工程实现

> 对应代码：`data_generation/probabilistic.py`、`model_training/probabilistic/em.py`、`pipelines/probabilistic/em.py`、`result_visualization/cluster_plot.py`
>  
> 运行方式：`python -m pipelines.probabilistic.em`

## 本章目标

1. 看清当前 EM / GMM 分册在仓库中的模块分层与调用关系。
2. 理解从命令行入口到聚类对比图落盘，中间依次发生了什么。
3. 明确哪些逻辑属于数据层、训练层、流水线层和可视化层。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成层 | `data_generation/probabilistic.py` | `ProbabilisticData.em()` 构造二维混合高斯数据 |
| 数据导出层 | `data_generation/__init__.py` | 提供 `em_data` 给外部导入 |
| 训练层 | `model_training/probabilistic/em.py` | 定义 `train_model(...)` 并训练 GMM |
| 流水线层 | `pipelines/probabilistic/em.py` | 负责标准化、训练、预测、画图 |
| 聚类可视化层 | `result_visualization/cluster_plot.py` | 负责聚类对比图绘制与保存 |

## 1. 入口命令如何触发整条链路

### 示例代码

```bash
python -m pipelines.probabilistic.em
```

### 理解重点

- 这个命令会执行 `pipelines/probabilistic/em.py` 中的 `run()`。
- `run()` 是真正的工程入口，其他模块都被它按顺序调用。
- 所以理解工程实现时，最清晰的方式也是先从入口脚本往下追踪。

## 2. 模块之间的调用关系

### 示例代码

```python
from data_generation import em_data
from model_training.probabilistic.em import train_model
from result_visualization.cluster_plot import plot_clusters
```

### 理解重点

- `pipelines` 层不自己造数据、不自己实现模型，也不自己画图，而是扮演调度者角色。
- 这种分层使每个文件职责单一：数据文件只关心数据，训练文件只关心模型，画图文件只关心结果展示。
- 当前 EM 分册虽然只有一张核心图，但同样具备清晰的工程层次。

## 3. 流水线层真正负责什么

### 参数速览（本节）

适用逻辑（分项）：

1. 复制数据
2. 拆出 `true_label`
3. 提取观测特征
4. 标准化
5. 调用训练函数
6. 预测簇标签
7. 输出聚类对比图

| 步骤 | 所在文件 | 当前职责 |
|---|---|---|
| 读取 `em_data` | `pipelines/probabilistic/em.py` | 拿到统一数据入口 |
| `true_label` / `X` 拆分 | `pipelines/probabilistic/em.py` | 区分对比标签与训练输入 |
| 标准化 `X` | `pipelines/probabilistic/em.py` | 生成模型训练输入 |
| 调用 `train_model(...)` | `pipelines/probabilistic/em.py` | 获得训练好的 GMM |
| `predict(...)` + `plot_clusters(...)` | `pipelines/probabilistic/em.py` | 完成结果输出 |

### 理解重点

- 当前仓库没有使用 `Pipeline` 类，而是把标准化、训练和预测显式写在 `run()` 中。
- 这种写法更适合教学，因为每一步都能直接看到变量名和执行顺序。
- EM 分册最容易被误读的地方，就是 `true_label` 的边界，因此显式写法反而更有帮助。

## 4. 为什么这里没有 train/test split

### 理解重点

- 当前 EM 分册属于无监督聚类流程，不像监督学习那样围绕训练/测试标签误差展开。
- 当前实现选择直接在全量数据上标准化、训练和预测，以便更直观看到整体聚类分布。
- 这是一种教学型简化实现，文档需要如实说明，而不是套用监督学习的默认结构。

## 5. 训练层真正负责什么

### 参数速览（本节）

适用函数：`train_model(...)`

| 输出项 | 作用 |
|---|---|
| `model` | 返回已训练好的 `GaussianMixture` 模型 |
| 控制台日志 | 打印分量数、协方差类型、训练耗时和 `log-likelihood` |

### 理解重点

- 训练层并不负责拆分标签，也不负责绘制聚类图。
- 它的核心任务是构建 GMM、执行 EM 训练，并输出与模型设定和收敛相关的日志。
- 和监督学习分册相比，这里打印的重点从“预测性能”转成了“模型结构与收敛状态”。

## 6. 可视化层真正负责什么

### 参数速览（本节）

适用函数：`plot_clusters(...)`

| 参数名 | 当前用途 |
|---|---|
| `X` | 聚类散点图输入 |
| `labels_pred` | 预测簇标签着色 |
| `labels_true` | 真实分量标签对比 |
| `dataset_name` | 决定输出目录，如 `em` |
| `model_name` | 决定文件名前缀，如 `gmm` |

### 理解重点

- 当前聚类可视化层只关心二维特征和标签着色，不关心 GMM 内部参数。
- 当同时传入 `labels_pred` 和 `labels_true` 时，它会并排绘制两张图进行对比。
- 这正是当前 EM 分册核心评估结果的来源。

## 7. 常量 `DATASET` 和 `MODEL` 的作用

### 参数速览（本节）

适用常量：

1. `DATASET = "em"`
2. `MODEL = "gmm"`

| 常量 | 当前作用 |
|---|---|
| `DATASET` | 决定图片输出的上层目录 |
| `MODEL` | 决定图片文件名前缀 |

### 理解重点

- 这两个常量的作用，不是影响模型训练，而是统一结果文件的命名和归档。
- 这样当前 EM 分册生成的图像会被稳定保存到固定位置。
- 这也是为什么当前工程结构适合后续继续扩展更多聚类可视化结果。

## 8. 从命令到结果图的执行链

### 示例代码

```python
python -m pipelines.probabilistic.em
    -> run()
    -> em_data.copy()
    -> data["true_label"] / data.drop(columns=["true_label"])
    -> StandardScaler().fit_transform(...)
    -> train_model(...)
    -> model.predict(...)
    -> plot_clusters(...)
    -> savefig(...)
```

### 理解重点

- 这条链里最关键的中间产物有三个：`X_scaled`、训练后的 `model`、预测簇标签 `labels_pred`。
- 一旦这些中间变量理解清楚，整个 EM 分册的代码结构就基本串起来了。
- 文档中的各章节，其实就是在拆解这条执行链上的不同环节。

## 常见坑

1. 把 `pipelines` 层和 `model_training` 层职责混在一起，误以为训练函数负责全部工程流程。
2. 不理解为什么这里没有 train/test split，从而误把当前流程套成监督学习结构。
3. 忽略 `true_label`、`DATASET` 和 `MODEL` 的作用，看不懂聚类对比图和输出目录为什么能稳定生成。

## 小结

- 当前 EM / GMM 实现采用了清晰的分层结构：数据层、训练层、流水线层、可视化层各司其职。
- 入口脚本负责调度，训练模块负责模型，画图模块负责结果呈现。
- 这种结构既方便阅读，也方便后续继续补聚类指标、责任度可视化或分量数选择实验。
