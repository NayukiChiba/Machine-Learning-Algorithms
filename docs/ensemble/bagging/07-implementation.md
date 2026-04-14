---
title: Bagging 与随机森林 — 工程实现
outline: deep
---

# 工程实现

> 对应代码：`data_generation/ensemble.py`、`model_training/ensemble/bagging.py`、`pipelines/ensemble/bagging.py`、`result_visualization/confusion_matrix.py`、`result_visualization/roc_curve.py`
>  
> 运行方式：`python -m pipelines.ensemble.bagging`

## 本章目标

1. 看清当前 Bagging 分册在仓库中的模块分层与调用关系。
2. 理解从命令行入口到分类评估图落盘，中间依次发生了什么。
3. 明确哪些逻辑属于数据层、训练层、流水线层和可视化层。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成层 | `data_generation/ensemble.py` | `EnsembleData.bagging()` 生成数据 |
| 数据导出层 | `data_generation/__init__.py` | 提供 `bagging_data` 给外部导入 |
| 训练层 | `model_training/ensemble/bagging.py` | 定义 `train_model(...)` 并训练 Bagging 模型 |
| 流水线层 | `pipelines/ensemble/bagging.py` | 负责切分、标准化、训练、预测、画图 |
| 混淆矩阵层 | `result_visualization/confusion_matrix.py` | 负责混淆矩阵绘制与保存 |
| ROC 曲线层 | `result_visualization/roc_curve.py` | 在概率输出可用时绘制与保存 |

## 1. 入口命令如何触发整条链路

### 示例代码

```bash
python -m pipelines.ensemble.bagging
```

### 理解重点

- 这个命令会执行 `pipelines/ensemble/bagging.py` 中的 `run()`。
- `run()` 是真正的工程入口，其他模块都被它按顺序调用。
- 所以理解工程实现时，最清晰的方式也是先从入口脚本往下追踪。

## 2. 模块之间的调用关系

### 示例代码

```python
from data_generation import bagging_data
from model_training.ensemble.bagging import train_model
from result_visualization.confusion_matrix import plot_confusion_matrix
from result_visualization.roc_curve import plot_roc_curve
```

### 理解重点

- `pipelines` 层不自己造数据、不自己实现模型，也不自己画图，而是扮演调度者角色。
- 这种分层使每个文件职责单一：数据文件只关心数据，训练文件只关心模型，画图文件只关心结果展示。
- 当前 Bagging 分册虽然图像种类不多，但依然具备清晰的工程层次。

## 3. 流水线层真正负责什么

### 参数速览（本节）

适用逻辑（分项）：

1. 复制数据
2. 拆分特征与标签
3. 分层切分训练/测试集
4. 标准化
5. 调用训练函数
6. 预测类别
7. 条件性预测概率
8. 输出评估图像

| 步骤 | 所在文件 | 当前职责 |
|---|---|---|
| 读取 `bagging_data` | `pipelines/ensemble/bagging.py` | 拿到统一数据入口 |
| `X` / `y` 拆分 | `pipelines/ensemble/bagging.py` | 明确特征与标签 |
| 分层切分 | `pipelines/ensemble/bagging.py` | 保持类别比例稳定 |
| 标准化 | `pipelines/ensemble/bagging.py` | 生成训练和测试输入 |
| 调用 `train_model(...)` | `pipelines/ensemble/bagging.py` | 获得训练好的 Bagging 模型 |
| `predict(...)` / 条件性 `predict_proba(...)` + 画图函数 | `pipelines/ensemble/bagging.py` | 完成结果输出 |

### 理解重点

- 当前仓库没有使用 `Pipeline` 类，也没有学习曲线或特征重要性可视化。
- 这种显式写法更适合教学，因为每一步都能直接看到变量名和执行顺序。
- Bagging 分册最容易被误读的地方，是把“并行集成 + OOB”误写成 boosting 流程，因此工程章节要特别明确当前实现边界。

## 4. 训练层真正负责什么

### 参数速览（本节）

适用函数：`train_model(...)`

| 输出项 | 作用 |
|---|---|
| `model` | 返回已训练好的 `BaggingClassifier` 模型 |
| 控制台日志 | 打印采样配置、Bootstrap 配置、OOB 得分和训练耗时 |

### 理解重点

- 训练层并不负责切分数据，也不负责绘制混淆矩阵或 ROC 曲线。
- 它的核心任务是构建 Bagging 模型、拟合训练数据，并回显当前采样配置。
- 和 GBDT / LightGBM 相比，这里日志最有代表性的内容是 `OOB 得分`。

## 5. 为什么 ROC 曲线是条件性输出

### 示例代码

```python
if hasattr(model, "predict_proba"):
    y_scores = model.predict_proba(X_test_s)
    plot_roc_curve(...)
```

### 理解重点

- 当前 ROC 曲线的输出依赖模型是否支持概率预测接口。
- 这说明当前实现没有把 ROC 曲线当作无条件固定结果，而是先检查能力边界。
- 文档必须把这点写清楚，否则读者容易误解成“每次都一定会有 ROC 图”。

## 6. 常量 `DATASET` 和 `MODEL` 的作用

### 参数速览（本节）

适用常量：

1. `DATASET = "bagging"`
2. `MODEL = "bagging"`

| 常量 | 当前作用 |
|---|---|
| `DATASET` | 决定图片输出的上层目录 |
| `MODEL` | 决定图片文件名前缀 |

### 理解重点

- 这两个常量的作用，不是影响模型训练，而是统一结果文件的命名和归档。
- 这样当前 Bagging 分册生成的图像会被稳定保存到固定位置。
- 这也是为什么当前工程结构适合后续继续扩展更多分类评估图表。

## 7. sklearn 版本兼容逻辑体现在哪里

### 理解重点

- 当前训练模块会先尝试使用 `estimator=base` 构造 `BaggingClassifier`。
- 如果旧版本 sklearn 不支持该参数名，就回退到 `base_estimator=base`。
- 这说明当前工程实现已经考虑到了 API 兼容边界，但这属于工程细节，不属于算法差异。

## 8. 从命令到结果图的执行链

### 示例代码

```python
python -m pipelines.ensemble.bagging
    -> run()
    -> bagging_data.copy()
    -> train_test_split(..., stratify=y)
    -> StandardScaler().fit_transform(...)
    -> train_model(...)
    -> model.predict(...)
    -> plot_confusion_matrix(...)
    -> conditional plot_roc_curve(...)
    -> savefig(...)
```

### 理解重点

- 这条链里最关键的中间产物有三个：训练后的 `model`、预测类别 `y_pred`、可选的概率输出 `y_scores`。
- 一旦这些中间变量理解清楚，整个 bagging 分册的代码结构就基本串起来了。
- 文档中的各章节，其实就是在拆解这条执行链上的不同环节。

## 常见坑

1. 把 `pipelines` 层和 `model_training` 层职责混在一起，误以为训练函数负责全部工程流程。
2. 不理解为什么当前分册没有学习曲线或特征重要性图，从而误读当前实现能力边界。
3. 忽略 `OOB 得分`、条件性 ROC 和 sklearn 兼容逻辑这些关键工程细节。

## 小结

- 当前 Bagging 实现采用了清晰的分层结构：数据层、训练层、流水线层、可视化层各司其职。
- 入口脚本负责调度，训练模块负责模型，画图模块负责结果呈现。
- 这种结构既方便阅读，也方便后续继续补数值指标、学习曲线或决策边界可视化实验。
