---
title: LightGBM — 工程实现
outline: deep
---

# 工程实现

> 对应代码：`data_generation/ensemble.py`、`model_training/ensemble/lightgbm.py`、`pipelines/ensemble/lightgbm.py`、`result_visualization/confusion_matrix.py`、`result_visualization/roc_curve.py`、`result_visualization/feature_importance.py`
>  
> 运行方式：`python -m pipelines.ensemble.lightgbm`

## 本章目标

1. 看清当前 LightGBM 分册在仓库中的模块分层与调用关系。
2. 理解从命令行入口到三类结果图落盘，中间依次发生了什么。
3. 明确哪些逻辑属于数据层、训练层、流水线层和可视化层。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成层 | `data_generation/ensemble.py` | `EnsembleData.lightgbm()` 生成数据 |
| 数据导出层 | `data_generation/__init__.py` | 提供 `lightgbm_data` 给外部导入 |
| 训练层 | `model_training/ensemble/lightgbm.py` | 定义 `train_model(...)` 并训练 LightGBM 模型 |
| 流水线层 | `pipelines/ensemble/lightgbm.py` | 负责切分、标准化、训练、预测、画图 |
| 混淆矩阵层 | `result_visualization/confusion_matrix.py` | 负责混淆矩阵绘制与保存 |
| ROC 曲线层 | `result_visualization/roc_curve.py` | 负责 ROC 曲线绘制与保存 |
| 特征重要性层 | `result_visualization/feature_importance.py` | 负责特征重要性图绘制与保存 |

## 1. 入口命令如何触发整条链路

### 示例代码

```bash
python -m pipelines.ensemble.lightgbm
```

### 理解重点

- 这个命令会执行 `pipelines/ensemble/lightgbm.py` 中的 `run()`。
- `run()` 是真正的工程入口，其他模块都被它按顺序调用。
- 所以理解工程实现时，最清晰的方式也是先从入口脚本往下追踪。

## 2. 模块之间的调用关系

### 示例代码

```python
from data_generation import lightgbm_data
from model_training.ensemble.lightgbm import train_model
from result_visualization.confusion_matrix import plot_confusion_matrix
from result_visualization.roc_curve import plot_roc_curve
from result_visualization.feature_importance import plot_feature_importance
```

### 理解重点

- `pipelines` 层不自己造数据、不自己实现模型，也不自己画图，而是扮演调度者角色。
- 这种分层使每个文件职责单一：数据文件只关心数据，训练文件只关心模型，画图文件只关心结果展示。
- 当前 LightGBM 分册比 XGBoost 多了分类专属的混淆矩阵和 ROC 曲线输出，因此结果层更偏分类诊断。

## 3. 流水线层真正负责什么

### 参数速览（本节）

适用逻辑（分项）：

1. 复制数据
2. 拆分特征与标签
3. 保存 `feature_names`
4. 分层切分训练/测试集
5. 标准化
6. 调用训练函数
7. 预测类别与概率
8. 输出三类图像

| 步骤 | 所在文件 | 当前职责 |
|---|---|---|
| 读取 `lightgbm_data` | `pipelines/ensemble/lightgbm.py` | 拿到统一数据入口 |
| `X` / `y` 拆分 | `pipelines/ensemble/lightgbm.py` | 明确特征与标签 |
| 保存 `feature_names` | `pipelines/ensemble/lightgbm.py` | 供重要性图使用 |
| 分层切分 | `pipelines/ensemble/lightgbm.py` | 保持类别比例稳定 |
| 标准化 | `pipelines/ensemble/lightgbm.py` | 生成训练和测试输入 |
| 调用 `train_model(...)` | `pipelines/ensemble/lightgbm.py` | 获得训练好的 LightGBM 模型 |
| `predict(...)` / `predict_proba(...)` + 画图函数 | `pipelines/ensemble/lightgbm.py` | 完成结果输出 |

### 理解重点

- 当前仓库没有使用 `Pipeline` 类，也没有显式验证集和早停流程。
- 这种显式写法更适合教学，因为每一步都能直接看到变量名和执行顺序。
- LightGBM 分册最容易被误读的地方，是它虽然是树模型，但当前实现依然显式做了标准化和多分类概率输出。

## 4. 训练层真正负责什么

### 参数速览（本节）

适用函数：`train_model(...)`

| 输出项 | 作用 |
|---|---|
| `model` | 返回已训练好的 `LGBMClassifier` 模型 |
| 控制台日志 | 打印关键 boosting 超参数和训练耗时 |

### 理解重点

- 训练层并不负责切分数据，也不负责绘制混淆矩阵、ROC 曲线或特征重要性图。
- 它的核心任务是构建 LightGBM 模型、拟合训练数据，并回显当前超参数配置。
- 和 XGBoost 分册类似，这里日志的重点也是超参数集合而不是单个结构性指标。

## 5. 为什么这里同时需要 `predict(...)` 和 `predict_proba(...)`

### 示例代码

```python
y_pred = model.predict(X_test_s)
y_scores = model.predict_proba(X_test_s)
```

### 理解重点

- `predict(...)` 输出离散类别，用于混淆矩阵。
- `predict_proba(...)` 输出每个类别的概率，用于 ROC 曲线。
- 这说明当前 LightGBM 分册的评估不仅看最终分类结果，还看概率排序能力。

## 6. 常量 `DATASET` 和 `MODEL` 的作用

### 参数速览（本节）

适用常量：

1. `DATASET = "lightgbm"`
2. `MODEL = "lightgbm"`

| 常量 | 当前作用 |
|---|---|
| `DATASET` | 决定图片输出的上层目录 |
| `MODEL` | 决定图片文件名前缀 |

### 理解重点

- 这两个常量的作用，不是影响模型训练，而是统一结果文件的命名和归档。
- 这样当前 LightGBM 分册生成的图像会被稳定保存到固定位置。
- 这也是为什么当前工程结构适合后续继续扩展更多分类评估图表。

## 7. 缺少 `lightgbm` 依赖时会发生什么

### 理解重点

- 当前训练模块会先尝试导入 `LGBMClassifier`。
- 如果导入失败，`train_model(...)` 会抛出明确的 `ImportError`，提醒当前环境缺少 `lightgbm` 依赖。
- 这说明当前工程实现已经考虑到了外部依赖边界，但没有在流水线中内置自动安装逻辑。

## 8. 从命令到结果图的执行链

### 示例代码

```python
python -m pipelines.ensemble.lightgbm
    -> run()
    -> lightgbm_data.copy()
    -> train_test_split(..., stratify=y)
    -> StandardScaler().fit_transform(...)
    -> train_model(...)
    -> model.predict(...)
    -> model.predict_proba(...)
    -> plot_confusion_matrix(...)
    -> plot_roc_curve(...)
    -> plot_feature_importance(...)
    -> savefig(...)
```

### 理解重点

- 这条链里最关键的中间产物有四个：`feature_names`、训练后的 `model`、预测类别 `y_pred`、预测概率 `y_scores`。
- 一旦这些中间变量理解清楚，整个 lightgbm 分册的代码结构就基本串起来了。
- 文档中的各章节，其实就是在拆解这条执行链上的不同环节。

## 常见坑

1. 把 `pipelines` 层和 `model_training` 层职责混在一起，误以为训练函数负责全部工程流程。
2. 不理解为什么当前分册需要概率输出，从而误读 ROC 曲线的输入来源。
3. 忽略 `feature_names`、`DATASET` 和 `MODEL` 的作用，看不懂三类图像和输出目录为什么能稳定生成。

## 小结

- 当前 LightGBM 实现采用了清晰的分层结构：数据层、训练层、流水线层、可视化层各司其职。
- 入口脚本负责调度，训练模块负责模型，画图模块负责结果呈现。
- 这种结构既方便阅读，也方便后续继续补数值指标、学习曲线、验证集或早停实验。
