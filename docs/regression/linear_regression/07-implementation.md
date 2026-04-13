---
title: 线性回归 — 工程实现
outline: deep
---

# 工程实现

> 对应代码：`data_generation/regression.py`、`model_training/regression/linear_regression.py`、`pipelines/regression/linear_regression.py`、`result_visualization/residual_plot.py`、`result_visualization/learning_curve.py`
>  
> 运行方式：`python -m pipelines.regression.linear_regression`

## 本章目标

1. 看清当前线性回归分册在仓库中的模块分层与调用关系。
2. 理解从命令行入口到结果图落盘，中间依次发生了什么。
3. 明确哪些逻辑属于数据层、训练层、流水线层和可视化层。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成层 | `data_generation/regression.py` | `RegressionData.linear_regression()` 构造数据 |
| 数据导出层 | `data_generation/__init__.py` | 提供 `linear_regression_data` 给外部导入 |
| 训练层 | `model_training/regression/linear_regression.py` | 定义 `train_model(...)` 并训练线性回归模型 |
| 流水线层 | `pipelines/regression/linear_regression.py` | 负责切分、训练、预测、画图 |
| 残差可视化层 | `result_visualization/residual_plot.py` | 负责残差图绘制与保存 |
| 学习曲线可视化层 | `result_visualization/learning_curve.py` | 负责学习曲线绘制与保存 |

## 1. 入口命令如何触发整条链路

### 示例代码

```bash
python -m pipelines.regression.linear_regression
```

### 理解重点

- 这个命令会执行 `pipelines/regression/linear_regression.py` 中的 `run()`。
- `run()` 是真正的工程入口，其他模块都被它按顺序调用。
- 所以理解工程实现时，最清晰的方式也是先从入口脚本往下追踪。

## 2. 模块之间的调用关系

### 示例代码

```python
from data_generation import linear_regression_data
from model_training.regression.linear_regression import train_model
from result_visualization.residual_plot import plot_residuals
from result_visualization.learning_curve import plot_learning_curve
```

### 理解重点

- `pipelines` 层不自己造数据、不自己实现模型，也不自己画图，而是扮演调度者角色。
- 这种分层使每个文件职责单一：数据文件只关心数据，训练文件只关心模型，画图文件只关心结果展示。
- 当前线性回归分册结构虽然简单，但已经具备清晰的工程层次。

## 3. 流水线层真正负责什么

### 参数速览（本节）

适用逻辑（分项）：

1. 复制数据
2. 拆分特征与标签
3. 切分训练/测试集
4. 调用训练函数
5. 预测测试集
6. 输出残差图与学习曲线

| 步骤 | 所在文件 | 当前职责 |
|---|---|---|
| 读取 `linear_regression_data` | `pipelines/regression/linear_regression.py` | 拿到统一数据入口 |
| `X` / `y` 拆分 | `pipelines/regression/linear_regression.py` | 明确特征与标签 |
| 训练/测试切分 | `pipelines/regression/linear_regression.py` | 生成训练和评估输入 |
| 调用 `train_model(...)` | `pipelines/regression/linear_regression.py` | 获得训练好的模型 |
| `predict(...)` + 两种画图函数 | `pipelines/regression/linear_regression.py` | 完成结果输出 |

### 理解重点

- 当前仓库没有使用 `Pipeline` 类，也没有把预处理单独抽成步骤对象。
- 这种显式写法更适合教学，因为每一步都能直接看到变量名和执行顺序。
- 对线性回归这种基础分册来说，这种简单实现反而更利于理解。

## 4. 训练层真正负责什么

### 参数速览（本节）

适用函数：`train_model(...)`

| 输出项 | 作用 |
|---|---|
| `model` | 返回已训练好的 `LinearRegression` 模型 |
| 控制台日志 | 打印截距和各特征系数 |

### 理解重点

- 训练层并不负责切分数据，也不负责计算残差图或学习曲线。
- 它的核心任务只有两个：构建 `LinearRegression()`，拟合训练数据。
- 同时它还承担了教学型日志输出职责，这也是为什么要打印截距和系数。

## 5. 可视化层真正负责什么

### 参数速览（本节）

适用函数（分项）：

1. `plot_residuals(...)`
2. `plot_learning_curve(...)`

| 参数名 | 当前用途 |
|---|---|
| `dataset_name` | 决定保存目录，如 `linear_regression` |
| `model_name` | 决定文件名前缀，如 `linear_regression` |
| `title` | 决定图上的展示标题 |

### 理解重点

- 残差图函数只关心真实值和预测值，不关心模型内部细节。
- 学习曲线函数只关心模型实例、训练数据和评分方式，不直接依赖训练后的 `model` 对象。
- 这种设计说明当前工程实现已经在复用同一套通用可视化工具。

## 6. 为什么学习曲线单独传一个新的 `LinearRegression()`

### 示例代码

```python
plot_learning_curve(
    LinearRegression(),
    X_train,
    y_train,
    scoring="r2",
    ...
)
```

### 理解重点

- `plot_learning_curve(...)` 内部会调用 `sklearn.model_selection.learning_curve(...)`，它需要一个可重复训练的模型实例。
- 因此这里传入的是新的 `LinearRegression()`，而不是已经在完整训练集上拟合完成的 `model`。
- 这说明学习曲线和测试集预测虽然都属于“评估”，但两者的执行方式并不相同。

## 7. 常量 `DATASET` 和 `MODEL` 的作用

### 参数速览（本节）

适用常量：

1. `DATASET = "linear_regression"`
2. `MODEL = "linear_regression"`

| 常量 | 当前作用 |
|---|---|
| `DATASET` | 决定图片输出的上层目录 |
| `MODEL` | 决定图片文件名前缀 |

### 理解重点

- 这两个常量的作用，不是影响模型训练，而是统一结果文件的命名和归档。
- 这样同一算法分册下生成的残差图和学习曲线会被放到稳定位置。
- 这也是为什么当前工程结构适合继续扩展更多图表或指标输出。

## 8. 从命令到结果图的执行链

### 示例代码

```python
python -m pipelines.regression.linear_regression
    -> run()
    -> linear_regression_data.copy()
    -> train_test_split(...)
    -> train_model(...)
    -> model.predict(...)
    -> plot_residuals(...)
    -> plot_learning_curve(LinearRegression(), ...)
    -> savefig(...)
```

### 理解重点

- 这条链里最关键的中间产物有三个：`X_train` / `X_test`、训练后的 `model`、测试集预测 `y_pred`。
- 一旦这些中间变量理解清楚，整个 linear_regression 分册的代码结构就基本串起来了。
- 文档中的各章节，其实就是在拆解这条执行链上的不同环节。

## 常见坑

1. 把 `pipelines` 层和 `model_training` 层职责混在一起，误以为训练函数负责全部工程流程。
2. 不理解为什么学习曲线不用训练好的 `model`，从而误读 `plot_learning_curve(...)` 的工作方式。
3. 忽略 `DATASET` 和 `MODEL` 的作用，看不懂图像文件为什么会落到固定目录。

## 小结

- 当前线性回归实现采用了清晰的分层结构：数据层、训练层、流水线层、可视化层各司其职。
- 入口脚本负责调度，训练模块负责模型，画图模块负责结果呈现。
- 这种结构既方便阅读，也方便后续继续补指标打印或更复杂的回归实验。
