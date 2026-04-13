---
title: 决策树回归 — 工程实现
outline: deep
---

# 工程实现

> 对应代码：`data_generation/regression.py`、`model_training/regression/decision_tree.py`、`pipelines/regression/decision_tree.py`、`result_visualization/residual_plot.py`、`result_visualization/feature_importance.py`、`result_visualization/learning_curve.py`
>  
> 运行方式：`python -m pipelines.regression.decision_tree`

## 本章目标

1. 看清当前决策树回归分册在仓库中的模块分层与调用关系。
2. 理解从命令行入口到三类结果图落盘，中间依次发生了什么。
3. 明确哪些逻辑属于数据层、训练层、流水线层和可视化层。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成层 | `data_generation/regression.py` | `RegressionData.decision_tree()` 加载数据 |
| 数据导出层 | `data_generation/__init__.py` | 提供 `decision_tree_regression_data` 给外部导入 |
| 训练层 | `model_training/regression/decision_tree.py` | 定义 `train_model(...)` 并训练树模型 |
| 流水线层 | `pipelines/regression/decision_tree.py` | 负责切分、训练、预测、画图 |
| 残差可视化层 | `result_visualization/residual_plot.py` | 负责残差图绘制与保存 |
| 特征重要性层 | `result_visualization/feature_importance.py` | 负责特征重要性图绘制与保存 |
| 学习曲线层 | `result_visualization/learning_curve.py` | 负责学习曲线绘制与保存 |

## 1. 入口命令如何触发整条链路

### 示例代码

```bash
python -m pipelines.regression.decision_tree
```

### 理解重点

- 这个命令会执行 `pipelines/regression/decision_tree.py` 中的 `run()`。
- `run()` 是真正的工程入口，其他模块都被它按顺序调用。
- 所以理解工程实现时，最清晰的方式也是先从入口脚本往下追踪。

## 2. 模块之间的调用关系

### 示例代码

```python
from data_generation import decision_tree_regression_data
from model_training.regression.decision_tree import train_model
from result_visualization.residual_plot import plot_residuals
from result_visualization.feature_importance import plot_feature_importance
from result_visualization.learning_curve import plot_learning_curve
```

### 理解重点

- `pipelines` 层不自己造数据、不自己实现模型，也不自己画图，而是扮演调度者角色。
- 这种分层使每个文件职责单一：数据文件只关心数据，训练文件只关心模型，画图文件只关心结果展示。
- 当前决策树分册比线性回归多了一层特征重要性可视化，但整体分层依然很清楚。

## 3. 流水线层真正负责什么

### 参数速览（本节）

适用逻辑（分项）：

1. 复制数据
2. 拆分特征与标签
3. 保存 `feature_names`
4. 切分训练/测试集
5. 调用训练函数
6. 预测测试集
7. 输出三类图像

| 步骤 | 所在文件 | 当前职责 |
|---|---|---|
| 读取 `decision_tree_regression_data` | `pipelines/regression/decision_tree.py` | 拿到统一数据入口 |
| `X` / `y` 拆分 | `pipelines/regression/decision_tree.py` | 明确特征与标签 |
| 保存 `feature_names` | `pipelines/regression/decision_tree.py` | 供重要性图使用 |
| 训练/测试切分 | `pipelines/regression/decision_tree.py` | 生成训练和评估输入 |
| 调用 `train_model(...)` | `pipelines/regression/decision_tree.py` | 获得训练好的树模型 |
| `predict(...)` + 三种画图函数 | `pipelines/regression/decision_tree.py` | 完成结果输出 |

### 理解重点

- 当前仓库没有使用 `Pipeline` 类，也没有把预处理单独抽成步骤对象。
- 这种显式写法更适合教学，因为每一步都能直接看到变量名和执行顺序。
- 对决策树这种强调结构和图像诊断的分册来说，这种实现方式也很利于阅读。

## 4. 训练层真正负责什么

### 参数速览（本节）

适用函数：`train_model(...)`

| 输出项 | 作用 |
|---|---|
| `model` | 返回已训练好的 `DecisionTreeRegressor` 模型 |
| 控制台日志 | 打印训练耗时、树深度、叶子节点数 |

### 理解重点

- 训练层并不负责切分数据，也不负责绘制残差图、重要性图或学习曲线。
- 它的核心任务是构建树模型、拟合训练数据，并输出与模型结构相关的日志。
- 和线性回归分册相比，这里打印的重点从“系数”变成了“结构复杂度”。

## 5. 为什么训练和预测阶段显式使用 `.values`

### 示例代码

```python
model = train_model(X_train.values, y_train.values)
y_pred = model.predict(X_test.values)
```

### 理解重点

- 当前流水线把 `DataFrame` / `Series` 转成 NumPy 数组后再送入训练和预测。
- 这是当前实现选择的一种输入形式，不是决策树算法的强制要求。
- 因为特征名已经单独保存在 `feature_names` 里，所以这种转换不会影响后续特征重要性图的可读性。

## 6. 可视化层真正负责什么

### 参数速览（本节）

适用函数（分项）：

1. `plot_residuals(...)`
2. `plot_feature_importance(...)`
3. `plot_learning_curve(...)`

| 函数 | 当前作用 |
|---|---|
| `plot_residuals(...)` | 看预测误差分布 |
| `plot_feature_importance(...)` | 看特征分裂贡献 |
| `plot_learning_curve(...)` | 看样本量变化下的训练/验证走势 |

### 理解重点

- 残差图函数只关心真实值和预测值。
- 特征重要性函数只关心模型的 `feature_importances_` 与特征名映射。
- 学习曲线函数则会重新训练多个模型实例来观察不同样本规模下的表现。

## 7. 为什么学习曲线单独传一个新的 `DecisionTreeRegressor(...)`

### 示例代码

```python
plot_learning_curve(
    DecisionTreeRegressor(max_depth=6, random_state=42),
    X_train.values,
    y_train.values,
    scoring="r2",
    ...
)
```

### 理解重点

- `plot_learning_curve(...)` 内部会调用 `sklearn.model_selection.learning_curve(...)`，它需要一个可重复训练的模型实例。
- 因此这里传入的是新的 `DecisionTreeRegressor(...)`，而不是已经在完整训练集上拟合完成的 `model`。
- 当前这里保留了最关键的对比参数 `max_depth=6` 和 `random_state=42`，以便和主训练配置保持一致。

## 8. 常量 `DATASET` 和 `MODEL` 的作用

### 参数速览（本节）

适用常量：

1. `DATASET = "decision_tree_reg"`
2. `MODEL = "decision_tree"`

| 常量 | 当前作用 |
|---|---|
| `DATASET` | 决定图片输出的上层目录 |
| `MODEL` | 决定图片文件名前缀 |

### 理解重点

- 这两个常量的作用，不是影响模型训练，而是统一结果文件的命名和归档。
- 需要注意，`DATASET` 的值是 `decision_tree_reg`，它是结果目录名，不是文档分册路由名。
- 这也是为什么当前工程结构适合继续扩展更多图表或指标输出。

## 9. 从命令到结果图的执行链

### 示例代码

```python
python -m pipelines.regression.decision_tree
    -> run()
    -> decision_tree_regression_data.copy()
    -> train_test_split(...)
    -> train_model(...)
    -> model.predict(...)
    -> plot_residuals(...)
    -> plot_feature_importance(...)
    -> plot_learning_curve(DecisionTreeRegressor(...), ...)
    -> savefig(...)
```

### 理解重点

- 这条链里最关键的中间产物有四个：`feature_names`、`X_train` / `X_test`、训练后的 `model`、测试集预测 `y_pred`。
- 一旦这些中间变量理解清楚，整个 decision_tree 分册的代码结构就基本串起来了。
- 文档中的各章节，其实就是在拆解这条执行链上的不同环节。

## 常见坑

1. 把 `pipelines` 层和 `model_training` 层职责混在一起，误以为训练函数负责全部工程流程。
2. 不理解为什么学习曲线不用训练好的 `model`，从而误读 `plot_learning_curve(...)` 的工作方式。
3. 忽略 `feature_names`、`DATASET` 和 `MODEL` 的作用，看不懂特征重要性图和输出目录为什么能稳定生成。

## 小结

- 当前决策树回归实现采用了清晰的分层结构：数据层、训练层、流水线层、可视化层各司其职。
- 入口脚本负责调度，训练模块负责模型，画图模块负责结果呈现。
- 这种结构既方便阅读，也方便后续继续补指标打印、树结构导出或更多树模型实验。
