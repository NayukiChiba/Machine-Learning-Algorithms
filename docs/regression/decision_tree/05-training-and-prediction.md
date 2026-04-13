---
title: 决策树回归 — 训练与预测
outline: deep
---

# 训练与预测

> 对应代码：`pipelines/regression/decision_tree.py`、`model_training/regression/decision_tree.py`
>  
> 运行方式：`python -m pipelines.regression.decision_tree`

## 本章目标

1. 明确当前流水线从取数到生成三类图像的完整执行顺序。
2. 理解训练阶段、预测阶段、特征重要性图和学习曲线分别由哪个函数负责。
3. 明确当前决策树实现没有标准化步骤，并且训练/预测阶段显式使用 `.values`。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `run()` | 函数 | 决策树回归端到端流水线入口 |
| `train_test_split(...)` | 函数 | 拆分训练集与测试集 |
| `train_model(...)` | 函数 | 训练决策树回归模型 |
| `model.predict(X_test.values)` | 方法 | 对测试集做回归预测 |
| `plot_residuals(...)` | 函数 | 绘制残差分析图 |
| `plot_feature_importance(...)` | 函数 | 绘制特征重要性图 |
| `plot_learning_curve(...)` | 函数 | 绘制学习曲线 |

## 1. 端到端入口 `run()`

### 参数速览（本节）

适用函数：`run()`

| 项目 | 当前实现 |
|---|---|
| 数据源 | `decision_tree_regression_data.copy()` |
| 标签列 | `price` |
| 切分方式 | `test_size=0.2, random_state=42` |
| 训练入口 | `train_model(X_train.values, y_train.values)` |
| 预测入口 | `model.predict(X_test.values)` |
| 可视化入口 | `plot_residuals(...)`、`plot_feature_importance(...)`、`plot_learning_curve(...)` |

### 示例代码

```python
def run():
    data = decision_tree_regression_data.copy()
    X = data.drop(columns=["price"])
    y = data["price"]
    feature_names = list(X.columns)
```

### 理解重点

- 整个分册的运行入口就是 `pipelines/regression/decision_tree.py` 里的 `run()`。
- 这个函数不负责实现树的分裂搜索本身，而是把取数、训练、预测和可视化串成一条完整链路。
- `feature_names` 会在这里提前保存下来，后续供特征重要性图使用。

## 2. 训练前的数据准备顺序

### 参数速览（本节）

适用 API（分项）：

1. `train_test_split(X, y, test_size=0.2, random_state=42)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `test_size` | `0.2` | 测试集占比 |
| `random_state` | `42` | 保证可复现划分 |
| 返回值 | `X_train`、`X_test`、`y_train`、`y_test` | 训练/测试集拆分结果 |

### 示例代码

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 理解重点

- 当前流水线在数据切分后直接进入训练，没有额外的标准化步骤。
- 这一点和 `svr`、`regularization` 分册不同，文档里必须明确区分。
- 决策树当前实现更关注分裂结构，而不是量纲统一后的距离或系数惩罚。

## 3. 训练阶段：调用 `train_model(...)`

### 参数速览（本节）

适用函数：`train_model(X_train.values, y_train.values)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_train.values` | 训练特征数组 | 当前显式转为 NumPy 数组后传入 |
| `y_train.values` | 训练标签数组 | 当前显式转为 NumPy 数组后传入 |
| 返回值 | `model` | 已训练好的 `DecisionTreeRegressor` 模型 |

### 示例代码

```python
model = train_model(X_train.values, y_train.values)
```

### 理解重点

- 当前训练阶段最重要的结果，不只是 `model` 对象，还有训练日志中的树深度、叶子节点数和耗时信息。
- `values` 的使用是当前实现选择的一种输入形式，并不影响后续继续用保存好的 `feature_names` 画图。
- 这种写法也让训练层更明确地接收数值矩阵而不是带标签表结构。

## 4. 预测阶段：直接调用 `predict(...)`

### 参数速览（本节）

适用流程（分项）：

1. `y_pred = model.predict(X_test.values)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `model` | 已训练完成模型 | 来自 `train_model(...)` 返回值 |
| `X_test.values` | 测试特征数组 | 当前显式转为 NumPy 数组后传入 |
| `y_pred` | 预测值数组 | 用于残差图分析 |

### 示例代码

```python
y_pred = model.predict(X_test.values)
```

### 理解重点

- 当前仓库没有额外封装 `predict_model(...)`，而是直接使用 scikit-learn 统一的 `predict(...)` 接口。
- 由于当前分册没有标准化步骤，所以预测阶段也直接使用原始测试特征的数组形式。
- 训练和预测两边都使用 `.values`，是为了保持输入形式一致。

## 5. 预测后的残差图与特征重要性图输出

### 参数速览（本节）

适用函数（分项）：

1. `plot_residuals(...)`
2. `plot_feature_importance(...)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `title`（残差图） | `"决策树回归 残差分析"` | 图标题 |
| `title`（重要性图） | `"决策树回归 特征重要性"` | 图标题 |
| `dataset_name` | `"decision_tree_reg"` | 输出目录名 |
| `model_name` | `"decision_tree"` | 输出文件名前缀 |
| `feature_names` | `list(X.columns)` | 给特征重要性图提供真实列名 |

### 示例代码

```python
plot_residuals(
    y_test,
    y_pred,
    title="决策树回归 残差分析",
    dataset_name=DATASET,
    model_name=MODEL,
)

plot_feature_importance(
    model,
    feature_names=feature_names,
    title="决策树回归 特征重要性",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 残差图负责观察预测误差分布。
- 特征重要性图负责观察树模型在分裂过程中更依赖哪些特征。
- 当前分册之所以提前保存 `feature_names`，就是为了把 `feature_importances_` 和真实列名对应起来。

## 6. 学习曲线是如何接入流水线的

### 参数速览（本节）

适用函数：`plot_learning_curve(DecisionTreeRegressor(max_depth=6, random_state=42), X_train.values, y_train.values, scoring='r2', ...)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `model` | `DecisionTreeRegressor(max_depth=6, random_state=42)` | 一个新的未训练模型实例 |
| `X` | `X_train.values` | 使用训练集特征 |
| `y` | `y_train.values` | 使用训练集标签 |
| `scoring` | `"r2"` | 当前分册使用的评分指标 |

### 示例代码

```python
plot_learning_curve(
    DecisionTreeRegressor(max_depth=6, random_state=42),
    X_train.values,
    y_train.values,
    scoring="r2",
    title="决策树回归 学习曲线",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 学习曲线不是直接基于训练好的 `model` 绘制，而是由 `plot_learning_curve(...)` 内部重新做不同训练规模下的交叉验证。
- 这里传入的新模型实例只保留了当前对比最关键的 `max_depth=6` 和 `random_state=42`。
- 这一步关注的是样本量变化下的训练/验证走势，而不是某一次测试集预测结果。

## 7. 用伪代码看完整流程

### 示例代码

```python
data = decision_tree_regression_data.copy()
X = data.drop(columns=["price"])
y = data["price"]
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(...)

model = train_model(X_train.values, y_train.values)
y_pred = model.predict(X_test.values)

plot_residuals(...)
plot_feature_importance(model, feature_names=feature_names, ...)
plot_learning_curve(DecisionTreeRegressor(max_depth=6, random_state=42), ...)
```

### 理解重点

- 当前决策树流水线的主线非常清楚：取数、切分、训练、预测、画残差图、画特征重要性图、画学习曲线。
- 这条链路里最关键的中间变量是 `feature_names`、训练后的 `model` 和预测结果 `y_pred`。
- 只要把这条流程走清楚，整个 decision_tree 分册的工程部分就基本读懂了。

## 常见坑

1. 把其他分册里的标准化流程误套到当前决策树实现上。
2. 误以为特征重要性图可以在不保留 `feature_names` 的情况下自动解释特征含义。
3. 只看到模型训练成功，没有继续看残差图、特征重要性图和学习曲线这三类输出。

## 小结

- 当前流水线把数据准备、单模型训练、测试集预测和三种可视化输出串成了一条完整路径。
- 训练函数负责“得到树模型”，流水线函数负责“组织执行和产出结果”。
- 把这一层执行顺序读清楚，后续看评估与工程实现章节就会更顺。
