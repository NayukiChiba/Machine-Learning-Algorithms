---
title: 线性回归 — 训练与预测
outline: deep
---

# 训练与预测

> 对应代码：`pipelines/regression/linear_regression.py`、`model_training/regression/linear_regression.py`
>  
> 运行方式：`python -m pipelines.regression.linear_regression`

## 本章目标

1. 明确当前流水线从取数到生成图像的完整执行顺序。
2. 理解训练阶段、预测阶段、残差图和学习曲线分别由哪个函数负责。
3. 明确当前线性回归实现没有标准化步骤。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `run()` | 函数 | 线性回归端到端流水线入口 |
| `train_test_split(...)` | 函数 | 拆分训练集与测试集 |
| `train_model(...)` | 函数 | 训练线性回归模型 |
| `model.predict(X_test)` | 方法 | 对测试集做回归预测 |
| `plot_residuals(...)` | 函数 | 绘制残差分析图 |
| `plot_learning_curve(...)` | 函数 | 绘制学习曲线 |

## 1. 端到端入口 `run()`

### 参数速览（本节）

适用函数：`run()`

| 项目 | 当前实现 |
|---|---|
| 数据源 | `linear_regression_data.copy()` |
| 标签列 | `price` |
| 切分方式 | `test_size=0.2, random_state=42` |
| 训练入口 | `train_model(X_train, y_train)` |
| 预测入口 | `model.predict(X_test)` |
| 可视化入口 | `plot_residuals(...)`、`plot_learning_curve(...)` |

### 示例代码

```python
def run():
    data = linear_regression_data.copy()
    X = data.drop(columns=["price"])
    y = data["price"]
```

### 理解重点

- 整个分册的运行入口就是 `pipelines/regression/linear_regression.py` 里的 `run()`。
- 这个函数不负责实现线性回归求解本身，而是把取数、训练、预测和画图串成一条流程。
- 相比更复杂的流水线，这里的执行链非常短，适合初学时逐步跟读。

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

- 当前流水线在数据切分后直接开始训练，没有额外的标准化步骤。
- 这一点和 `svr`、`regularization` 分册不同，文档里必须明确区分。
- 因为当前数据关系简单、维度低且量纲相对直观，所以这里保留了最基础的线性回归流程。

## 3. 训练阶段：调用 `train_model(...)`

### 参数速览（本节）

适用函数：`train_model(X_train, y_train)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_train` | 训练特征 | 当前直接传入原始训练特征 |
| `y_train` | 训练标签 | 连续值目标 |
| 返回值 | `model` | 已训练好的 `LinearRegression` 模型 |

### 示例代码

```python
model = train_model(X_train, y_train)
```

### 理解重点

- 当前实现没有把训练和预测揉成同一个函数，而是先得到训练好的模型，再单独调用 `predict(...)`。
- 训练阶段最重要的副产物，不只是 `model` 对象，还有控制台里打印出的截距和系数。
- 因为数据关系透明，这些训练日志本身就具有很强解释价值。

## 4. 预测阶段：直接调用 `predict(...)`

### 参数速览（本节）

适用流程（分项）：

1. `y_pred = model.predict(X_test)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `model` | 已训练完成模型 | 来自 `train_model(...)` 返回值 |
| `X_test` | 测试特征 | 当前直接传入未标准化测试集 |
| `y_pred` | 预测值数组 | 用于残差图分析 |

### 示例代码

```python
y_pred = model.predict(X_test)
```

### 理解重点

- 当前仓库没有额外封装 `predict_model(...)`，而是直接使用 scikit-learn 统一的 `predict(...)` 接口。
- 由于本分册没有标准化步骤，所以预测阶段也直接使用原始测试特征。
- 这让整条流程更容易对应到“训练一个线性函数，然后代入新样本求值”的直觉。

## 5. 预测后的残差图输出

### 参数速览（本节）

适用函数：`plot_residuals(y_test, y_pred, title=..., dataset_name=..., model_name=...)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `y_test` | 测试标签 | 真实值 |
| `y_pred` | 预测值 | 模型对测试集的输出 |
| `title` | `"线性回归 残差分析"` | 图标题 |
| `dataset_name` | `"linear_regression"` | 输出目录名 |
| `model_name` | `"linear_regression"` | 输出文件名前缀 |

### 示例代码

```python
plot_residuals(
    y_test,
    y_pred,
    title="线性回归 残差分析",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 残差图用来检查预测值和真实值之间的误差分布。
- `DATASET` 和 `MODEL` 都是固定字符串，因此当前线性回归会把图输出到稳定的路径下。
- 这一步是测试集预测之后才发生的，不属于训练过程本身。

## 6. 学习曲线是如何接入流水线的

### 参数速览（本节）

适用函数：`plot_learning_curve(LinearRegression(), X_train, y_train, scoring='r2', ...)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `model` | `LinearRegression()` | 传入一个新的未训练模型实例 |
| `X` | `X_train` | 使用训练集画学习曲线 |
| `y` | `y_train` | 使用训练标签画学习曲线 |
| `scoring` | `"r2"` | 当前分册使用的评分指标 |

### 示例代码

```python
plot_learning_curve(
    LinearRegression(),
    X_train,
    y_train,
    scoring="r2",
    title="线性回归 学习曲线",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 学习曲线不是基于训练好的 `model` 直接画出来的，而是由 `plot_learning_curve(...)` 内部重新做不同训练规模下的交叉验证。
- 当前函数输入的是 `X_train` 和 `y_train`，不是测试集。
- 这一步的作用，是观察样本量变化时训练得分和验证得分如何变化，而不是直接给出一次测试预测结果。

## 7. 用伪代码看完整流程

### 示例代码

```python
data = linear_regression_data.copy()
X = data.drop(columns=["price"])
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(...)

model = train_model(X_train, y_train)
y_pred = model.predict(X_test)

plot_residuals(...)
plot_learning_curve(LinearRegression(), X_train, y_train, scoring="r2", ...)
```

### 理解重点

- 当前线性回归流水线的主线非常清楚：取数、切分、训练、预测、画残差图、画学习曲线。
- 这条链路里最关键的中间变量是 `model` 和 `y_pred`。
- 只要把这条流程走清楚，整个 linear_regression 分册的工程部分就基本读懂了。

## 常见坑

1. 把其他分册里的标准化流程误套到当前线性回归实现上。
2. 误以为学习曲线是用测试集直接画的，实际上它基于训练集和交叉验证。
3. 只看到模型训练成功，没有继续看残差图和学习曲线这两类结果输出。

## 小结

- 当前流水线把数据准备、单模型训练、测试集预测和两种可视化输出串成了一条完整路径。
- 训练函数负责“得到模型”，流水线函数负责“组织执行和产出结果”。
- 把这一层执行顺序读清楚，后续看评估与工程实现章节就会更顺。
