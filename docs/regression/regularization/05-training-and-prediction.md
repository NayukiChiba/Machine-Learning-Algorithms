---
title: 正则化回归 — 训练与预测
outline: deep
---

# 训练与预测

> 对应代码：`pipelines/regression/regularization.py`、`model_training/regression/regularization.py`
>  
> 运行方式：`python -m pipelines.regression.regularization`

## 本章目标

1. 明确当前流水线从取数到生成残差图的完整执行顺序。
2. 理解训练阶段和预测阶段分别由哪个文件、哪个函数负责。
3. 理解三种模型如何在同一套预处理结果上被统一比较。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `run()` | 函数 | 正则化回归端到端流水线入口 |
| `train_test_split(...)` | 函数 | 拆分训练集与测试集 |
| `StandardScaler` | 类 | 对训练和测试特征做一致的标准化 |
| `train_model(...)` | 函数 | 一次训练三种正则化模型 |
| `model.predict(X_test_s)` | 方法 | 对测试集做回归预测 |
| `plot_residuals(...)` | 函数 | 为每个模型绘制残差图 |

## 1. 端到端入口 `run()`

### 参数速览（本节）

适用函数：`run()`

| 项目 | 当前实现 |
|---|---|
| 数据源 | `regularization_data.copy()` |
| 标签列 | `price` |
| 切分方式 | `test_size=0.2, random_state=42` |
| 训练入口 | `train_model(X_train_s, y_train, feature_names=feature_names)` |
| 预测入口 | `model.predict(X_test_s)` |
| 可视化入口 | `plot_residuals(...)` |

### 示例代码

```python
def run():
    data = regularization_data.copy()
    X = data.drop(columns=["price"])
    y = data["price"]
    feature_names = list(X.columns)
```

### 理解重点

- 整个分册的运行入口就是 `pipelines/regression/regularization.py` 里的 `run()`。
- 这个函数不负责实现模型本身，而是把数据、预处理、训练、预测、画图串成一条完整链路。
- `feature_names` 会从这里一路传到训练函数，用于后续系数打印。

## 2. 训练前的数据准备顺序

### 参数速览（本节）

适用 API（分项）：

1. `train_test_split(X, y, test_size=0.2, random_state=42)`
2. `StandardScaler().fit_transform(X_train)`
3. `StandardScaler().transform(X_test)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `test_size` | `0.2` | 测试集占比 |
| `random_state` | `42` | 保证可复现划分 |
| `X_train_s` | 标准化训练特征 | 供三种模型共同训练 |
| `X_test_s` | 标准化测试特征 | 供三种模型共同预测 |

### 示例代码

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

### 理解重点

- 三种模型使用完全相同的训练集、测试集和标准化结果，这样对比才公平。
- 这里的 `scaler` 不会被打包进单独的预测函数里，而是直接在流水线中显式使用。
- 正则化模型对特征量纲很敏感，因此标准化是训练前的关键步骤。

## 3. 训练阶段：一次拿到三种模型

### 参数速览（本节）

适用函数：`train_model(X_train_s, y_train, feature_names=feature_names)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_train_s` | 标准化后的训练特征 | 三模型共享输入 |
| `y_train` | 训练标签 | 三模型共享目标 |
| `feature_names` | `list(X.columns)` | 用于打印每个特征的系数 |
| 返回值 | `models` | `{"Lasso": ..., "Ridge": ..., "ElasticNet": ...}` |

### 示例代码

```python
models = train_model(X_train_s, y_train, feature_names=feature_names)
```

### 理解重点

- 当前实现没有把三种模型分成三套完全独立的流水线，而是先统一训练，再统一评估。
- 这使得文档可以围绕“同一数据、不同正则化策略”的比较来展开。
- 训练阶段最重要的副产物，不只是 `models` 字典，还有控制台里打印出来的系数信息。

## 4. 预测阶段：直接调用每个模型的 `predict(...)`

### 参数速览（本节）

适用流程（分项）：

1. `for name, model in models.items()`
2. `y_pred = model.predict(X_test_s)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `name` | `Lasso` / `Ridge` / `ElasticNet` | 当前正在评估的模型名 |
| `model` | 已训练完成模型 | 从 `models` 字典中取出 |
| `X_test_s` | 标准化后的测试特征 | 与训练时保持同分布和同预处理 |
| `y_pred` | 预测值数组 | 用于残差分析 |

### 示例代码

```python
for name, model in models.items():
    y_pred = model.predict(X_test_s)
```

### 理解重点

- 当前仓库没有额外封装 `predict_model(...)`，而是直接使用 scikit-learn 统一的 `predict(...)` 接口。
- 这种写法很简单，但前提是你必须保证传入的还是训练时同样标准化过的特征。
- 一旦跳过 `scaler.transform(X_test)`，预测结果会失去可比性，甚至明显变差。

## 5. 预测后的残差图输出

### 参数速览（本节）

适用函数：`plot_residuals(y_test, y_pred, title=..., dataset_name=..., model_name=...)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `y_test` | 测试标签 | 真实值 |
| `y_pred` | 当前模型预测值 | 用于与真实值对比 |
| `title` | `f"{name} 残差分析"` | 图标题 |
| `dataset_name` | `"regularization"` | 结果保存目录名 |
| `model_name` | `name.lower()` | 输出文件名前缀 |

### 示例代码

```python
plot_residuals(
    y_test,
    y_pred,
    title=f"{name} 残差分析",
    dataset_name=DATASET,
    model_name=name.lower(),
)
```

### 理解重点

- 残差图是在预测之后逐模型生成的，所以每个模型都会单独得到一份结果图。
- `DATASET = "regularization"` 会把图片统一保存到正则化分册对应的目录下。
- `name.lower()` 则把模型名映射到更稳定的文件名前缀，例如 `lasso_residual.png`。

## 6. 用伪代码看完整流程

### 示例代码

```python
data = regularization_data.copy()
X = data.drop(columns=["price"])
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(...)
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = train_model(X_train_s, y_train, feature_names=feature_names)

for name, model in models.items():
    y_pred = model.predict(X_test_s)
    plot_residuals(...)
```

### 理解重点

- 训练和预测的核心思路并不复杂，重点在于顺序不能错。
- 先切分、后标准化、再训练、再预测、最后画图，这是当前实现的固定主线。
- 文档后面提到的所有诊断结论，都是建立在这条执行顺序正确的前提下。

## 常见坑

1. 训练阶段使用 `X_train_s`，预测阶段却误把未标准化的 `X_test` 传给 `predict(...)`。
2. 误以为每个模型有独立训练入口，实际三者都由同一个 `train_model(...)` 返回。
3. 只运行到训练成功为止，没有继续观察每个模型生成的残差图。

## 小结

- 当前流水线把数据准备、三模型训练、统一预测和残差图输出串成了一条完整路径。
- 训练函数负责“生成模型”，流水线函数负责“组织比较”。
- 把这一层执行顺序读清楚，后续看评估与工程实现章节就会更顺。
