---
title: 决策树回归 — 数据构成
outline: deep
---

# 数据构成

> 对应代码：`data_generation/regression.py`、`data_generation/__init__.py`、`pipelines/regression/decision_tree.py`
>  
> 相关对象：`RegressionData.decision_tree()`、`decision_tree_regression_data`

## 本章目标

1. 明确本仓库决策树回归数据来自 `RegressionData.decision_tree()` 的真实数据加载逻辑。
2. 明确 California Housing 数据中的特征列与标签列在流水线中的拆分方式。
3. 明确训练集/测试集切分顺序，以及当前实现没有标准化步骤这一事实。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `RegressionData.decision_tree()` | 方法 | 加载决策树回归使用的 California Housing 数据 |
| `fetch_california_housing(as_frame=True)` | 函数 | scikit-learn 提供的加州房价真实数据集加载器 |
| `decision_tree_regression_data` | 变量 | 在 `data_generation/__init__.py` 中导出的数据对象 |
| `price` | 列名 | 当前流水线中的回归目标列 |

## 1. 本仓库数据入口

- 数据变量：`data_generation/__init__.py` 中导出的 `decision_tree_regression_data`
- 生成来源：`data_generation/regression.py` 中的 `RegressionData.decision_tree()`
- 流水线使用：`pipelines/regression/decision_tree.py` 中的 `data = decision_tree_regression_data.copy()`

### 理解重点

- `decision_tree_regression_data` 在导入时就已经加载完成，因此流水线里直接 `.copy()` 使用即可。
- 使用 `.copy()` 的目的，是避免后续切分或调试过程意外修改原始数据对象。

## 2. 数据生成函数 `RegressionData.decision_tree()`

### 参数速览（本节）

适用 API（分项）：

1. `RegressionData.decision_tree()`
2. `fetch_california_housing(as_frame=True)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `as_frame` | `True` | 直接返回带列名的 `DataFrame` |
| 返回值 | `DataFrame` | 含 California Housing 特征与 `price` 的数据表 |
| `n_samples` | 无效 | 当前方法使用真实数据集，不受该属性控制 |

### 示例代码

```python
data = fetch_california_housing(as_frame=True)
df = data.frame.rename(columns={"MedHouseVal": "price"})
return df
```

### 理解重点

- 当前分册不是使用手工合成数据，而是使用 scikit-learn 自带的真实房价数据集。
- 标签列在源码里被统一重命名为 `price`，这样不同回归分册的训练代码结构就更统一。
- 由于是真实数据集，这里不能像线性回归分册那样直接对照一个显式生成公式来解释标签。

## 3. 特征列与标签列

当前数据表来自 California Housing，包含 8 个原始特征和 1 个标签列。

### 参数速览（本节）

适用列组（本节）：

1. 原始特征列
2. 标签列

| 列组 | 当前内容 | 作用 |
|---|---|---|
| 特征列 | `MedInc`、`HouseAge`、`AveRooms`、`AveBedrms`、`Population`、`AveOccup`、`Latitude`、`Longitude` | 提供地理位置、人口与住房结构等回归信号 |
| 标签列 | `price` | 房价目标值 |

### 示例代码

```python
X = data.drop(columns=["price"])
y = data["price"]
feature_names = list(X.columns)
```

### 理解重点

- 当前流水线先把 `feature_names` 提前保存下来，是为了后续绘制特征重要性图时使用。
- 决策树模型不会像线性回归那样输出系数，因此这里更重要的是特征名和重要性之间的对应关系。
- 真实数据中的特征交互更复杂，这也是决策树回归比简单线性模型更适合此数据的一大原因。

## 4. 切分方式与预处理边界

### 参数速览（本节）

适用 API（分项）：

1. `train_test_split(X, y, test_size=0.2, random_state=42)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `test_size` | `0.2` | 测试集占比 |
| `random_state` | `42` | 保证可复现划分 |
| 返回值 | `X_train`、`X_test`、`y_train`、`y_test` | 训练/测试数据拆分结果 |

### 示例代码

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 理解重点

- 当前决策树回归流水线只有训练/测试集切分，没有额外的标准化步骤。
- 这和 `svr`、`regularization` 分册不同，文档中必须如实写清楚。
- 需要注意的是，“没有标准化”是当前工程实现事实，并不等于所有回归任务都不需要预处理。

## 5. 为什么训练阶段显式使用 `.values`

### 参数速览（本节）

适用代码（分项）：

1. `train_model(X_train.values, y_train.values)`
2. `model.predict(X_test.values)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_train.values` | `ndarray` | 传给训练函数的训练特征数组 |
| `y_train.values` | `ndarray` | 传给训练函数的训练标签数组 |
| `X_test.values` | `ndarray` | 传给预测函数的测试特征数组 |

### 理解重点

- 当前流水线在训练和预测时显式把 `DataFrame` / `Series` 转成了 NumPy 数组。
- 这不是决策树算法本身的硬性要求，而是当前实现选择的一种输入形式。
- 因为后续还要单独保存 `feature_names`，所以即使转成数组，也不会丢失特征名信息。

## 常见坑

1. 把当前数据误认为手工构造数据，忽略它其实来自 California Housing 真实数据集。
2. 看到回归任务就默认写入标准化步骤，但当前源码并没有这样做。
3. 只关注 `X_train.values`，忽略流水线其实已经提前保存了 `feature_names` 用于特征重要性图。

## 小结

- 当前决策树回归数据来自 `RegressionData.decision_tree()`，底层使用的是 `fetch_california_housing(as_frame=True)`。
- 数据表由 8 个真实房价相关特征和标签列 `price` 组成。
- 读懂数据来源、字段结构和输入形式，是理解后续模型分裂、特征重要性和评估图像的前提。
