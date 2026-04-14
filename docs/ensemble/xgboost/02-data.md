---
title: XGBoost — 数据构成
outline: deep
---

# 数据构成

> 对应代码：`data_generation/ensemble.py`、`data_generation/__init__.py`、`pipelines/ensemble/xgboost.py`
>  
> 相关对象：`EnsembleData.xgboost()`、`xgboost_data`

## 本章目标

1. 明确本仓库 XGBoost 数据来自 `EnsembleData.xgboost()` 的真实数据加载逻辑。
2. 明确 California Housing 数据中的特征列与标签列在流水线中的拆分方式。
3. 明确训练集/测试集切分顺序，以及当前实现没有标准化步骤这一事实。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `EnsembleData.xgboost()` | 方法 | 加载 XGBoost 回归使用的 California Housing 数据 |
| `fetch_california_housing(as_frame=True)` | 函数 | scikit-learn 提供的加州房价真实数据集加载器 |
| `xgboost_data` | 变量 | 在 `data_generation/__init__.py` 中导出的数据对象 |
| `price` | 列名 | 当前流水线中的回归目标列 |

## 1. 本仓库数据入口

- 数据变量：`data_generation/__init__.py` 中导出的 `xgboost_data`
- 生成来源：`data_generation/ensemble.py` 中的 `EnsembleData.xgboost()`
- 流水线使用：`pipelines/ensemble/xgboost.py` 中的 `data = xgboost_data.copy()`

### 理解重点

- `xgboost_data` 在导入时就已经加载完成，因此流水线里直接 `.copy()` 使用即可。
- 使用 `.copy()` 的目的，是避免后续切分或调试时意外修改原始数据对象。

## 2. 数据生成函数 `EnsembleData.xgboost()`

### 参数速览（本节）

适用 API（分项）：

1. `EnsembleData.xgboost()`
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
- 标签列在源码里被统一重命名为 `price`，使不同回归分册的训练代码结构保持一致。
- 这份真实表格数据很适合展示 XGBoost 在结构化回归任务上的优势。

## 3. 特征列与标签列

当前数据表来自 California Housing，包含 8 个原始特征和 1 个标签列。

### 参数速览（本节）

适用列组（本节）：

1. 原始特征列
2. 标签列

| 列组 | 当前内容 | 作用 |
|---|---|---|
| 特征列 | `MedInc`、`HouseAge`、`AveRooms`、`AveBedrms`、`Population`、`AveOccup`、`Latitude`、`Longitude` | 提供收入、住房结构、人口和地理位置等回归信号 |
| 标签列 | `price` | 房价目标值 |

### 示例代码

```python
X = data.drop(columns=["price"])
y = data["price"]
feature_names = list(X.columns)
```

### 理解重点

- 当前流水线先把 `feature_names` 提前保存下来，是为了后续绘制特征重要性图时使用。
- XGBoost 不会像线性回归那样输出一组易解释的线性系数，因此特征重要性和真实列名的对应关系更重要。
- 真实表格数据中的非线性关系和特征交互，是 XGBoost 更擅长捕捉的部分。

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

- 当前 XGBoost 流水线只有训练/测试集切分，没有额外的标准化步骤。
- 这一点必须和 `svr`、`regularization` 等分册区分开，因为它们都显式做了标准化。
- 文档只应描述当前实现真实存在的流程，不能把常见预处理习惯误写成当前代码逻辑。

## 常见坑

1. 把当前数据误认为手工构造数据，忽略它其实来自 California Housing 真实数据集。
2. 看到回归任务就默认写入标准化步骤，但当前源码并没有这样做。
3. 只看到 `X_train` / `X_test`，忽略流水线其实已经提前保存了 `feature_names` 供特征重要性图使用。

## 小结

- 当前 XGBoost 数据来自 `EnsembleData.xgboost()`，底层使用的是 `fetch_california_housing(as_frame=True)`。
- 数据表由 8 个真实房价相关特征和标签列 `price` 组成。
- 读懂数据来源、字段结构和切分方式，是理解后续 boosting 训练、特征重要性和评估图像的前提。
