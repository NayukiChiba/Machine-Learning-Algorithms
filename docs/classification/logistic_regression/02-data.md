---
title: LogisticRegression 逻辑回归分类 — 数据构成
outline: deep
---

# 数据构成

> 对应代码：`data_generation/classification.py`、`data_generation/__init__.py`、`pipelines/classification/logistic_regression.py`
>
> 相关对象：`ClassificationData.logistic_regression()`、`logistic_regression_data`

## 本章目标

1. 明确本仓库 Logistic Regression 数据来自 `ClassificationData.logistic_regression()` 的生成逻辑。
2. 明确特征列与标签列在当前流水线中的拆分方式。
3. 明确训练集/测试集切分与标准化的顺序和边界。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `ClassificationData.logistic_regression()` | 方法 | 生成逻辑回归使用的高维二分类数据 |
| `make_classification(...)` | 函数 | scikit-learn 提供的监督分类数据生成器 |
| `logistic_regression_data` | 变量 | 在 `data_generation/__init__.py` 中导出的数据对象 |
| `label` | 列名 | 当前流水线中的监督分类标签 |
| `StandardScaler` | 类 | 对特征做标准化，改善训练与系数解释 |

## 1. 本仓库数据入口

- 数据变量：`data_generation/__init__.py` 中导出的 `logistic_regression_data`
- 生成来源：`data_generation/classification.py` 中的 `ClassificationData.logistic_regression()`
- 流水线使用：`pipelines/classification/logistic_regression.py` 中的 `data = logistic_regression_data.copy()`

### 理解重点

- `logistic_regression_data` 在导入时就已经生成完成，因此流水线里直接 `.copy()` 使用即可。
- 用 `.copy()` 的目的，是避免后续处理意外修改原始数据对象。
- 当前数据是为逻辑回归教学场景专门构造的，因此和线性分类边界假设比较匹配。

## 2. 数据生成函数 `ClassificationData.logistic_regression()`

### 参数速览（本节）

适用 API（分项）：

1. `ClassificationData.logistic_regression()`
2. `make_classification(...)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n_samples` | `400` | 样本数 |
| `n_features` | `6` | 总特征数 |
| `n_informative` | `3` | 有效特征数 |
| `n_redundant` | `1` | 冗余特征数 |
| `n_repeated` | `0` | 重复特征数 |
| `n_classes` | `2` | 二分类任务 |
| `class_sep` | `1.2` | 类别间分离程度 |
| `flip_y` | `0.03` | 标签噪声比例 |
| `random_state` | `42` | 随机种子，保证可复现 |
| 返回值 | `DataFrame` | 含 `x1` ~ `x6` 与 `label` 的数据表 |

### 示例代码

```python
X, y = make_classification(
    n_samples=self.n_samples,
    n_features=self.lr_n_feature,
    n_informative=self.lr_n_informative,
    n_redundant=self.lr_n_redundant,
    n_repeated=self.lr_n_repeated,
    n_classes=self.lr_n_classes,
    weights=self.lr_weights,
    class_sep=self.lr_class_sep,
    flip_y=self.lr_flip_y,
    random_state=self.random_state,
)
columns = [f"x{i + 1}" for i in range(self.lr_n_feature)]
data = DataFrame(X, columns=columns)
data["label"] = y
```

### 理解重点

- 当前数据是高维二分类数据，不是二维玩具边界问题。
- 它包含有效特征、冗余特征和少量标签噪声，适合展示逻辑回归在“近线性可分但不完美”场景下的行为。
- 这和 SVC 同心圆数据、Naive Bayes iris 数据的教学目的都不同。

## 3. 特征列与标签列

当前数据表结构如下：

- 特征列：`x1` ~ `x6`
- 标签列：`label`

### 示例代码

```python
X = data.drop(columns=["label"])
y = data["label"]
```

### 理解重点

- `label` 是监督训练标签，会真实参与 `model.fit(X_train, y_train)`。
- 当前流水线把特征和标签明确拆开，这是后续切分、标准化和训练的前提。
- 与聚类分册不同，这里标签不是只用于对照，而是训练过程的一部分。

## 4. 切分与标准化

### 参数速览（本节）

适用 API（分项）：

1. `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`
2. `StandardScaler().fit_transform(X_train)`
3. `StandardScaler().transform(X_test)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `test_size` | `0.2` | 测试集占比 |
| `random_state` | `42` | 保证可复现划分 |
| `stratify` | `y` | 保持训练集和测试集的类别比例一致 |
| `X_train` | 训练特征 | 只在训练集上拟合标准化统计量 |
| `X_test` | 测试特征 | 使用训练集统计量做变换 |

### 示例代码

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

### 理解重点

- 标准化必须发生在切分之后，否则会造成数据泄露。
- 当前流水线显式使用 `stratify=y`，说明作者希望训练集和测试集在类别比例上保持稳定。
- 对逻辑回归来说，标准化不仅有利于优化器稳定收敛，也有利于后续系数解释与 PCA 可视化。

## 常见坑

1. 忘记把 `label` 从特征表中剥离出来。
2. 在切分之前就对全量数据做标准化。
3. 忽略 `stratify=y`，导致训练集和测试集类别比例不稳定。
4. 只看到“逻辑回归是线性模型”，却忽略当前数据中仍然包含噪声与冗余特征。

## 小结

- 当前 Logistic Regression 数据来自 `ClassificationData.logistic_regression()`，底层使用的是 `make_classification(...)`。
- 数据表结构清晰：`x1` ~ `x6` 是特征，`label` 是监督分类标签。
- 读懂数据来源、切分方式和标准化顺序，是理解后续训练与评估章节的前提。
