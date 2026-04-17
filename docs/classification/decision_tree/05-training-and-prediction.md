---
title: DecisionTreeClassifier 决策树分类 — 训练与预测
outline: deep
---

# 训练与预测

> 对应代码：`pipelines/classification/decision_tree.py`、`model_training/classification/decision_tree.py`
>
> 运行方式：`python -m pipelines.classification.decision_tree`

## 本章目标

1. 按源码顺序看清当前 Decision Tree 流水线到底执行了哪些步骤。
2. 理解训练集/测试集拆分、训练、类别预测和概率预测之间的连接关系。
3. 理解主模型与二维可视化模型在当前实现中的职责差异。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `decision_tree_classification_data.copy()` | 方法 | 复制原始数据，避免修改源对象 |
| `train_test_split(...)` | 方法 | 划分训练集与测试集 |
| `train_model(X_train.values, y_train.values)` | 函数 | 训练主分类树模型 |
| `model.predict(X_test.values)` | 方法 | 生成测试集类别预测结果 |
| `model.predict_proba(X_test.values)` | 方法 | 生成测试集类别概率输出 |
| `PCA(n_components=2)` | 类 | 为决策边界可视化构造二维表示 |
| `model_2d` | 模型 | 专门用于二维决策边界展示 |

## 1. 流水线从复制数据开始

### 示例代码

```python
data = decision_tree_classification_data.copy()
X = data.drop(columns=["label"])
y = data["label"]
feature_names = list(X.columns)
```

### 理解重点

- 当前流水线先复制 `decision_tree_classification_data`，再拆出 `X` 和 `y`。
- `feature_names` 会在后续特征重要性图中使用，因此流水线较早就把它保存下来。
- 当前任务是监督多分类，因此 `y` 会真实参与训练和预测评估。

## 2. 先切分训练集与测试集

### 示例代码

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 理解重点

- 当前流水线明确区分了训练阶段和测试阶段。
- `stratify=y` 的作用，是让训练集和测试集保持相近的类别比例。
- 对当前 4 分类任务来说，这是很常见也很必要的工程细节。

## 3. 主模型训练与正式预测

当前决策树主流程没有显式标准化步骤，而是直接把原始数值特征传入模型。

### 示例代码

```python
model = train_model(X_train.values, y_train.values)
y_pred = model.predict(X_test.values)
```

### 理解重点

- `model` 是当前分册的主模型，用于正式训练和测试集类别预测。
- 决策树通过阈值切分特征空间，因此不像 KNN、SVC 那样强依赖标准化。
- `y_pred` 是后续混淆矩阵评估的直接输入。

## 4. 概率输出如何进入流水线

### 示例代码

```python
y_scores = model.predict_proba(X_test.values)
```

### 理解重点

- `predict_proba(...)` 会给出每个测试样本在各个类别上的概率估计。
- 这些概率输出不是可有可无的附带信息，而是 ROC 曲线可视化的直接输入。
- 当前任务是多分类，因此后续 ROC 模块会按 One-vs-Rest 方式处理这些概率。

## 5. 特征重要性如何进入流水线

### 示例代码

```python
plot_feature_importance(
    model,
    feature_names=feature_names,
    title="决策树 特征重要性",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 这一步是当前决策树分册非常重要的补充，因为树模型天然具备特征重要性解释能力。
- `feature_names` 与 `feature_importances_` 的组合，可以把抽象的树分裂信息转成更直观的解释图。
- 这也是当前分册和很多其他分类分册在评估内容上的一个明显差异点。

## 6. 决策边界为什么要额外训练一个 `model_2d`

当前流水线里还有这样一段逻辑：

### 示例代码

```python
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X.values)
model_2d = DecisionTreeClassifier(max_depth=6, random_state=42)
model_2d.fit(pca.transform(X_train.values), y_train.values)
```

### 理解重点

- 这里的 `model_2d` 不是主评估模型，而是专门为二维可视化服务的辅助模型。
- 主模型训练在原始特征空间中，而决策边界图需要二维输入。
- 因此当前实现采用 `PCA` 把特征投影到二维空间，再单独训练一个二维决策树模型用来画图。
- 这是整个决策树分册里最需要讲清的工程细节之一。

## 7. 学习曲线如何接入流水线

### 示例代码

```python
plot_learning_curve(
    DecisionTreeClassifier(max_depth=6, random_state=42),
    X_train.values,
    y_train.values,
    title="决策树 学习曲线",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 学习曲线使用的是一个新的 `DecisionTreeClassifier(...)` 实例，而不是直接复用 `model`。
- 这是因为学习曲线函数内部会自行克隆和重复训练模型。
- 当前文档需要把“主模型用于正式预测”和“新模型实例用于曲线诊断”区分清楚。

## 常见坑

1. 把 `predict(...)` 和 `predict_proba(...)` 混为一谈。
2. 把特征重要性图看成与训练主流程无关的附加内容。
3. 把 `model_2d` 误认为正式预测模型本体。
4. 混淆主模型预测、二维可视化模型和学习曲线模型三者的职责。

## 小结

- 当前 Decision Tree 流水线的训练过程非常清晰：复制数据、切分、训练主模型、测试集类别预测、概率预测、特征重要性分析、再做多种可视化诊断。
- 对本仓库而言，`model`、`model_2d` 和学习曲线中的新模型实例分别承担不同职责。
- 把这条链路看清楚后，再读评估与工程实现章节会更容易建立全局理解。
