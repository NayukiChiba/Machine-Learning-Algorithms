---
title: GaussianNB 高斯朴素贝叶斯 — 评估与诊断
outline: deep
---

# 评估与诊断

> 对应代码：`pipelines/classification/naive_bayes.py`、`result_visualization/confusion_matrix.py`、`result_visualization/roc_curve.py`、`result_visualization/decision_boundary.py`、`result_visualization/learning_curve.py`
>
> 相关对象：`y_pred`、`y_scores`、`plot_confusion_matrix(...)`、`plot_roc_curve(...)`、`plot_decision_boundary(...)`、`plot_learning_curve(...)`

## 本章目标

1. 明确当前仓库 Naive Bayes 实现实际上是如何做结果诊断的。
2. 理解混淆矩阵、ROC 曲线、PCA 决策边界图和学习曲线分别能说明什么。
3. 理解多分类 ROC 与二维决策边界图的展示边界。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `y_pred` | 预测结果 | 测试集类别输出 |
| `y_scores` | 预测概率 | 测试集各类别概率输出 |
| `plot_confusion_matrix(...)` | 函数 | 绘制预测标签与真实标签的混淆矩阵 |
| `plot_roc_curve(...)` | 函数 | 绘制二分类或多分类 One-vs-Rest ROC 曲线 |
| `plot_decision_boundary(...)` | 函数 | 绘制 PCA 2D 空间下的分类边界 |
| `plot_learning_curve(...)` | 函数 | 绘制训练/验证得分随样本量变化的曲线 |

## 1. 当前仓库的评估入口

当前 Naive Bayes 流水线里的主要结果诊断手段有四个：

1. 混淆矩阵
2. ROC 曲线
3. PCA 2D 决策边界图
4. 学习曲线

### 示例代码

```python
y_pred = model.predict(X_test_s)
y_scores = model.predict_proba(X_test_s)

plot_confusion_matrix(...)
plot_roc_curve(...)
plot_decision_boundary(...)
plot_learning_curve(...)
```

### 理解重点

- 当前实现没有把所有诊断都压缩成一个数字，而是同时提供结果矩阵、概率曲线、边界图和曲线图四类视角。
- 对教学型仓库来说，这样的设计比只打印单个分数更利于理解模型行为。
- 四种可视化分别回答的是不同问题，不能互相替代。

## 2. 混淆矩阵能观察什么

### 参数速览（本节）

适用函数：`plot_confusion_matrix(y_true, y_pred, ...)`

| 参数名 | 当前对象 | 说明 |
|---|---|---|
| `y_true` | `y_test` | 测试集真实标签 |
| `y_pred` | `model.predict(X_test_s)` | 测试集预测标签 |

### 理解重点

- 混淆矩阵最适合回答：模型把哪一类分对了，哪一类更容易混淆。
- 对 iris 三分类任务来说，它能直观看到不同类别之间的误分类方向。
- 当前流水线没有显式打印 accuracy，但混淆矩阵已经能给出很强的误差结构信息。

## 3. ROC 曲线能观察什么

### 参数速览（本节）

适用函数：`plot_roc_curve(y_test, y_scores, ...)`

| 参数名 | 当前对象 | 说明 |
|---|---|---|
| `y_true` | `y_test` | 测试集真实标签 |
| `y_scores` | `model.predict_proba(X_test_s)` | 测试集各类别预测概率 |

### 理解重点

- 当前 ROC 曲线模块支持二分类和多分类 One-vs-Rest。
- 对 iris 三分类任务来说，它会把每个类别都当作“正类 vs 其余类别”分别计算一条 ROC 曲线。
- 这也是为什么当前分册里必须强调 `predict_proba(...)` 的作用，因为没有概率输出就无法形成 ROC 曲线。

## 4. PCA 2D 决策边界图能观察什么

### 参数速览（本节）

适用函数：`plot_decision_boundary(model_2d, X_2d, y.values, ...)`

| 参数名 | 当前对象 | 说明 |
|---|---|---|
| `model_2d` | 单独训练的二维 GaussianNB | 用于可视化边界 |
| `X_2d` | `PCA` 降到二维后的特征 | 用于画图 |
| `y.values` | 全量标签数组 | 用于点着色 |

### 理解重点

- 这张图最适合回答：当前概率分类模型在二维投影视角下，形成了怎样的类别区域划分。
- 它能帮助读者直观理解 GaussianNB 在低维可视化空间中是如何划分类别的。
- 但它只是 PCA 投影空间中的近似展示，不是原始高维特征空间分类区域的完整真相。

## 5. 学习曲线能观察什么

### 参数速览（本节）

适用函数：`plot_learning_curve(model, X_train_s, y_train, scoring='accuracy', ...)`

| 参数名 | 当前对象 | 说明 |
|---|---|---|
| `model` | `GaussianNB()` | 用于曲线诊断的新模型实例 |
| `X_train_s` | 标准化后的训练特征 | 学习曲线输入特征 |
| `y_train` | 训练标签 | 学习曲线输入标签 |
| `scoring` | `accuracy` | 当前评分类指标 |

### 理解重点

- 学习曲线最适合回答：当前模型是否随着训练样本数增加而继续改善。
- 训练得分与验证得分之间的距离，也能帮助判断是否存在欠拟合或过拟合倾向。
- 当前实现采用 `accuracy` 作为学习曲线评分指标，但这不等于对模型质量的全部评价。

## 6. 当前实现中尚未纳入但常见的分类指标

在一般分类任务中，还常见以下指标：

- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1 分数

### 理解重点

- 当前仓库并没有在 Naive Bayes 流水线中显式打印这些指标。
- 文档可以提到它们是常见扩展方向，但不能写成“当前源码已经在计算”。
- 现阶段最准确的表述是：当前实现以混淆矩阵、ROC 曲线、决策边界图和学习曲线为主要诊断手段。

## 评估图表

![混淆矩阵](../../../outputs/naive_bayes/confusion_matrix.png)

![ROC 曲线](../../../outputs/naive_bayes/roc_curve.png)

## 常见坑

1. 把 `predict(...)` 和 `predict_proba(...)` 的用途混为一谈。
2. 把多分类 ROC 曲线误解成只有一条全局曲线。
3. 把 PCA 决策边界图误认为原始特征空间决策面的完整表达。
4. 把当前仓库未实现的 accuracy、precision、recall、f1 写成现有流程的一部分。

## 小结

- 当前仓库对 Naive Bayes 的评估方式很明确：结果矩阵上看混淆矩阵，概率区分能力上看 ROC 曲线，边界形状上看 PCA 决策边界图，训练行为上看学习曲线。
- 四者组合起来，比单一指标更能解释当前 `GaussianNB` 的实际表现。
- 对当前 iris 教学数据而言，这样的评估设计兼顾了直观性与工程可读性。
