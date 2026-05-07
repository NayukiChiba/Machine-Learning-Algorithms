---
title: DecisionTreeClassifier 决策树分类 — 工程实现
outline: deep
---

# 工程实现


## 本章目标

1. 从工程角度看清 Decision Tree 在本仓库中的完整调用链。
2. 理解数据生成、模型训练、流水线编排和结果可视化分别负责什么。
3. 理解为什么当前实现要把训练逻辑、正式预测逻辑、概率输出逻辑和可视化逻辑拆开。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/classification.py` | `ClassificationData.decision_tree()` 生成二维多分类 blob 数据 |
| 数据导出 | `data_generation/__init__.py` | 向外暴露 `decision_tree_classification_data` |
| 训练封装 | `model_training/classification/decision_tree.py` | 构建并训练 `DecisionTreeClassifier`，打印训练日志 |
| 流水线入口 | `pipelines/classification/decision_tree.py` | 组织切分、训练、预测与可视化评估的完整编排 |
| 混淆矩阵可视化 | `result_visualization/confusion_matrix.py` | 保存混淆矩阵图 |
| ROC 曲线可视化 | `result_visualization/roc_curve.py` | 保存多分类 One-vs-Rest ROC 曲线图 |
| 特征重要性可视化 | `result_visualization/feature_importance.py` | 保存特征重要性图 |
| 决策边界可视化 | `result_visualization/decision_boundary.py` | 保存 PCA 二维决策边界图 |
| 学习曲线可视化 | `result_visualization/learning_curve.py` | 保存学习曲线图 |

## 1. 端到端运行入口

### 示例代码

```bash
python -m pipelines.classification.decision_tree
```

### 理解重点

- 对大多数读者来说，这个命令是理解当前 Decision Tree 工程实现的最佳入口。
- 它会依次完成数据读取、特征准备、模型训练、测试集预测、概率输出和结果绘图。
- 如果只读一个文件，建议先读 `pipelines/classification/decision_tree.py`——它是整个决策树流程的编排层。

## 2. run() 串起了整个流程

当前流水线的核心函数 `run()` 采用线性编排风格，每一步的输入明确来自上一步的输出。

### 核心逻辑

```python
def run():
    # 1. 复制数据 & 拆出特征/标签
    data = decision_tree_classification_data.copy()
    X = data.drop(columns=["label"])
    y = data["label"]
    feature_names = list(X.columns)

    # 2. 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. 训练主模型 & 正式预测
    model = train_model(X_train.values, y_train.values)
    y_pred = model.predict(X_test.values)
    y_scores = model.predict_proba(X_test.values)

    # 4. 特征重要性
    plot_feature_importance(model, feature_names=feature_names, ...)

    # 5. 可视化诊断（混淆矩阵、ROC、决策边界、学习曲线）
    plot_confusion_matrix(y_test, y_pred, ...)
    plot_roc_curve(y_test, y_scores, ...)
    plot_decision_boundary(model_2d, X_2d, y.values, ...)
    plot_learning_curve(DecisionTreeClassifier(...), X_train.values, y_train.values, ...)
```

### 理解重点

- `run()` 本身没有复杂算法，它的职责是把不同模块串起来。
- 这类文件更像"编排层"（orchestrator），重点是流程顺序正确、调用关系清楚。
- 每一步的数据流向都是单向的：上一步的输出直接作为下一步的输入，没有循环依赖。

## 3. 训练模块负责什么

`model_training/classification/decision_tree.py` 里的 `train_model(...)` 是训练逻辑的封装层。

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like`，形状 `(n_samples, n_features)` | 训练特征矩阵。将原样传入 `DecisionTreeClassifier.fit()` | `X_train.values` |
| `y_train` | `array_like`，形状 `(n_samples,)` | 训练标签向量，取值 $\{0, 1, \dots, K-1\}$ | `y_train.values` |
| `max_depth` | `int` | 树的最大深度 $d_{\max}$，限制划分轮数。默认 `6` | `3`、`6`、`None` |
| `min_samples_split` | `int` | 内部节点继续分裂所需最小样本数。默认 `4` | `2`、`4`、`10` |
| `min_samples_leaf` | `int` | 叶节点最少样本数。默认 `2` | `1`、`2`、`5` |
| `criterion` | `str` | 不纯度度量：`"gini"` = $1 - \sum p_k^2$，`"entropy"` = $-\sum p_k \log_2 p_k$。默认 `"gini"` | `"gini"`、`"entropy"` |
| `random_state` | `int` | 随机种子，保证分裂中的随机性可复现。默认 `None` | `42` |
| 返回值 | `DecisionTreeClassifier` | 已训练完成的主模型对象，含 `classes_`、`feature_importances_`、`tree_` 等属性 | — |

### 职责清单

`train_model(...)` 主要负责四件事：

1. 创建 `DecisionTreeClassifier(...)` 实例（按给定超参数）
2. 调用 `model.fit(X_train, y_train)` 执行 CART 递归划分
3. 打印训练日志：训练耗时、实际深度、叶节点数、criterion
4. 返回训练完成的主模型对象

### 理解重点

- 这层抽离让"模型训练逻辑"和"业务流程编排逻辑"分开。
- 训练函数既可以被流水线调用，也可以单独运行做局部验证（模块级 `if __name__ == "__main__"` 自测）。
- 这也是当前仓库多个算法分册共享的组织方式：`model_training/classification/*.py` 各封装一个算法的训练逻辑。

## 4. 五类评估模块分别负责什么

当前 Decision Tree 的五种可视化各由一个独立模块负责，每个模块职责单一。

### 模块职责速览

| 模块 | 函数 | 输入 | 输出 |
|---|---|---|---|
| 混淆矩阵 | `plot_confusion_matrix(...)` | `y_test`、`y_pred` | 混淆矩阵图片（PNG） |
| ROC 曲线 | `plot_roc_curve(...)` | `y_test`、`y_scores` | 多分类 ROC 曲线图片（PNG） |
| 特征重要性 | `plot_feature_importance(...)` | `model`、`feature_names` | 特征重要性条状图（PNG） |
| 决策边界 | `plot_decision_boundary(...)` | `model_2d`、`X_2d`、`y.values` | 二维分类边界图（PNG） |
| 学习曲线 | `plot_learning_curve(...)` | `estimator`、`X_train.values`、`y_train.values` | 训练/验证得分曲线（PNG） |

### 理解重点

- 五类可视化都不是训练的一部分，而是训练完成后的诊断步骤——它们不修改模型参数。
- 特征重要性图是当前分册区别于很多其他分类分册的重要评估入口，因为它直接读取树模型内部结构。
- 决策边界图依赖单独训练的 `model_2d`（在 PCA 空间），ROC 曲线依赖 `predict_proba(...)` 输出的概率矩阵。

## 5. 模块间的数据依赖关系

整个流水线的数据流向可以概括为以下依赖关系：

| 数据 | 生产者 | 消费者 |
|---|---|---|
| `decision_tree_classification_data` | `data_generation/classification.py` | `pipelines/classification/decision_tree.py` |
| `model`（主模型） | `train_model(...)` | `predict`、`predict_proba`、`plot_feature_importance` |
| `y_pred` | `model.predict(...)` | `plot_confusion_matrix` |
| `y_scores` | `model.predict_proba(...)` | `plot_roc_curve` |
| `model_2d` | `DecisionTreeClassifier.fit(...)`（PCA 空间） | `plot_decision_boundary` |
| 图片产物 | 各可视化函数 | `outputs/decision_tree/` 目录 |

### 理解重点

- 数据流向是单向的：数据生成 → 训练 → 预测 → 评估，各环节之间没有循环依赖。
- 这种清晰的数据流使得每个模块可以独立测试和替换。
- `model_2d` 和主模型 `model` 共享相同的 `max_depth=6` 超参数，但在不同特征空间中训练。

## 6. 运行后能得到什么

### 输出项

| 输出类型 | 当前结果 | 用途 |
|---|---|---|
| 终端标题 | `决策树分类流水线` | 在终端中定位当前运行入口 |
| 训练日志 | 训练耗时、树深度 $d$、叶节点数 $\vert T\vert$、`criterion` | 理解树复杂度和训练效率 |
| 混淆矩阵图 | `outputs/decision_tree/confusion_matrix.png` | 观察各类别误分类方向 |
| ROC 曲线图 | `outputs/decision_tree/roc_curve.png` | 评估多分类概率区分能力 |
| 特征重要性图 | `outputs/decision_tree/feature_importance.png` | 理解特征在树分裂中的贡献 |
| 决策边界图 | `outputs/decision_tree/decision_boundary.png` | 观察 PCA 2D 空间下的轴对齐切分 |
| 学习曲线图 | `outputs/decision_tree/learning_curve.png` | 诊断过拟合/欠拟合倾向 |

### 理解重点

- 运行结果并不只是一个模型对象，还包括面向阅读者的日志和多种图像产物。
- 对教学仓库而言，这种"代码 + 日志 + 图像"的组合比单纯返回分类结果更易理解。

## 7. 推荐的源码阅读顺序

1. 先看 `pipelines/classification/decision_tree.py` — 入口，了解整体流程
2. 再看 `model_training/classification/decision_tree.py` — 训练封装，理解超参数和日志
3. 再看 `result_visualization/confusion_matrix.py` — 最基础的分类结果评估
4. 再看 `result_visualization/roc_curve.py` — 概率区分能力评估
5. 再看 `result_visualization/feature_importance.py` — 树模型特有的解释性评估
6. 再看 `result_visualization/decision_boundary.py` — 空间划分可视化
7. 再看 `result_visualization/learning_curve.py` — 训练行为诊断
8. 最后回到 `data_generation/classification.py` — 理解数据生成参数

### 理解重点

- 先从入口看整体流程，再下钻到训练与可视化细节，阅读成本最低。
- 如果一开始就只看某一个可视化模块，容易看见局部却看不见完整链路。
- 这个阅读顺序也对应了数据流的方向：数据 → 训练 → 预测 → 评估。

## 常见坑

1. 把 `pipeline` 文件误认为训练算法实现本体——它只是编排层，真正的训练在 `model_training/` 中。
2. 不区分"主模型""二维可视化模型"和"学习曲线模型实例"的职责边界——三者训练在不同空间或用不同数据子集。
3. 不区分类别预测输出（`predict`）、概率输出（`predict_proba`）和特征重要性（`feature_importances_`）的用途——它们分别服务于不同的评估方式。
4. 只看单个文件，不顺着调用链理解整体执行流程——缺少全局视角容易产生误解。

## 小结

- 当前 Decision Tree 工程实现采用清晰的模块分层：数据生成 → 训练封装 → 流水线编排 → 结果可视化。
- `run()` 负责串联流程，`train_model(...)` 负责训练主模型，各可视化函数负责结果展示与诊断。
- 数据流是单向的，各模块职责单一，便于教学讲解和后续扩展其他算法的同类结构。
