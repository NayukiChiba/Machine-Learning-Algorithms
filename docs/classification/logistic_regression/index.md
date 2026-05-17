---
title: LogisticRegression 逻辑回归分类 — 总览
outline: deep
---

# LogisticRegression 逻辑回归分类

## 本章目标

1. 明确本分册对应的 Logistic Regression 源码入口与运行方式。
2. 理解当前 Logistic Regression 文档各章节分别负责解释什么内容。
3. 建立从数据、模型、训练到评估的整体阅读路线。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/classification.py` | `ClassificationData.logistic_regression()` 生成高维二分类数据 |
| 数据导出 | `data_generation/__init__.py` | 导出 `logistic_regression_data` |
| 训练封装 | `model_training/classification/logistic_regression.py` | `train_model(...)` 封装 `sklearn.linear_model.LogisticRegression` 训练 |
| 端到端流水线 | `pipelines/classification/logistic_regression.py` | 完成切分、标准化、训练、预测与可视化 |
| 混淆矩阵可视化 | `result_visualization/confusion_matrix.py` | 绘制预测结果混淆矩阵 |
| ROC 曲线可视化 | `result_visualization/roc_curve.py` | 绘制二分类 ROC 曲线 |
| 决策边界可视化 | `result_visualization/decision_boundary.py` | 绘制 PCA 2D 空间下的决策边界 |
| 学习曲线可视化 | `result_visualization/learning_curve.py` | 绘制训练/验证得分曲线 |

## 默认配置速览（来自源码）

| 项目 | 当前实现 |
|---|---|
| 训练模型 | `LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, class_weight=None, random_state=42)` |
| 数据切分 | `train_test_split(..., test_size=0.2, random_state=42, stratify=y)` |
| 特征预处理 | `StandardScaler` 仅在训练集 `fit`，测试集 `transform`——逻辑回归基于梯度优化，标准化有助于收敛稳定和系数可比性 |
| 正式预测输出 | `y_pred = model.predict(X_test_s)` |
| 概率输出 | `y_scores = model.predict_proba(X_test_s)`——Sigmoid 映射后的正类概率 |
| 评估方式 | 混淆矩阵 + ROC 曲线 + PCA 2D 决策边界 + 学习曲线 |

## 阅读路线

1. [数学原理](/classification/logistic_regression/01-mathematics)
2. [数据构成](/classification/logistic_regression/02-data)
3. [思路与直觉](/classification/logistic_regression/03-intuition)
4. [模型构建](/classification/logistic_regression/04-model)
5. [训练与预测](/classification/logistic_regression/05-training-and-prediction)
6. [评估与诊断](/classification/logistic_regression/06-evaluation)
7. [工程实现](/classification/logistic_regression/07-implementation)
8. [练习与参考文献](/classification/logistic_regression/08-exercises-and-references)

## 如何运行

### 示例代码

```bash
python -m pipelines.classification.logistic_regression
```

### 理解重点

- 这个命令会串起当前 Logistic Regression 分册中最核心的工程流程。
- 运行后会训练一个逻辑回归模型（通过 `lbfgs` 优化器最小化交叉熵损失），并输出混淆矩阵、ROC 曲线、决策边界图和学习曲线。
- 当前任务是监督二分类，因此 `label` 会真实参与模型拟合与测试集预测。

## 先修

- [库生态总览](/foundations/overview)
- [NumPy 基础与数组概念](/foundations/numpy/01-basics)
- [预处理](/foundations/sklearn/02-preprocessing)
- [项目架构](/appendix/)

## 小结

- 本分册严格对应当前仓库中的 Logistic Regression 源码实现。
- 逻辑回归的核心特点：线性打分 + Sigmoid 概率映射 + 交叉熵优化——天然兼具分类能力与概率解释。
- 与 KNN（懒惰学习、局部投票）和决策树（递归轴对齐切分）不同，逻辑回归学习的是全局线性概率边界 $\mathbf{w}^T\mathbf{x} + b = 0$。
