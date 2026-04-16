---
title: GaussianNB 高斯朴素贝叶斯 — 总览
outline: deep
---

# GaussianNB 高斯朴素贝叶斯

> 对应代码：`pipelines/classification/naive_bayes.py`、`model_training/classification/naive_bayes.py`
>
> 运行方式：`python -m pipelines.classification.naive_bayes`

## 本章目标

1. 明确本分册对应的 Naive Bayes 源码入口与运行方式。
2. 理解当前 Naive Bayes 文档各章节分别负责解释什么内容。
3. 建立从数据、模型、训练到评估的整体阅读路线。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据加载 | `data_generation/classification.py` | `ClassificationData.naive_bayes()` 加载 iris 多分类数据 |
| 数据导出 | `data_generation/__init__.py` | 导出 `naive_bayes_data` |
| 训练封装 | `model_training/classification/naive_bayes.py` | `train_model(...)` 封装 `sklearn.naive_bayes.GaussianNB` 训练 |
| 端到端流水线 | `pipelines/classification/naive_bayes.py` | 完成切分、标准化、训练、预测与可视化 |
| 混淆矩阵可视化 | `result_visualization/confusion_matrix.py` | 绘制预测结果混淆矩阵 |
| ROC 曲线可视化 | `result_visualization/roc_curve.py` | 绘制多分类 One-vs-Rest ROC 曲线 |
| 决策边界可视化 | `result_visualization/decision_boundary.py` | 绘制 PCA 2D 空间下的决策边界 |
| 学习曲线可视化 | `result_visualization/learning_curve.py` | 绘制训练/验证得分曲线 |

## 默认配置速览（来自源码）

| 项目 | 当前实现 |
|---|---|
| 训练模型 | `GaussianNB(var_smoothing=1e-9)` |
| 数据切分 | `train_test_split(..., test_size=0.2, random_state=42, stratify=y)` |
| 特征预处理 | `StandardScaler` 仅在训练集 `fit`，测试集 `transform` |
| 正式预测输出 | `y_pred = model.predict(X_test_s)` |
| 概率输出 | `y_scores = model.predict_proba(X_test_s)` |
| 评估方式 | 混淆矩阵 + ROC 曲线 + PCA 2D 决策边界 + 学习曲线 |

## 阅读路线

1. [数学原理](/classification/naive_bayes/01-mathematics)
2. [数据构成](/classification/naive_bayes/02-data)
3. [思路与直觉](/classification/naive_bayes/03-intuition)
4. [模型构建](/classification/naive_bayes/04-model)
5. [训练与预测](/classification/naive_bayes/05-training-and-prediction)
6. [评估与诊断](/classification/naive_bayes/06-evaluation)
7. [工程实现](/classification/naive_bayes/07-implementation)
8. [练习与参考文献](/classification/naive_bayes/08-exercises-and-references)

## 如何运行

### 示例代码

```bash
python -m pipelines.classification.naive_bayes
```

### 理解重点

- 这个命令会串起当前 Naive Bayes 分册中最核心的工程流程。
- 运行后会训练一个 `GaussianNB` 模型，并输出混淆矩阵、ROC 曲线、决策边界图和学习曲线。
- 当前任务是监督多分类，因此 `label` 会真实参与模型拟合与测试集预测。

## 先修

- [库生态总览](/foundations/overview)
- [NumPy 基础与数组概念](/foundations/numpy/01-basics)
- [预处理](/foundations/sklearn/02-preprocessing)
- [术语表](/appendix/glossary)

## 小结

- 本分册严格对应当前仓库中的 Naive Bayes 源码实现。
- 阅读时建议始终把文档内容与 `pipelines/classification/naive_bayes.py` 和 `model_training/classification/naive_bayes.py` 对照起来看。
- 如果已经熟悉整体入口，可以直接从“数据构成”“模型构建”或“训练与预测”章节开始阅读。
