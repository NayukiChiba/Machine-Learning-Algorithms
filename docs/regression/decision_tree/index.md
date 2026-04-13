---
title: 决策树回归 — 总览
outline: deep
---

# 决策树回归

> 对应代码：`pipelines/regression/decision_tree.py`、`model_training/regression/decision_tree.py`
>  
> 运行方式：`python -m pipelines.regression.decision_tree`

## 本章目标

1. 明确本分册对应的决策树回归源码入口与运行方式。
2. 理解当前决策树回归文档各章节分别负责解释什么内容。
3. 建立从数据、模型、训练到评估的整体阅读路线。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/regression.py` | `RegressionData.decision_tree()` 加载 California Housing 真实回归数据 |
| 数据导出 | `data_generation/__init__.py` | 导出 `decision_tree_regression_data` |
| 训练封装 | `model_training/regression/decision_tree.py` | `train_model(...)` 封装 `sklearn.tree.DecisionTreeRegressor` 训练 |
| 端到端流水线 | `pipelines/regression/decision_tree.py` | 完成数据切分、训练、预测、残差图、特征重要性图和学习曲线输出 |
| 残差可视化 | `result_visualization/residual_plot.py` | 绘制预测-真实与残差分布图 |
| 特征重要性可视化 | `result_visualization/feature_importance.py` | 绘制树模型 `feature_importances_` 柱状图 |
| 学习曲线可视化 | `result_visualization/learning_curve.py` | 绘制训练/验证得分曲线 |

## 默认配置速览（来自源码）

| 项目 | 当前实现 |
|---|---|
| 训练模型 | `DecisionTreeRegressor(max_depth=6, min_samples_split=6, min_samples_leaf=3, random_state=42)` |
| 数据切分 | `train_test_split(..., test_size=0.2, random_state=42)` |
| 特征预处理 | 当前流水线未使用标准化 |
| 数据来源 | `fetch_california_housing(as_frame=True)`，标签列重命名为 `price` |
| 评估方式 | 残差图 + 特征重要性图 + 学习曲线（`scoring='r2'`） |

## 阅读路线

1. [数学原理](/regression/decision_tree/01-mathematics)
2. [数据构成](/regression/decision_tree/02-data)
3. [思路与直觉](/regression/decision_tree/03-intuition)
4. [模型构建](/regression/decision_tree/04-model)
5. [训练与预测](/regression/decision_tree/05-training-and-prediction)
6. [评估与诊断](/regression/decision_tree/06-evaluation)
7. [工程实现](/regression/decision_tree/07-implementation)
8. [练习与参考文献](/regression/decision_tree/08-exercises-and-references)

## 如何运行

### 示例代码

```bash
python -m pipelines.regression.decision_tree
```

### 理解重点

- 这个命令会串起当前决策树回归分册中最核心的工程流程。
- 运行后会生成残差图、特征重要性图和学习曲线，并在控制台打印树深度、叶子节点数和训练耗时日志。
- 当前实现重点在于展示树模型如何处理真实数据中的非线性关系与特征重要性。

## 先修

- [库生态总览](/foundations/overview)
- [NumPy 基础与数组概念](/foundations/numpy/01-basics)
- [预处理](/foundations/sklearn/02-preprocessing)
- [术语表](/appendix/glossary)

## 小结

- 本分册严格对应当前仓库中的决策树回归源码实现。
- 阅读时建议始终把文档内容与 `pipelines/regression/decision_tree.py` 和 `model_training/regression/decision_tree.py` 对照起来看。
- 如果已经熟悉整体入口，可以直接从“模型构建”或“训练与预测”章节开始阅读。
