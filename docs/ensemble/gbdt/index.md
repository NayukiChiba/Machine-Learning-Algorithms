---
title: GBDT 梯度提升树 — 总览
outline: deep
---

# GBDT 梯度提升树

## 本章目标

1. 明确本分册对应的 GBDT 源码入口与运行方式。
2. 理解当前 GBDT 文档各章节分别负责解释什么内容。
3. 建立从数据、模型、训练到分类评估的整体阅读路线——注意这是多分类集成，包含混淆矩阵、ROC 曲线、特征重要性和学习曲线四项评估。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/ensemble.py` | `EnsembleData.gbdt()` 生成多类别分类数据（8 特征 × 3 类别） |
| 数据导出 | `data_generation/__init__.py` | 导出 `gbdt_data` |
| 训练封装 | `model_training/ensemble/gbdt.py` | `train_model(...)` 封装 `sklearn.ensemble.GradientBoostingClassifier` 训练 |
| 端到端流水线 | `pipelines/ensemble/gbdt.py` | 完成数据拆分、标准化、GBDT 训练、预测和四项评估 |
| 混淆矩阵可视化 | `result_visualization/confusion_matrix.py` | 绘制测试集混淆矩阵 |
| ROC 曲线可视化 | `result_visualization/roc_curve.py` | 绘制多分类 ROC 曲线 |
| 特征重要性可视化 | `result_visualization/feature_importance.py` | 绘制特征重要性柱状图 |
| 学习曲线可视化 | `result_visualization/learning_curve.py` | 绘制学习曲线（训练集/测试集准确率随迭代数变化） |

## 默认配置速览（来自源码）

| 项目 | 当前实现 |
|---|---|
| 训练模型 | `GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, subsample=1.0, random_state=42)` |
| 基学习器 | 浅层决策树（`max_depth=3`）——高偏差低方差，GBDT 通过串行纠错逐步降低偏差 |
| 数据来源 | `make_classification(n_samples=500, n_features=8, n_informative=4, n_redundant=2, n_classes=3, class_sep=0.7, random_state=42)` |
| 特征预处理 | `StandardScaler().fit_transform(X_train)`、`transform(X_test)`——训练/测试分离标准化 |
| 数据拆分 | `train_test_split(test_size=0.2, stratify=y, random_state=42)`——分层抽样 |
| 评估呈现 | 混淆矩阵 + ROC 曲线 + 特征重要性 + 学习曲线 |

## 阅读路线

1. [数学原理](/ensemble/gbdt/01-mathematics)
2. [数据构成](/ensemble/gbdt/02-data)
3. [思路与直觉](/ensemble/gbdt/03-intuition)
4. [模型构建](/ensemble/gbdt/04-model)
5. [训练与预测](/ensemble/gbdt/05-training-and-prediction)
6. [评估与诊断](/ensemble/gbdt/06-evaluation)
7. [工程实现](/ensemble/gbdt/07-implementation)
8. [练习与参考文献](/ensemble/gbdt/08-exercises-and-references)

## 如何运行

### 示例代码

```bash
python -m pipelines.ensemble.gbdt
```

### 理解重点

- 这个命令会串起当前 GBDT 分册中最核心的工程流程。
- 运行后会以浅层决策树（`max_depth=3`）为基学习器，训练一个含 200 个弱学习器的 GBDT 集成，并输出混淆矩阵、ROC 曲线、特征重要性和学习曲线。
- 当前流程是有监督多分类——包含训练/测试切分、标准化（训练集拟合/测试集变换）、预测和概率输出。
- 与 Bagging 分册的关键差异：GBDT 有特征重要性（`feature_importances_`）和学习曲线两项额外评估。

## 先修

- [库生态总览](/foundations/overview)
- [NumPy 基础与数组概念](/foundations/numpy/01-basics)
- [预处理](/foundations/sklearn/02-preprocessing)
- [决策树分类](/classification/decision_tree)
- [Bagging 集成学习](/ensemble/bagging)
- [术语表](/appendix/glossary)

## 小结

- 本分册严格对应当前仓库中的 GBDT 源码实现。
- GBDT 的核心特点：串行梯度提升 + 浅层决策树作为弱学习器 + 学习率收缩 + 特征重要性——与 Bagging（并行、强学习器、降方差、OOB 估计）在建模策略上有本质区别。
- 当前使用多类别（3 类）数据 + 浅层决策树 + `GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)`，是展示 Boosting 偏差缩减能力最经典的教学配置。
