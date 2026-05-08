---
title: Bagging 集成学习 — 总览
outline: deep
---

# Bagging 集成学习

## 本章目标

1. 明确本分册对应的 Bagging 源码入口与运行方式。
2. 理解当前 Bagging 文档各章节分别负责解释什么内容。
3. 建立从数据、模型、训练到分类评估的整体阅读路线——注意这是集成分类，包含混淆矩阵和 ROC 曲线评估。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/ensemble.py` | `EnsembleData.bagging()` 生成高噪声双月牙二分类数据 |
| 数据导出 | `data_generation/__init__.py` | 导出 `bagging_data` |
| 训练封装 | `model_training/ensemble/bagging.py` | `train_model(...)` 封装 `sklearn.ensemble.BaggingClassifier` 训练 |
| 端到端流水线 | `pipelines/ensemble/bagging.py` | 完成数据拆分、标准化、Bagging 训练、预测和分类评估 |
| 混淆矩阵可视化 | `result_visualization/confusion_matrix.py` | 绘制测试集混淆矩阵 |
| ROC 曲线可视化 | `result_visualization/roc_curve.py` | 绘制 ROC 曲线（当 `predict_proba` 可用时） |

## 默认配置速览（来自源码）

| 项目 | 当前实现 |
|---|---|
| 训练模型 | `BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=None), n_estimators=80, max_samples=0.8, max_features=1.0, bootstrap=True, oob_score=True, n_jobs=-1, random_state=42)` |
| 基学习器 | `DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1)`——完全生长的决策树（高方差低偏差） |
| 数据来源 | `make_moons(n_samples=500, noise=0.35, random_state=42)`——高噪声双月牙二分类 |
| 特征预处理 | `StandardScaler().fit_transform(X_train)`、`transform(X_test)`——训练/测试分离标准化 |
| 数据拆分 | `train_test_split(test_size=0.2, stratify=y, random_state=42)`——分层抽样 |
| 评估呈现 | 混淆矩阵 + ROC 曲线（条件可用）+ OOB 得分日志 |

## 阅读路线

1. [数学原理](/ensemble/bagging/01-mathematics)
2. [数据构成](/ensemble/bagging/02-data)
3. [思路与直觉](/ensemble/bagging/03-intuition)
4. [模型构建](/ensemble/bagging/04-model)
5. [训练与预测](/ensemble/bagging/05-training-and-prediction)
6. [评估与诊断](/ensemble/bagging/06-evaluation)
7. [工程实现](/ensemble/bagging/07-implementation)
8. [练习与参考文献](/ensemble/bagging/08-exercises-and-references)

## 如何运行

### 示例代码

```bash
python -m pipelines.ensemble.bagging
```

### 理解重点

- 这个命令会串起当前 Bagging 分册中最核心的工程流程。
- 运行后会以完全生长的决策树为基学习器，训练一个含 80 个基学习器的 Bagging 集成，并输出混淆矩阵、ROC 曲线（条件可用）和 OOB 得分。
- 当前流程是有监督分类——包含训练/测试切分、标准化（训练集拟合/测试集变换）、预测和概率输出。

## 先修

- [库生态总览](/foundations/overview)
- [NumPy 基础与数组概念](/foundations/numpy/01-basics)
- [预处理](/foundations/sklearn/02-preprocessing)
- [决策树分类](/classification/decision_tree)
- [术语表](/appendix/glossary)

## 小结

- 本分册严格对应当前仓库中的 Bagging 源码实现。
- Bagging 的核心特点：Bootstrap 并行采样 + 投票聚合 + OOB 误差估计 + 方差缩减——与 Boosting（串行、纠正残差、偏差缩减）在建模策略上有本质区别。
- 当前使用高噪声双月牙数据 + 完全生长决策树 + `BaggingClassifier(n_estimators=80)`，是展示 Bagging 方差缩减能力最经典的教学配置。
