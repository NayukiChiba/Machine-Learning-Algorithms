---
title: Bagging 与随机森林 — 总览
outline: deep
---

# Bagging 与随机森林

> 对应代码：`pipelines/ensemble/bagging.py`、`model_training/ensemble/bagging.py`
>  
> 运行方式：`python -m pipelines.ensemble.bagging`

## 本章目标

1. 明确本分册对应的 Bagging 源码入口与运行方式。
2. 理解当前 Bagging 文档各章节分别负责解释什么内容。
3. 建立从数据、模型、训练到评估的整体阅读路线。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/ensemble.py` | `EnsembleData.bagging()` 生成高噪声双月牙二分类数据 |
| 数据导出 | `data_generation/__init__.py` | 导出 `bagging_data` |
| 训练封装 | `model_training/ensemble/bagging.py` | `train_model(...)` 封装 `sklearn.ensemble.BaggingClassifier` 训练 |
| 端到端流水线 | `pipelines/ensemble/bagging.py` | 完成分层切分、标准化、训练、预测和分类评估输出 |
| 混淆矩阵可视化 | `result_visualization/confusion_matrix.py` | 绘制分类混淆矩阵 |
| ROC 曲线可视化 | `result_visualization/roc_curve.py` | 在概率输出可用时绘制 ROC 曲线 |

## 默认配置速览（来自源码）

| 项目 | 当前实现 |
|---|---|
| 训练模型 | `BaggingClassifier(estimator=DecisionTreeClassifier(...), n_estimators=80, max_samples=0.8, max_features=1.0, bootstrap=True, oob_score=True, random_state=42, n_jobs=-1)` |
| 数据切分 | `train_test_split(..., test_size=0.2, random_state=42, stratify=y)` |
| 特征预处理 | `StandardScaler` 仅在训练集 `fit`，测试集 `transform` |
| 数据来源 | `make_moons(n_samples=500, noise=0.35, random_state=42)` |
| 评估方式 | 混淆矩阵 + 条件性 ROC 曲线 + OOB 得分日志 |

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
- 运行后会生成混淆矩阵，并在模型支持概率输出时额外生成 ROC 曲线，同时在控制台打印 OOB 得分。
- 当前实现重点在于展示 Bagging 如何通过 Bootstrap 重采样和并行集成降低高方差模型的不稳定性。

## 先修

- [库生态总览](/foundations/overview)
- [NumPy 基础与数组概念](/foundations/numpy/01-basics)
- [预处理](/foundations/sklearn/02-preprocessing)
- [术语表](/appendix/glossary)

## 小结

- 本分册严格对应当前仓库中的 Bagging 源码实现。
- 阅读时建议始终把文档内容与 `pipelines/ensemble/bagging.py` 和 `model_training/ensemble/bagging.py` 对照起来看。
- 如果已经熟悉整体入口，可以直接从“模型构建”或“训练与预测”章节开始阅读。
