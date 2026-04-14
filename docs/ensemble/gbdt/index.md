---
title: GBDT — 总览
outline: deep
---

# GBDT

> 对应代码：`pipelines/ensemble/gbdt.py`、`model_training/ensemble/gbdt.py`
>  
> 运行方式：`python -m pipelines.ensemble.gbdt`

## 本章目标

1. 明确本分册对应的 GBDT 源码入口与运行方式。
2. 理解当前 GBDT 文档各章节分别负责解释什么内容。
3. 建立从数据、模型、训练到评估的整体阅读路线。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/ensemble.py` | `EnsembleData.gbdt()` 生成中等难度多分类数据 |
| 数据导出 | `data_generation/__init__.py` | 导出 `gbdt_data` |
| 训练封装 | `model_training/ensemble/gbdt.py` | `train_model(...)` 封装 `sklearn.ensemble.GradientBoostingClassifier` 训练 |
| 端到端流水线 | `pipelines/ensemble/gbdt.py` | 完成分层切分、标准化、训练、预测和四类评估输出 |
| 混淆矩阵可视化 | `result_visualization/confusion_matrix.py` | 绘制分类混淆矩阵 |
| ROC 曲线可视化 | `result_visualization/roc_curve.py` | 绘制多分类 ROC 曲线 |
| 特征重要性可视化 | `result_visualization/feature_importance.py` | 绘制特征重要性柱状图 |
| 学习曲线可视化 | `result_visualization/learning_curve.py` | 绘制训练/验证得分曲线 |

## 默认配置速览（来自源码）

| 项目 | 当前实现 |
|---|---|
| 训练模型 | `GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, subsample=1.0, random_state=42)` |
| 数据切分 | `train_test_split(..., test_size=0.2, random_state=42, stratify=y)` |
| 特征预处理 | `StandardScaler` 仅在训练集 `fit`，测试集 `transform` |
| 数据来源 | `make_classification(...)` 生成 8 维 3 分类数据 |
| 评估方式 | 混淆矩阵 + ROC 曲线 + 特征重要性图 + 学习曲线 |

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
- 运行后会生成混淆矩阵、ROC 曲线、特征重要性图和学习曲线，并在控制台打印训练使用的关键超参数。
- 当前实现重点在于展示 GBDT 如何通过串行拟合伪残差逐步提升多分类边界效果。

## 先修

- [库生态总览](/foundations/overview)
- [NumPy 基础与数组概念](/foundations/numpy/01-basics)
- [预处理](/foundations/sklearn/02-preprocessing)
- [术语表](/appendix/glossary)

## 小结

- 本分册严格对应当前仓库中的 GBDT 源码实现。
- 阅读时建议始终把文档内容与 `pipelines/ensemble/gbdt.py` 和 `model_training/ensemble/gbdt.py` 对照起来看。
- 如果已经熟悉整体入口，可以直接从“模型构建”或“训练与预测”章节开始阅读。
