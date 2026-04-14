---
title: LightGBM — 总览
outline: deep
---

# LightGBM

> 对应代码：`pipelines/ensemble/lightgbm.py`、`model_training/ensemble/lightgbm.py`
>  
> 运行方式：`python -m pipelines.ensemble.lightgbm`

## 本章目标

1. 明确本分册对应的 LightGBM 源码入口与运行方式。
2. 理解当前 LightGBM 文档各章节分别负责解释什么内容。
3. 建立从数据、模型、训练到评估的整体阅读路线。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/ensemble.py` | `EnsembleData.lightgbm()` 生成高维多分类数据 |
| 数据导出 | `data_generation/__init__.py` | 导出 `lightgbm_data` |
| 训练封装 | `model_training/ensemble/lightgbm.py` | `train_model(...)` 封装 `lightgbm.LGBMClassifier` 训练 |
| 端到端流水线 | `pipelines/ensemble/lightgbm.py` | 完成分层切分、标准化、训练、预测和三类可视化输出 |
| 混淆矩阵可视化 | `result_visualization/confusion_matrix.py` | 绘制分类混淆矩阵 |
| ROC 曲线可视化 | `result_visualization/roc_curve.py` | 绘制多分类 ROC 曲线 |
| 特征重要性可视化 | `result_visualization/feature_importance.py` | 绘制特征重要性柱状图 |

## 默认配置速览（来自源码）

| 项目 | 当前实现 |
|---|---|
| 训练模型 | `LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1, subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1)` |
| 数据切分 | `train_test_split(..., test_size=0.2, random_state=42, stratify=y)` |
| 特征预处理 | `StandardScaler` 仅在训练集 `fit`，测试集 `transform` |
| 数据来源 | `make_classification(...)` 生成 20 维 4 分类数据 |
| 评估方式 | 混淆矩阵 + ROC 曲线 + 特征重要性图 |

## 阅读路线

1. [数学原理](/ensemble/lightgbm/01-mathematics)
2. [数据构成](/ensemble/lightgbm/02-data)
3. [思路与直觉](/ensemble/lightgbm/03-intuition)
4. [模型构建](/ensemble/lightgbm/04-model)
5. [训练与预测](/ensemble/lightgbm/05-training-and-prediction)
6. [评估与诊断](/ensemble/lightgbm/06-evaluation)
7. [工程实现](/ensemble/lightgbm/07-implementation)
8. [练习与参考文献](/ensemble/lightgbm/08-exercises-and-references)

## 如何运行

### 示例代码

```bash
python -m pipelines.ensemble.lightgbm
```

### 理解重点

- 这个命令会串起当前 LightGBM 分册中最核心的工程流程。
- 运行后会生成混淆矩阵、ROC 曲线和特征重要性图，并在控制台打印训练使用的关键超参数。
- 当前实现重点在于展示 LightGBM 如何高效处理高维多分类表格数据。

## 先修

- [库生态总览](/foundations/overview)
- [NumPy 基础与数组概念](/foundations/numpy/01-basics)
- [预处理](/foundations/sklearn/02-preprocessing)
- [术语表](/appendix/glossary)

## 小结

- 本分册严格对应当前仓库中的 LightGBM 源码实现。
- 阅读时建议始终把文档内容与 `pipelines/ensemble/lightgbm.py` 和 `model_training/ensemble/lightgbm.py` 对照起来看。
- 如果已经熟悉整体入口，可以直接从“模型构建”或“训练与预测”章节开始阅读。
