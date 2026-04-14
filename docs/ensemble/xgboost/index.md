---
title: XGBoost — 总览
outline: deep
---

# XGBoost

> 对应代码：`pipelines/ensemble/xgboost.py`、`model_training/ensemble/xgboost.py`
>  
> 运行方式：`python -m pipelines.ensemble.xgboost`

## 本章目标

1. 明确本分册对应的 XGBoost 源码入口与运行方式。
2. 理解当前 XGBoost 文档各章节分别负责解释什么内容。
3. 建立从数据、模型、训练到评估的整体阅读路线。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/ensemble.py` | `EnsembleData.xgboost()` 加载 California Housing 真实回归数据 |
| 数据导出 | `data_generation/__init__.py` | 导出 `xgboost_data` |
| 训练封装 | `model_training/ensemble/xgboost.py` | `train_model(...)` 封装 `xgboost.XGBRegressor` 训练 |
| 端到端流水线 | `pipelines/ensemble/xgboost.py` | 完成数据切分、训练、预测、残差图与特征重要性图输出 |
| 残差可视化 | `result_visualization/residual_plot.py` | 绘制预测-真实与残差分布图 |
| 特征重要性可视化 | `result_visualization/feature_importance.py` | 绘制特征重要性柱状图 |

## 默认配置速览（来自源码）

| 项目 | 当前实现 |
|---|---|
| 训练模型 | `XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=1, subsample=0.9, colsample_bytree=0.9, gamma=0.0, reg_alpha=0.0, reg_lambda=1.0, random_state=42, n_jobs=-1)` |
| 数据切分 | `train_test_split(..., test_size=0.2, random_state=42)` |
| 特征预处理 | 当前流水线未使用标准化 |
| 数据来源 | `fetch_california_housing(as_frame=True)`，标签列重命名为 `price` |
| 评估方式 | 残差图 + 特征重要性图 |

## 阅读路线

1. [数学原理](/ensemble/xgboost/01-mathematics)
2. [数据构成](/ensemble/xgboost/02-data)
3. [思路与直觉](/ensemble/xgboost/03-intuition)
4. [模型构建](/ensemble/xgboost/04-model)
5. [训练与预测](/ensemble/xgboost/05-training-and-prediction)
6. [评估与诊断](/ensemble/xgboost/06-evaluation)
7. [工程实现](/ensemble/xgboost/07-implementation)
8. [练习与参考文献](/ensemble/xgboost/08-exercises-and-references)

## 如何运行

### 示例代码

```bash
python -m pipelines.ensemble.xgboost
```

### 理解重点

- 这个命令会串起当前 XGBoost 分册中最核心的工程流程。
- 运行后会生成残差图和特征重要性图，并在控制台打印当前训练使用的关键超参数。
- 当前实现重点在于展示 XGBoost 如何在真实表格回归数据上进行高性能树模型拟合。

## 先修

- [库生态总览](/foundations/overview)
- [NumPy 基础与数组概念](/foundations/numpy/01-basics)
- [预处理](/foundations/sklearn/02-preprocessing)
- [术语表](/appendix/glossary)

## 小结

- 本分册严格对应当前仓库中的 XGBoost 源码实现。
- 阅读时建议始终把文档内容与 `pipelines/ensemble/xgboost.py` 和 `model_training/ensemble/xgboost.py` 对照起来看。
- 如果已经熟悉整体入口，可以直接从“模型构建”或“训练与预测”章节开始阅读。
