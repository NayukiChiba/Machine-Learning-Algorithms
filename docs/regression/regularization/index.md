---
title: 正则化回归 — 总览
outline: deep
---

# 正则化回归

> 对应代码：`pipelines/regression/regularization.py`、`model_training/regression/regularization.py`
>  
> 运行方式：`python -m pipelines.regression.regularization`

## 本章目标

1. 明确本分册对应的正则化回归源码入口与运行方式。
2. 理解当前文档各章节分别解释哪一层实现细节。
3. 建立从数据、模型、训练到诊断的整体阅读路线。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/regression.py` | `RegressionData.regularization()` 构造 diabetes + 共线性 + 噪声特征数据 |
| 数据导出 | `data_generation/__init__.py` | 导出 `regularization_data` |
| 训练封装 | `model_training/regression/regularization.py` | `train_model(...)` 一次训练 `Lasso`、`Ridge`、`ElasticNet` |
| 端到端流水线 | `pipelines/regression/regularization.py` | 完成数据切分、标准化、训练、预测、残差图输出 |
| 残差可视化 | `result_visualization/residual_plot.py` | 为每个模型绘制预测-真实图与残差分布图 |

## 默认配置速览（来自源码）

| 项目 | 当前实现 |
|---|---|
| 训练模型 | `Lasso(alpha=0.15)`、`Ridge(alpha=2.0)`、`ElasticNet(alpha=0.2, l1_ratio=0.5)` |
| 数据切分 | `train_test_split(..., test_size=0.2, random_state=42)` |
| 特征预处理 | `StandardScaler` 仅在训练集 `fit`，测试集 `transform` |
| 数据来源 | `load_diabetes(as_frame=True)` 后追加相关特征与纯噪声特征 |
| 评估方式 | 残差图 + 控制台系数日志 |

## 阅读路线

1. [数学原理](/regression/regularization/01-mathematics)
2. [数据构成](/regression/regularization/02-data)
3. [思路与直觉](/regression/regularization/03-intuition)
4. [模型构建](/regression/regularization/04-model)
5. [训练与预测](/regression/regularization/05-training-and-prediction)
6. [评估与诊断](/regression/regularization/06-evaluation)
7. [工程实现](/regression/regularization/07-implementation)
8. [练习与参考文献](/regression/regularization/08-exercises-and-references)

## 如何运行

### 示例代码

```bash
python -m pipelines.regression.regularization
```

### 理解重点

- 这个命令会串起当前分册里最核心的工程流程。
- 运行后会依次训练三种正则化模型，并为每个模型生成一张残差分析图。
- 控制台还会打印 `alpha`、`l1_ratio`、截距、接近 0 的系数数量与各特征系数。

## 先修

- [库生态总览](/foundations/overview)
- [NumPy 基础与数组概念](/foundations/numpy/01-basics)
- [预处理](/foundations/sklearn/02-preprocessing)
- [术语表](/appendix/glossary)

## 小结

- 本分册对应的是当前仓库里一套对比式的正则化回归实现，而不是单一模型教程。
- 阅读时建议始终把文档与 `pipelines/regression/regularization.py` 和 `model_training/regression/regularization.py` 对照起来看。
- 如果已经熟悉整体入口，可以直接从“模型构建”或“训练与预测”章节开始阅读。
