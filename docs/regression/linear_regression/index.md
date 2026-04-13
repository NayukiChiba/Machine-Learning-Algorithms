---
title: 线性回归 — 总览
outline: deep
---

# 线性回归

> 对应代码：`pipelines/regression/linear_regression.py`、`model_training/regression/linear_regression.py`
>  
> 运行方式：`python -m pipelines.regression.linear_regression`

## 本章目标

1. 明确本分册对应的线性回归源码入口与运行方式。
2. 理解当前线性回归文档各章节分别负责解释什么内容。
3. 建立从数据、模型、训练到评估的整体阅读路线。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/regression.py` | `RegressionData.linear_regression()` 生成手工合成的线性房价数据 |
| 数据导出 | `data_generation/__init__.py` | 导出 `linear_regression_data` |
| 训练封装 | `model_training/regression/linear_regression.py` | `train_model(...)` 封装 `sklearn.linear_model.LinearRegression` 训练 |
| 端到端流水线 | `pipelines/regression/linear_regression.py` | 完成数据切分、训练、预测、残差图和学习曲线输出 |
| 残差可视化 | `result_visualization/residual_plot.py` | 绘制预测-真实与残差分布图 |
| 学习曲线可视化 | `result_visualization/learning_curve.py` | 绘制训练/验证得分曲线 |

## 默认配置速览（来自源码）

| 项目 | 当前实现 |
|---|---|
| 训练模型 | `LinearRegression()` |
| 数据切分 | `train_test_split(..., test_size=0.2, random_state=42)` |
| 特征预处理 | 当前流水线未使用标准化 |
| 数据来源 | 手工合成的 `面积`、`房间数`、`房龄` 与 `price` 数据 |
| 评估方式 | 残差图 + 学习曲线（`scoring='r2'`） |

## 阅读路线

1. [数学原理](/regression/linear_regression/01-mathematics)
2. [数据构成](/regression/linear_regression/02-data)
3. [思路与直觉](/regression/linear_regression/03-intuition)
4. [模型构建](/regression/linear_regression/04-model)
5. [训练与预测](/regression/linear_regression/05-training-and-prediction)
6. [评估与诊断](/regression/linear_regression/06-evaluation)
7. [工程实现](/regression/linear_regression/07-implementation)
8. [练习与参考文献](/regression/linear_regression/08-exercises-and-references)

## 如何运行

### 示例代码

```bash
python -m pipelines.regression.linear_regression
```

### 理解重点

- 这个命令会串起当前线性回归分册中最核心的工程流程。
- 运行后会生成残差图和学习曲线，并在控制台打印截距与各特征系数。
- 当前实现是教学型的基础线性回归示例，重点在于关系透明、系数可解释、流程简单。

## 先修

- [库生态总览](/foundations/overview)
- [NumPy 基础与数组概念](/foundations/numpy/01-basics)
- [预处理](/foundations/sklearn/02-preprocessing)
- [术语表](/appendix/glossary)

## 小结

- 本分册严格对应当前仓库中的线性回归源码实现。
- 阅读时建议始终把文档内容与 `pipelines/regression/linear_regression.py` 和 `model_training/regression/linear_regression.py` 对照起来看。
- 如果已经熟悉整体入口，可以直接从“模型构建”或“训练与预测”章节开始阅读。
