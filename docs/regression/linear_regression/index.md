---
title: 线性回归 — 总览
outline: deep
---

# 线性回归

## 本章目标

1. 明确本分册对应的线性回归源码入口与运行方式——这是本仓库最基础的回归模型，关系透明、系数可解释。
2. 理解当前线性回归文档各章节分别负责解释什么内容。
3. 建立从线性模型、OLS 求解、系数解释到残差评估的整体阅读路线。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `src/mlAlgorithms/datasets/tabular/regressionDatasets.py` | `RegressionDatasetFactory.loadLinearRegressionDataset()` 手工合成线性房价数据 |
| 训练封装 | `src/mlAlgorithms/training/regression/regressionModels.py` | `trainLinearRegressionModel(...)` 封装 `sklearn.linear_model.LinearRegression` 训练 |
| 流水线注册 | `src/mlAlgorithms/catalog/pipelines.py` | `PipelineSpec("regression.linear_regression", ...)`——注册数据集、训练器、可视化配置 |
| 端到端流水线 | `src/mlAlgorithms/workflows/regressionRunner.py` | 完成数据切分、训练、预测、残差图和学习曲线输出 |

## 默认配置速览（来自源码）

| 项目 | 当前实现 |
|---|---|
| 训练模型 | `LinearRegression()`——使用 scikit-learn 默认配置，无超参数 |
| 数据来源 | 手工合成——`price = 2×面积 + 10×房间数 - 3×房龄 + N(0,10²) + 50`，200 样本 |
| 数据形态 | 3 个连续特征——`面积` $\in [20,80]$、`房间数` $\in [1,5]$、`房龄` $\in [1,20]$ |
| 特征预处理 | **无**——当前流水线未使用标准化（数据量纲直观且关系简单） |
| 数据切分 | `train_test_split(test_size=0.2, random_state=42)`——随机切分 |
| 评估方式 | 残差图 + 学习曲线（`scoring='r2'`） |

## 线性回归与本仓库其他回归算法的定位对比

| 配置项 | 线性回归 | 决策树回归 | SVR |
|---|---|---|---|
| 算法范式 | 全局线性拟合——OLS 闭式解 | 递归区域划分 + 局部常数 | 核方法 + ε-不敏感损失 |
| 关系假设 | 全局线性 | 无条件假设 | 非线性（核映射） |
| 可解释性 | 极强——`coef_` 直接解释影响方向与大小 | 中等——`feature_importances_` 只看分裂贡献 | 弱——支持向量难以直接解释 |
| 标准化 | 无（当前实现） | 无 | 有（`StandardScaler`） |
| 核心输出 | `coef_`、`intercept_` | `feature_importances_`、`get_depth()` | `support_vectors_`、`dual_coef_` |
| 超参数数 | 0 | 3 | 4 |
| 数据来源 | 手工合成（关系透明） | California Housing 真实数据 | `make_friedman1` 合成非线性 |
| 教学定位 | 回归起点——建立系数直觉 | 非线性 + 特征交互 | 核方法 + 最大间隔 |

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
python -m src.mlAlgorithms.workflows.regressionRunner --pipeline regression.linear_regression
```

### 理解重点

- 这个命令会训练一个线性回归模型——在手工合成的房价数据上拟合线性参数。
- 运行后会打印截距和各特征系数，并生成残差图和学习曲线。
- 当前流程是**有监督回归**——`price` 是训练标签，模型通过最小化平方误差学习 $\mathbf{w}$ 和 $b$。

## 先修

- [库生态总览](/foundations/overview)
- [NumPy 基础与数组概念](/foundations/numpy/01-basics)
- [术语表](/appendix/glossary)

## 小结

- 本分册严格对应当前仓库中的线性回归源码实现——手工合成数据、OLS 拟合、系数解释、残差评估构成最基础的回归流水线。
- 线性回归的核心特点：全局线性假设 + 闭式解 + 系数直接可解释——是回归学习的逻辑起点，也是后续正则化、SVR、决策树回归的对比基线。
- 当前使用显式公式生成的 3 特征合成数据 + 默认 `LinearRegression()`，是展示"关系透明 → 系数可验证"这一教学闭环的最简配置。
