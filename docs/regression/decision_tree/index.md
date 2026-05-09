---
title: 决策树回归 — 总览
outline: deep
---

# 决策树回归

## 本章目标

1. 明确本分册对应的决策树回归源码入口与运行方式——注意这是回归任务，与分类决策树的评估指标和可视化有本质差异。
2. 理解当前决策树回归文档各章节分别负责解释什么内容。
3. 建立从区域划分、平方误差最小化、复杂度控制到残差评估的整体阅读路线。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `src/mlAlgorithms/datasets/tabular/regressionDatasets.py` | `RegressionDatasetFactory.loadDecisionTreeRegressionDataset()` 加载 California Housing 真实回归数据 |
| 训练封装 | `src/mlAlgorithms/training/regression/regressionModels.py` | `trainDecisionTreeRegressionModel(...)` 封装 `sklearn.tree.DecisionTreeRegressor` 训练 |
| 流水线注册 | `src/mlAlgorithms/catalog/pipelines.py` | `PipelineSpec("regression.decision_tree", ...)`——注册数据集、训练器、可视化配置 |
| 端到端流水线 | `src/mlAlgorithms/workflows/regressionRunner.py` | 完成数据切分、训练、预测、残差图、特征重要性图、学习曲线和树结构图输出 |

## 默认配置速览（来自源码）

| 项目 | 当前实现 |
|---|---|
| 训练模型 | `DecisionTreeRegressor(max_depth=6, min_samples_split=6, min_samples_leaf=3, random_state=42)` |
| 数据来源 | `fetch_california_housing(as_frame=True)`——真实加州房价数据集，标签列重命名为 `price` |
| 数据形态 | 20640 样本 × 8 特征——`MedInc`、`HouseAge`、`AveRooms`、`AveBedrms`、`Population`、`AveOccup`、`Latitude`、`Longitude` |
| 特征预处理 | **无**——树模型基于特征阈值分裂，不需要标准化 |
| 数据切分 | `train_test_split(test_size=0.2, random_state=42)`——随机切分 |
| 评估方式 | 残差图 + 特征重要性图 + 学习曲线（`scoring='r2'`）+ 树结构图 |

## 决策树回归与本仓库其他回归算法的定位对比

| 配置项 | 线性回归 | SVR | 决策树回归 |
|---|---|---|---|
| 算法范式 | 全局线性拟合 | 核方法 + 最大间隔 | **递归区域划分 + 局部常数** |
| 关系假设 | 全局线性 | 非线性（核映射） | **无全局假设——分段常数** |
| 特征交互 | 需手工构造 | 核隐式处理 | **自然表达——条件分支** |
| 标准化 | 有（`StandardScaler`） | 有（`StandardScaler`） | **无** |
| 核心输出 | 系数 $\beta_j$ | 支持向量 | **`feature_importances_` + 树结构** |
| 可解释性 | 系数层面强 | 较弱 | **规则路径直观** |
| 过拟合风险 | 低 | 中 | **高——需 `max_depth` 等约束** |
| 数据来源 | 手动合成 | `make_friedman1` | **California Housing 真实数据** |

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
python -m src.mlAlgorithms.workflows.regressionRunner --pipeline regression.decision_tree
```

### 理解重点

- 这个命令会训练一个决策树回归模型——在 California Housing 上做区域划分和局部常数预测。
- 运行后会生成残差图、特征重要性图、学习曲线和树结构图——四类可视化从不同角度诊断模型行为。
- 当前流程是**有监督回归**——`price` 是训练标签，树通过最小化平方误差学习最优分裂。

## 先修

- [库生态总览](/foundations/overview)
- [NumPy 基础与数组概念](/foundations/numpy/01-basics)
- [线性回归](/regression/linear_regression)
- [术语表](/appendix/glossary)

## 小结

- 本分册严格对应当前仓库中的决策树回归源码实现——数据加载、区域分裂、局部常数预测、四类可视化构成完整回归树流水线。
- 决策树回归的核心特点：平方误差最小化分裂 + 局部常数预测 + 无全局函数假设 + 自然处理非线性与特征交互——与线性回归的全局线性假设形成根本差异。
- 当前使用 California Housing 真实数据 + `DecisionTreeRegressor(max_depth=6)` 三重复杂度约束，是展示树模型处理真实非线性回归问题最经典的教学配置。
