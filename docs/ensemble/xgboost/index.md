---
title: XGBoost 极限梯度提升 — 总览
outline: deep
---

# XGBoost 极限梯度提升

## 本章目标

1. 明确本分册对应的 XGBoost 源码入口与运行方式——注意这是回归任务，与 Bagging/GBDT/LightGBM 的分类任务不同。
2. 理解当前 XGBoost 文档各章节分别负责解释什么内容。
3. 建立从数据、模型、训练到回归评估的整体阅读路线。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/ensemble.py` | `EnsembleData.xgboost()` 返回加州房价真实数据集 |
| 数据导出 | `data_generation/__init__.py` | 导出 `xgboost_data` |
| 训练封装 | `model_training/ensemble/xgboost.py` | `train_model(...)` 封装 `xgboost.XGBRegressor` 训练——含可选依赖检查 |
| 端到端流水线 | `pipelines/ensemble/xgboost.py` | 完成数据拆分、XGBoost 训练、预测和回归评估 |
| 残差分析可视化 | `result_visualization/residual_plot.py` | 绘制预测残差散点图和分布图 |
| 特征重要性可视化 | `result_visualization/feature_importance.py` | 绘制特征重要性柱状图（8 个特征排序） |

## 默认配置速览（来自源码）

| 项目 | 当前实现 |
|---|---|
| 训练模型 | `XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=1, subsample=0.9, colsample_bytree=0.9, gamma=0.0, reg_alpha=0.0, reg_lambda=1.0, random_state=42, n_jobs=-1)` |
| 数据来源 | `fetch_california_housing(as_frame=True)`——加州房价真实数据集，20640 样本 × 8 特征 |
| 特征预处理 | **无标准化**——树模型天然对特征缩放不敏感 |
| 数据拆分 | `train_test_split(test_size=0.2, random_state=42)`——**无 stratify**（回归无类别） |
| 评估呈现 | 残差分析图 + 特征重要性 + 训练耗时日志 |

## XGBoost 与本仓库其他集成模型的对比

| 配置项 | Bagging | GBDT | LightGBM | XGBoost |
|---|---|---|---|---|
| 任务类型 | **分类** | **分类** | **分类** | **回归** |
| 数据 | 双月牙（合成） | 8 维合成 | 20 维合成 | 加州房价（真实） |
| 样本数 | 500 | 500 | 1000 | **20640** |
| 基学习器 | `DecisionTree` | `GradientBoosting` | `LGBMClassifier` | **`XGBRegressor`** |
| 树数量 | 80 | 200 | 300 | 300 |
| 学习率 | — | 0.1 | 0.05 | 0.05 |
| 树深度 | `max_depth=None` | `max_depth=3` | `num_leaves=31` | **`max_depth=6`** |
| 标准化 | 有 | 有 | 有 | **无** |
| 分层抽样 | 有 | 有 | 有 | **无** |
| 依赖 | sklearn 内置 | sklearn 内置 | `pip install lightgbm` | **`pip install xgboost`** |
| 评估 | 混淆矩阵 + ROC | 混淆矩阵 + ROC + 特征重要性 + 学习曲线 | 混淆矩阵 + ROC + 特征重要性 | **残差图 + 特征重要性** |

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
# 前置：安装 xgboost
pip install xgboost

# 运行流水线
python -m pipelines.ensemble.xgboost
```

### 理解重点

- `xgboost` 是可选依赖——首次运行前需手动安装。当前训练源码有 `try/except ImportError` 保护。
- 这是本仓库集成学习分册中唯一的**回归**任务——输出是连续的房价预测值，而非离散类别。
- XGBoost 使用真实数据集（非合成数据）——20640 条加州房价记录，充分体现 XGBoost 在工业级表格数据上的工程实力。

## 先修

- [库生态总览](/foundations/overview)
- [NumPy 基础与数组概念](/foundations/numpy/01-basics)
- [决策树分类](/classification/decision_tree/)
- [GBDT 梯度提升树](/ensemble/gbdt/)
- [LightGBM](/ensemble/lightgbm/)
- [项目架构](/appendix/)

## 小结

- 本分册严格对应当前仓库中的 XGBoost 源码实现。
- XGBoost 的核心特点：二阶泰勒展开（Hessian）+ 显式 L1/L2 正则化 + 加权分位数草图 + 稀疏感知——在 GBDT 数学框架上引入了更精确的目标函数近似和更强的正则化手段。
- 当前使用加州房价真实数据 + `XGBRegressor(n_estimators=300, max_depth=6)`，是展示 XGBoost 在工业回归任务上综合实力的经典教学配置。
