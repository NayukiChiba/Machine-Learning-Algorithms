---
title: LightGBM 梯度提升机 — 总览
outline: deep
---

# LightGBM 梯度提升机

## 本章目标

1. 明确本分册对应的 LightGBM 源码入口与运行方式。
2. 理解当前 LightGBM 文档各章节分别负责解释什么内容。
3. 建立从数据、模型、训练到分类评估的整体阅读路线。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/ensemble.py` | `EnsembleData.lightgbm()` 生成高维多类别分类数据 |
| 数据导出 | `data_generation/__init__.py` | 导出 `lightgbm_data` |
| 训练封装 | `model_training/ensemble/lightgbm.py` | `train_model(...)` 封装 `lightgbm.LGBMClassifier` 训练——含可选依赖检查 |
| 端到端流水线 | `pipelines/ensemble/lightgbm.py` | 完成数据拆分、标准化、LightGBM 训练、预测和分类评估 |
| 混淆矩阵可视化 | `result_visualization/confusion_matrix.py` | 绘制测试集混淆矩阵（4×4 多分类热力图） |
| ROC 曲线可视化 | `result_visualization/roc_curve.py` | 绘制 ROC 曲线（多分类 one-vs-rest） |
| 特征重要性可视化 | `result_visualization/feature_importance.py` | 绘制特征重要性柱状图（20 个特征排序） |

## 默认配置速览（来自源码）

| 项目 | 当前实现 |
|---|---|
| 训练模型 | `LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1, subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1)` |
| 数据来源 | `make_classification(n_samples=1000, n_features=20, n_informative=8, n_redundant=5, n_classes=4, class_sep=0.6)`——高维四分类 |
| 特征预处理 | `StandardScaler().fit_transform(X_train)`、`transform(X_test)`——训练/测试分离标准化 |
| 数据拆分 | `train_test_split(test_size=0.2, stratify=y, random_state=42)`——分层抽样 |
| 评估呈现 | 混淆矩阵 + ROC 曲线 + 特征重要性 + 训练耗时日志 |

## LightGBM vs GBDT 默认配置对比

| 配置项 | GBDT (sklearn) | LightGBM |
|---|---|---|
| 数据维度 | 8 特征 × 3 类 | **20 特征 × 4 类** |
| 树数量 | 200 | **300** |
| 学习率 | 0.1 | **0.05** |
| 复杂度控制 | `max_depth=3` | **`num_leaves=31` + `max_depth=-1`** |
| 行采样 | `subsample=1.0` | **`subsample=0.9`（GOSS）** |
| 列采样 | 无 | **`colsample_bytree=0.9`** |
| 依赖 | sklearn 内置 | **`pip install lightgbm`** |
| 训练方式 | Level-wise 生长 | **Leaf-wise 生长 + 直方图加速** |

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
# 前置：安装 lightgbm
pip install lightgbm

# 运行流水线
python -m pipelines.ensemble.lightgbm
```

### 理解重点

- `lightgbm` 是可选依赖——首次运行前需手动安装。当前训练源码有 `try/except ImportError` 保护。
- 这个命令会串起当前 LightGBM 分册中最核心的工程流程——以 Leaf-wise 浅层回归树为基学习器，训练一个含 300 个基学习器的 GBDT 集成。
- 当前流程是有监督分类——包含训练/测试切分、标准化（训练集拟合/测试集变换）、预测和概率输出。
- 与 Bagging 和 GBDT 不同，LightGBM 使用直方图算法加速训练——在处理大规模数据时速度优势显著。

## 先修

- [库生态总览](/foundations/overview)
- [NumPy 基础与数组概念](/foundations/numpy/01-basics)
- [预处理](/foundations/sklearn/02-preprocessing)
- [决策树分类](/classification/decision_tree/)
- [GBDT 梯度提升树](/ensemble/gbdt/)（重要——LightGBM 是 GBDT 的高效工程实现）
- [项目架构](/appendix/)

## 小结

- 本分册严格对应当前仓库中的 LightGBM 源码实现。
- LightGBM 的核心特点：Leaf-wise 生长 + 直方图加速 + GOSS 采样 + EFB 特征捆绑——在 GBDT 的数学框架上进行了激进的工程优化。
- 当前使用高维四分类数据（20 特征 × 4 类）+ 浅层直方图树 `num_leaves=31` + `LGBMClassifier(n_estimators=300)`，是展示 LightGBM 处理高维数据速度优势的经典教学配置。
