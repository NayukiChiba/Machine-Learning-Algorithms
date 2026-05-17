---
title: 项目架构 — CLI 与流水线
outline: deep
---

# CLI 与流水线

## 本章目标

1. 掌握 CLI 四种命令的用法和执行流程。
2. 了解全部 20 条流水线的配置速览。

---

## 1. CLI 入口

**入口文件**：[`main.py`](https://github.com/NayukiChiba/Machine-Learning-Algorithms/blob/main/main.py)

### 1.1 命令一览

```bash
# 列出所有可用流水线
python main.py list

# 运行单个流水线
python main.py run <pipelineId>
python main.py run regression.linear_regression
python main.py run classification.svc
python main.py run probabilistic.hmm

# 运行一组流水线（按 domain 前缀筛选）
python main.py suite <groupName>
python main.py suite all              # 全部 20 条
python main.py suite classification   # 分类（6 条）
python main.py suite regression       # 回归（4 条）
python main.py suite ensemble         # 集成（4 条）
python main.py suite clustering       # 聚类（2 条）
python main.py suite dimensionality   # 降维（2 条）
python main.py suite probabilistic    # 概率模型（2 条）

# 仅做数据探索——加载数据并打印统计报告，不训练
python main.py analyze <pipelineId>
```

### 1.2 执行流程

```
main()
  ├─ list    → _printPipelineList()
  │             遍历 PIPELINE_REGISTRY → 按 ID 排序 → 逐条打印摘要
  │
  ├─ run     → _runPipeline(id)
  │             PIPELINE_REGISTRY.get(id)  → PipelineSpec
  │             DATASET_REGISTRY.get(datasetId) → DatasetSpec
  │             ensureOptionalDependencies()    → 检查可选依赖
  │             executePipeline(spec, ds)       → 分发到 Runner
  │
  ├─ suite   → _runSuite(group)
  │             按 domain 前缀筛选 pipelineId 列表
  │             逐个 _runPipeline() → 打印 [OK] / [FAIL] / [SKIP]
  │
  └─ analyze → _analyzePipeline(id)
               加载数据 → 构建探索报告 → 终端打印
```

### 1.3 可选依赖处理

部分流水线依赖可选第三方库：

| 依赖 | 涉及的流水线 |
|---|---|
| `hmmlearn` | `probabilistic.hmm` |
| `xgboost` | `ensemble.xgboost` |
| `lightgbm` | `ensemble.lightgbm` |

当可选依赖未安装时：
- CLI 打印 `[SKIP] <pipelineId>: 缺少可选依赖` 而非崩溃。
- `suite` 模式下其他流水线继续正常执行。

---

## 2. 全部流水线速览

### 2.1 分类（Classification）— 6 条

| 流水线 ID | 模型 | 预处理 | 数据特点 | 独有可视化 |
|---|---|---|---|---|
| `classification.logistic_regression` | `LogisticRegression(max_iter=1000)` | `standardScaler` | 线性可分二分类 | 决策边界、ROC |
| `classification.decision_tree` | `DecisionTreeClassifier(max_depth=6)` | `None` | blob 多分类 | 决策边界、树结构 |
| `classification.svc` | `SVC(rbf, probability=True)` | `standardScaler` | 同心圆二分类 | 决策边界、ROC |
| `classification.naive_bayes` | `GaussianNB` | `standardScaler` | Iris 真实数据 | 决策边界 |
| `classification.knn` | `KNeighborsClassifier(5)` | `standardScaler` | 双月牙二分类 | 决策边界 |
| `classification.random_forest` | `RandomForestClassifier(100)` | `None` | 高维多分类 | 特征重要性、决策边界 |

### 2.2 回归（Regression）— 4 条

| 流水线 ID | 模型 | 预处理 | 数据特点 | 独有可视化 |
|---|---|---|---|---|
| `regression.linear_regression` | `LinearRegression` | `None` | 合成线性房价（200 × 3） | 学习曲线、系数对照 |
| `regression.svr` | `SVR(rbf, C=10, ε=0.1)` | `standardScaler` | Friedman1 非线性（200 × 10） | 学习曲线、支持向量数 |
| `regression.decision_tree` | `DecisionTreeRegressor(max_depth=6)` | `None` | California Housing（20640 × 8） | 学习曲线、树结构 |
| `regression.regularization` | `Lasso / Ridge / ElasticNet` | `standardScaler` | diabetes + 共线 + 噪声（442 × 21） | 多模型对比、近零系数 |

### 2.3 聚类（Clustering）— 2 条

| 流水线 ID | 模型 | 预处理 | 数据特点 | 独有可视化 |
|---|---|---|---|---|
| `clustering.kmeans` | `KMeans` | `standardScaler` | 球形多簇 | K 值扫描（惯性曲线） |
| `clustering.dbscan` | `DBSCAN` | `standardScaler` | 双月牙非线性 | ε 扫描、k-距离图 |

### 2.4 降维（Dimensionality）— 2 条

| 流水线 ID | 模型 | 预处理 | 数据特点 | 独有可视化 |
|---|---|---|---|---|
| `dimensionality.pca` | `PCA` | `standardScaler` | 高维低秩合成 | 累计解释方差训练曲线 |
| `dimensionality.lda` | `LDA` | `standardScaler` | Wine 真实数据（13 维 → 2 维） | 分类评估（混淆矩阵 + ROC） |

### 2.5 集成学习（Ensemble）— 4 条

| 流水线 ID | 模型 | 预处理 | 数据特点 | 独有可视化 |
|---|---|---|---|---|
| `ensemble.bagging` | `BaggingClassifier(DT, n=30)` | `standardScaler` | 高噪声双月牙二分类 | 决策边界 |
| `ensemble.gbdt` | `GradientBoostingClassifier` | `standardScaler` | 中等难度多分类 | 特征重要性 |
| `ensemble.xgboost` | `XGBRegressor(n=300, lr=0.05)` | `None` | 回归数据 | 特征重要性 |
| `ensemble.lightgbm` | `LGBMClassifier(n=80)` | `standardScaler` | 高维四分类 | 特征重要性 |

### 2.6 概率模型（Probabilistic）— 2 条

| 流水线 ID | 模型 | 预处理 | 数据特点 | 独有可视化 |
|---|---|---|---|---|
| `probabilistic.em` | `GaussianMixture` | `standardScaler` | GMM 混合数据 | 分量数扫描（BIC/AIC） |
| `probabilistic.hmm` | `CategoricalHMM` | `None` | 离散序列 | 无图形化输出（仅终端日志） |

---

## 3. 回归流水线对比矩阵

四条回归流水线展示了四种典型的配置组合：

| 配置维度 | 线性回归 | SVR | 决策树回归 | 正则化回归 |
|---|---|---|---|---|
| 标准化 | 无 | **有** | 无 | **有** |
| 学习曲线 | **有** | **有** | **有** | 无 |
| 特征重要性 | **有（coef_）** | **无（RBF 核）** | **有** | **有** |
| 树结构 | 无 | 无 | **有** | 无 |
| 多模型 | 无 | 无 | 无 | **有（3 个）** |
| 训练函数行数 | 3 | 2 | ~5 | ~10 |
