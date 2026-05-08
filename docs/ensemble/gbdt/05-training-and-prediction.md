---
title: GBDT 梯度提升树 — 训练与预测
outline: deep
---

# 训练与预测

## 本章目标

1. 理解 GBDT 流水线的完整执行流程——从数据拆解到模型预测。
2. 认清 GBDT 作为监督多分类算法的流程特征——有训练/测试切分、有 `y_train` 参与训练、有 `predict` 和 `predict_proba`。
3. 理解流水线中每一步的意图和与算法原理的对应关系。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `run()` | 函数 | GBDT 多分类端到端流水线入口——串联数据准备、标准化、训练、预测和四项评估 |
| `train_test_split(..., stratify=y)` | 函数 | 分层训练/测试切分——保证三个类别的比例一致 |
| `StandardScaler` | 类 | Z-score 标准化——`fit_transform` 在训练集、`transform` 在测试集 |
| `model.predict(X_test)` | 方法 | 200 棵树加权累加后 softmax 取最大——输出硬分类标签 |
| `model.predict_proba(X_test)` | 方法 | softmax 概率输出——直接用于多分类 ROC 曲线 |

## 1. `run()` 流水线总览

GBDT 流水线是一个典型的有监督多分类流程——与 Bagging 结构相似，但 GBDT 多了特征重要性（`feature_importances_`）和学习曲线两项评估。

### 参数速览

适用函数：`run()`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| 无参数 | — | `run()` 无参数——所有配置硬编码在函数体内 | — |
| 返回值 | `None` | 触发完整的 GBDT 训练+预测+四项评估+可视化流程 | — |

### 示例代码

```python
from pipelines.ensemble.gbdt import run

run()
```

或命令行：

```bash
python -m pipelines.ensemble.gbdt
```

### 理解重点

- `run()` 是薄流程编排层——每个步骤调用现有模块，本身不包含算法逻辑。
- 与 Bagging 流水线的关键差异在于评估环节：GBDT 有 4 项评估输出（混淆矩阵 + ROC + 特征重要性 + 学习曲线），Bagging 有 2 项（混淆矩阵 + ROC + OOB 日志）。
- `feature_names = list(X.columns)` 是 GBDT 流水线特有的步骤——为特征重要性图表提供 x 轴标注。

## 2. 数据准备：复制、拆解、切分

### 参数速览

| 步骤 | 代码 | 意图 |
|---|---|---|
| 复制数据 | `data = gbdt_data.copy()` | 避免修改模块级全局变量 |
| 拆解 X/y | `X = data.drop(columns=["label"])`、`y = data["label"]` | 分离 8 个特征和 3 分类标签 |
| 提取特征名 | `feature_names = list(X.columns)` | 供特征重要性图表使用——`['x1', ..., 'x8']` |
| 分层切分 | `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)` | 80% 训练 + 20% 测试，按三类别比例分层 |

### 理解重点

- `feature_names` 是 GBDT 流水线独有的中间变量——Bagging 没有特征重要性图表，所以不需要这个步骤。
- `stratify=y` 对三分类场景尤其重要——避免某个类别在训练集或测试集中比例失衡。
- 8 个特征中只有 4 个有效——训练完成后特征重要性图表会揭示哪些特征"说了算"。

## 3. 标准化：训练集拟合、测试集变换

### 参数速览

| 步骤 | 代码 | 意图 |
|---|---|---|
| 训练集拟合 | `scaler.fit_transform(X_train)` → `X_train_s` | 在训练集上计算 8 维 $\mu$ 和 $\sigma$，同时变换 |
| 测试集变换 | `scaler.transform(X_test)` → `X_test_s` | 使用训练集的 $\mu$ 和 $\sigma$ 变换——防止数据泄露 |

### 理解重点

- 标准化对 GBDT 不是必需的——决策树天然不受特征尺度影响。但保留标准化是为了流水线一致性。
- 正确做法是 `fit_transform` 训练集、`transform` 测试集——如果在测试集上 `fit_transform`，会导致信息泄露。

## 4. 模型训练：`train_model(X_train_s, y_train)`

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train_s` | `ndarray`，形状 `(400, 8)` | 标准化后的训练特征 | `scaler.fit_transform(X_train)` |
| `y_train` | `Series`，形状 `(400,)` | 训练标签 $\{0, 1, 2\}$ | `y_train` |
| 返回值 | `GradientBoostingClassifier` | 已完成 `fit()` 的模型——含 200 × 3 = 600 棵回归树 | — |

### 理解重点

- `train_model(...)` **必须有 `y_train`**——GBDT 比 Bagging 更依赖标签，因为每个阶段的拟合目标（负梯度）由标签和当前预测共同决定。
- 训练过程是严格串行的——第 $m$ 棵树的训练依赖前 $m-1$ 棵的输出，无法并行化（与 Bagging 的 `n_jobs=-1` 形成对比）。
- 训练完成后终端打印 `n_estimators`、`learning_rate`、`max_depth`、`subsample`。

## 5. 预测：`predict()` 和 `predict_proba()`

### 参数速览

| 方法 | 输入 | 输出 | 机制 |
|---|---|---|---|
| `model.predict(X_test_s)` | `ndarray`，形状 `(100, 8)` | `ndarray`，形状 `(100,)`，取值 $\{0, 1, 2\}$ | 200 棵树加权累加 → softmax → argmax |
| `model.predict_proba(X_test_s)` | `ndarray`，形状 `(100, 8)` | `ndarray`，形状 `(100, 3)` | 200 棵树加权累加 → softmax → 三类概率 |

### 示例代码

```python
y_pred = model.predict(X_test_s)
# y_pred 形状 (100,)，取值 {0, 1, 2}

y_scores = model.predict_proba(X_test_s)
# y_scores 形状 (100, 3)，每行三个类别的 softmax 概率
```

### 输出

```text
# predict 输出示例（前 10 个预测）
[1 0 2 1 0 1 2 1 0 0]

# predict_proba 输出示例（前 3 行）
[[0.15 0.72 0.13]
 [0.81 0.10 0.09]
 [0.05 0.20 0.75]]
```

### 理解重点

- `predict()` 输出硬标签——200 棵树加权累加后取 softmax 最大概率对应的类别。
- `predict_proba()` 输出 softmax 概率——每行 3 个值之和为 1。直接用于多分类 ROC 曲线（one-vs-rest）。
- GBDT 流水线没有 `hasattr(model, "predict_proba")` 检查——直接调用，因为 `GradientBoostingClassifier` 始终支持概率输出。

## 6. 评估触发：四项输出

### 参数速览

| 步骤 | 触发条件 | 输入 | 输出 |
|---|---|---|---|
| 混淆矩阵 | **始终** | `y_test` + `y_pred` | `outputs/gbdt/confusion_matrix.png` |
| ROC 曲线 | **始终** | `y_test` + `y_scores` | `outputs/gbdt/roc_curve.png` |
| 特征重要性 | **始终** | `model` + `feature_names` | `outputs/gbdt/feature_importance.png` |
| 学习曲线 | **始终** | `GradientBoostingClassifier(...)` + `X_train_s` + `y_train` | `outputs/gbdt/learning_curve.png` |

### 理解重点

- 四项评估全部始终触发——没有条件判断（不像 Bagging 的 `hasattr` 检查）。
- 特征重要性是 GBDT 独有的评估——Bagging 没有此项。
- 学习曲线通过额外实例化一个 `GradientBoostingClassifier` 生成——不是用训练好的 `model`，而是用 `sklearn.model_selection.learning_curve` 进行交叉验证。

## 完整流程总结

```
gbdt_data.copy()
    │
    ├─ X = data.drop(columns=["label"])
    ├─ y = data["label"]
    ├─ feature_names = list(X.columns)
    │
    ├─ train_test_split(test_size=0.2, stratify=y)
    │   ├─ X_train (400, 8)、y_train (400,)
    │   └─ X_test (100, 8)、y_test (100,)
    │
    ├─ StandardScaler
    │   ├─ X_train_s = scaler.fit_transform(X_train)
    │   └─ X_test_s = scaler.transform(X_test)
    │
    ├─ model = train_model(X_train_s, y_train)
    │   └─ 终端打印: n_estimators, learning_rate, max_depth, subsample
    │
    ├─ y_pred = model.predict(X_test_s)               → 混淆矩阵
    ├─ y_scores = model.predict_proba(X_test_s)       → ROC 曲线
    ├─ plot_feature_importance(model, feature_names)  → 特征重要性
    └─ plot_learning_curve(..., X_train_s, y_train)   → 学习曲线
```

## 常见坑

1. 混淆 GBDT 的串行训练与 Bagging 的并行训练——GBDT 不能用 `n_jobs=-1` 并行训练多棵树（但 scikit-learn 在某些版本中支持并行化每棵树内部的分裂搜索）。
2. 在测试集上 `fit_transform` 而非 `transform`——数据泄露的经典错误。
3. 忘记 `feature_names` 的提取——没有特征名，特征重要性图表只有位置索引，可读性大打折扣。
4. 混淆 `predict` 的机制——GBDT 是加权累加后 softmax，不是投票。

## 小结

- GBDT 流水线是一个有监督多分类流程：数据拆解 → 分层切分 → 训练集拟合标准化/测试集变换 → 串行梯度提升训练 → 硬预测/软概率 → 四项评估（混淆矩阵 + ROC + 特征重要性 + 学习曲线）。
- 与 Bagging 流水线的核心差异：（1）训练是串行而非并行；（2）有 `feature_names` 提取步骤；（3）有特征重要性和学习曲线两项额外评估；（4）没有 `hasattr` 条件判断。
- `run()` 是薄编排层——每步调用现有模块，自身不含算法逻辑。
