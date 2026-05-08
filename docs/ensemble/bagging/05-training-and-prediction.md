---
title: Bagging 集成学习 — 训练与预测
outline: deep
---

# 训练与预测

## 本章目标

1. 理解 Bagging 流水线的完整执行流程——从数据拆解到模型预测。
2. 认清 Bagging 作为监督分类算法的流程特征——有训练/测试切分、有 `y_train` 参与训练、有 `predict` 和 `predict_proba`。
3. 理解流水线中每一步的意图和与算法原理的对应关系。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `run()` | 函数 | Bagging 分类端到端流水线入口——串联数据准备、标准化、训练、预测、评估 |
| `train_test_split(..., stratify=y)` | 函数 | 分层训练/测试切分——保证两集合类别比例一致 |
| `StandardScaler` | 类 | Z-score 标准化——`fit_transform` 在训练集、`transform` 在测试集 |
| `model.predict(X_test)` | 方法 | 80 棵树多数投票——输出硬分类标签 |
| `model.predict_proba(X_test)` | 方法 | 80 棵树概率平均——输出软分类概率 |
| `hasattr(model, "predict_proba")` | 函数 | 防御性检查——`BaggingClassifier` 始终支持概率输出 |

## 1. `run()` 流水线总览

Bagging 流水线是一个典型的监督分类流程——与 DBSCAN、KMeans 等无监督算法不同，它包含训练/测试切分步骤。

### 参数速览

适用函数：`run()`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| 无参数 | — | `run()` 无参数——所有配置硬编码在函数体内 | — |
| 返回值 | `None` | 触发完整的 Bagging 训练+预测+可视化流程 | — |

### 示例代码

```python
from pipelines.ensemble.bagging import run

run()
```

或命令行：

```bash
python -m pipelines.ensemble.bagging
```

### 理解重点

- `run()` 是薄流程编排层——每个步骤调用现有模块，本身不包含算法逻辑。
- 流水线中的每一步都是独立可替换的——换数据、换模型、换评估方式只需替换对应组件。
- 与 DBSCAN/KMeans 流水线的关键差异在于**有监督**——必须传入 `y_train` 给 `train_model`。

## 2. 数据准备：复制、拆解、切分

### 参数速览

| 步骤 | 代码 | 意图 |
|---|---|---|
| 复制数据 | `data = bagging_data.copy()` | 避免修改模块级全局变量 |
| 拆解 X/y | `X = data.drop(columns=["label"])`、`y = data["label"]` | 分离特征和标签——监督学习的标准操作 |
| 分层切分 | `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)` | 80% 训练 + 20% 测试，按类别比例分层 |

### 理解重点

- `stratify=y` 确保训练集和测试集中类别 0 和类别 1 的比例一致——对于 `noise=0.35` 的高噪声数据，虽然两个弯月类别数量本就大致均衡，但分层抽样仍是监督分类的标准做法。
- 训练集 400 样本、测试集 100 样本——规模适中，80 个 Bagging 子学习器在秒级完成训练。
- `bagging_data` 来自 `data_generation/__init__.py` 的模块级变量——`copy()` 是防御性工程习惯。

## 3. 标准化：训练集拟合、测试集变换

### 参数速览

| 步骤 | 代码 | 意图 |
|---|---|---|
| 训练集拟合 | `scaler.fit_transform(X_train)` → `X_train_s` | 在训练集上计算 $\mu$ 和 $\sigma$，同时变换 |
| 测试集变换 | `scaler.transform(X_test)` → `X_test_s` | 使用训练集的 $\mu$ 和 $\sigma$ 变换——防止数据泄露 |

### 理解重点

- 标准化对 Bagging 不是必需的——决策树天然不受特征尺度影响。但当前代码保留标准化是为了一致性（其他算法如 SVC、逻辑回归需要标准化）。
- 正确做法是 `fit_transform` 训练集、`transform` 测试集——如果在测试集上 `fit_transform`，会导致信息泄露（测试集的统计量不应该出现在训练阶段）。
- 标准化后的数据 `X_train_s` 和 `X_test_s` 是 `ndarray`（不是 `DataFrame`）——这是 `StandardScaler` 的标准行为。

## 4. 模型训练：`train_model(X_train_s, y_train)`

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train_s` | `ndarray`，形状 `(400, 2)` | 标准化后的训练特征 | `scaler.fit_transform(X_train)` |
| `y_train` | `Series`，形状 `(400,)` | 训练标签 $\{0, 1\}$ | `y_train` |
| 返回值 | `BaggingClassifier` | 已完成 `fit()` 的模型——含 80 棵完全生长的基学习器 | — |

### 理解重点

- `train_model(...)` **必须有 `y_train`**——这是 Bagging 作为监督分类算法的根本标志。与 PCA、KMeans 等无监督算法不同。
- 训练过程中内部并行执行：80 棵树独立 `fit`，利用 `n_jobs=-1` 使用全部 CPU 核心。
- 训练完成后终端打印 `n_estimators`、`max_samples`、`max_features`、`bootstrap` 以及 OOB 得分（4 位小数）。

## 5. 预测：`predict()` 和 `predict_proba()`

### 参数速览

| 方法 | 输入 | 输出 | 机制 |
|---|---|---|---|
| `model.predict(X_test_s)` | `ndarray`，形状 `(100, 2)` | `ndarray`，形状 `(100,)`，取值 $\{0, 1\}$ | 80 棵树硬投票——多数获胜 |
| `model.predict_proba(X_test_s)` | `ndarray`，形状 `(100, 2)` | `ndarray`，形状 `(100, 2)` | 80 棵树软投票——概率取平均 |

### 示例代码

```python
y_pred = model.predict(X_test_s)
# y_pred 形状 (100,)，取值 {0, 1}

if hasattr(model, "predict_proba"):
    y_scores = model.predict_proba(X_test_s)
    # y_scores 形状 (100, 2)，每行两个类别的预测概率
```

### 输出

```text
# predict 输出示例（前 10 个预测）
[1 0 1 0 0 1 1 0 1 0]

# predict_proba 输出示例（前 5 行）
[[0.1  0.9 ]
 [0.7  0.3 ]
 [0.05 0.95]
 [0.85 0.15]
 [0.6  0.4 ]]
```

### 理解重点

- `predict()` 输出硬标签——直接用于混淆矩阵。
- `predict_proba()` 输出概率——用于 ROC 曲线。概率输出比硬标签更有信息量（不仅知道预测类别，还知道置信度）。
- `hasattr(model, "predict_proba")` 是防御性检查——虽然 `BaggingClassifier` 始终支持概率输出（只要基学习器支持），但检查避免了因 API 变化导致的崩溃。
- 测试集 100 个样本——规模足够让混淆矩阵和 ROC 曲线展示有意义的统计信息。

## 6. 评估触发：混淆矩阵 + ROC 曲线

### 参数速览

| 步骤 | 触发条件 | 输入 | 输出 |
|---|---|---|---|
| 混淆矩阵 | **始终** | `y_test` + `y_pred` | `outputs/bagging/confusion_matrix.png` |
| ROC 曲线 | **条件**：`predict_proba` 可用 | `y_test` + `y_scores` | `outputs/bagging/roc_curve.png` |

### 理解重点

- 混淆矩阵是 Bagging 分类评估的必选项——无论 `predict_proba` 是否可用，都能用硬标签生成。
- ROC 曲线是条件可选项——需要概率输出。对 Bagging 而言这个条件始终满足，但条件判断是防御性工程习惯。
- 与 DBSCAN 流水线的对比：DBSCAN 的 `plot_clusters` 只做可视化，没有硬指标——Bagging 同时有混淆矩阵（硬标签）和 ROC 曲线（软概率），评估更全面。

## 完整流程总结

```
bagging_data.copy()
    │
    ├─ X = data.drop(columns=["label"])
    ├─ y = data["label"]
    │
    ├─ train_test_split(test_size=0.2, stratify=y)
    │   ├─ X_train (400, 2)、y_train (400,)
    │   └─ X_test (100, 2)、y_test (100,)
    │
    ├─ StandardScaler
    │   ├─ X_train_s = scaler.fit_transform(X_train)
    │   └─ X_test_s = scaler.transform(X_test)
    │
    ├─ model = train_model(X_train_s, y_train)
    │   └─ 终端打印: n_estimators, max_samples, max_features, bootstrap, OOB 得分
    │
    ├─ y_pred = model.predict(X_test_s)           → 混淆矩阵
    └─ y_scores = model.predict_proba(X_test_s)   → ROC 曲线（条件）
```

## 常见坑

1. 混淆 Bagging 的流程与无监督算法——Bagging 需要 `y_train`，有训练/测试切分，有 `predict` 和 `predict_proba`。
2. 在测试集上 `fit_transform` 而非 `transform`——这是数据泄露的经典错误。
3. 忘记 `stratify=y`——不平衡数据上可能导致测试集中某类别缺失。
4. 忽略 `hasattr(model, "predict_proba")` 防御性检查的作用——虽然 Bagging 始终支持，但换用其他模型时这个检查可能是必要的。

## 小结

- Bagging 流水线是一个标准的监督分类流程：数据拆解 → 分层切分 → 训练集拟合标准化/测试集变换 → 训练（含 OOB 估计） → 硬预测/软概率 → 混淆矩阵/ROC 曲线。
- 与 DBSCAN/KMeans 等无监督流水线的核心差异：有 `y_train`、有 `train_test_split`、有混淆矩阵和 ROC 曲线（而非聚类散点图）。
- `run()` 是薄编排层——每步调用现有模块，自身不含算法逻辑。
