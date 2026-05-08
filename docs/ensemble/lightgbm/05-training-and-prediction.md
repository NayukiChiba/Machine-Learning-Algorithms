---
title: LightGBM — 训练与预测
outline: deep
---

# 训练与预测

## 本章目标

1. 理解 `pipelines/ensemble/lightgbm.py` 的 `run()` 流水线——从数据加载到评估的完整 7 步流程。
2. 理解 LightGBM 的 `fit()` 训练过程与 GBDT 的本质区别——Leaf-wise 生长、直方图加速、行/列双重采样。
3. 理解 `predict()` 和 `predict_proba()` 的输出——多类别 softmax 聚合。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `run()` | 函数 | 端到端流水线编排——7 步串联数据准备、标准化、训练、预测和三项评估 |
| `model.fit(X_train_s, y_train)` | 方法 | 串行训练 300 棵 Leaf-wise 直方图回归树——行/列双采样 + 直方图加速 |
| `model.predict(X_test_s)` | 方法 | 300 棵树加权累加 → softmax → argmax——输出类别 $\{0, 1, 2, 3\}$ |
| `model.predict_proba(X_test_s)` | 方法 | 300 棵树加权累加 → softmax——输出 4 类概率分布 |
| `StandardScaler` | 类 | Z-score 标准化——`fit_transform` 在训练集计算统计量，`transform` 在测试集使用相同统计量 |

## 1. 完整流水线流程

### 流程概述

```
lightgbm_data.copy()
    │
    ├─ ① X = data.drop(columns=["label"]), y = data["label"]
    ├─ ② feature_names = list(X.columns)
    ├─ ③ X_train, X_test, y_train, y_test = train_test_split(test_size=0.2, stratify=y)
    ├─ ④ X_train_s = scaler.fit_transform(X_train), X_test_s = scaler.transform(X_test)
    ├─ ⑤ model = train_model(X_train_s, y_train)  # 含 ImportError 检查
    ├─ ⑥ y_pred = model.predict(X_test_s)
    └─ ⑦ 三项可视化评估
```

### 参数速览

| 步骤 | 操作 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| 复制数据 | `lightgbm_data.copy()` | 全局 `DataFrame` | 本地 `DataFrame`，`(1000, 21)` | 避免修改全局变量 |
| 分离 X/y | `data.drop(columns=["label"])` + `data["label"]` | `DataFrame` | `(DataFrame, Series)` | 特征 20 列 + 标签 1 列 |
| 提取特征名 | `list(X.columns)` | `DataFrame` | `list[str]` | 供特征重要性图表使用 |
| 切分数据 | `train_test_split(test_size=0.2, stratify=y)` | `(X, y)` | `(X_train, X_test, y_train, y_test)` | 800 训练 / 200 测试 |
| 标准化 | `scaler.fit_transform(X_train)` + `transform(X_test)` | `(DataFrame, DataFrame)` | `(ndarray, ndarray)` | 训练集计算 $\mu$/$\sigma$，测试集应用 |
| 训练 | `train_model(X_train_s, y_train)` | `(ndarray, Series)` | `LGBMClassifier` | 300 棵直方图树串行训练——使用 `n_jobs=-1` 特征级并行 |
| 预测 | `model.predict(X_test_s)` | `ndarray`，`(200, 20)` | `ndarray`，`(200,)` | 类别 $\{0, 1, 2, 3\}$ |
| 概率输出 | `model.predict_proba(X_test_s)` | `ndarray`，`(200, 20)` | `ndarray`，`(200, 4)` | 每类 softmax 概率 |

### 理解重点

- 与 GBDT 流水线的关键差异：步骤 5 的训练内部使用了 `num_leaves`（非 `max_depth`）控制复杂度，训练速度显著更快。
- 与 Bagging 流水线的关键差异：步骤 2（`feature_names`）是 Boosting 系列独有的——GBDT 和 LightGBM 都有特征重要性评估。
- 所有集成分类流水线的数据准备部分完全一致——`copy()` → 分离 X/y → 分层切分 → 训练集拟合标准化器。

## 2. 训练细节：`model.fit(X_train_s, y_train)`

LightGBM 的 `fit()` 在概念上与 GBDT 一致（加法模型 + 负梯度），但实现了多项工程优化。

### 训练过程（300 棵树串行）

1. **第 1 棵树**：在原始标签上训练——粗糙的初始分界面
2. **第 $m$ 棵树**（$m = 2, \dots, 300$）：拟合前 $m-1$ 棵树的负梯度（多类对数损失梯度），Leaf-wise 生长到最多 `num_leaves=31` 个叶子
3. **直方图分桶**：将 20 维连续特征离散化为 255 个 bins——分裂点搜索从 $O(N)$ 降到 $O(\#\text{bins})$
4. **行采样**：`subsample=0.9`——每轮随机保留 90% 训练样本
5. **列采样**：`colsample_bytree=0.9`——每棵树随机选择 18/20 个特征，增强树间多样性
6. **学习率收缩**：每棵树的输出乘以 `learning_rate=0.05`

### 参数速览

| 参数名 | 当前取值 | 训练中的作用 |
|---|---|---|
| `n_estimators` | `300` | 串行训练的弱学习器数量——步数更多但每步更小 |
| `learning_rate` | `0.05` | 每棵树输出的收缩乘数——防止单棵树修正过猛 |
| `num_leaves` | `31` | 每棵树的最大叶子数——Leaf-wise 生长的复杂度上限 |
| `max_depth` | `-1` | 不限制深度——Leaf-wise 生长由 `num_leaves` 控制复杂度 |
| `subsample` | `0.9` | 行采样比例——每轮迭代随机保留 90% 的训练样本 |
| `colsample_bytree` | `0.9` | 列采样比例——每棵树随机选择 90% 的特征 |
| `n_jobs` | `-1` | 直方图构建和特征扫描级并行——非基学习器级并行 |
| `random_state` | `42` | 保证采样和训练可复现 |

### 理解重点

- LightGBM 的训练**在概念上**仍是串行 Boosting——第 $m$ 棵树依赖前 $m-1$ 棵树的结果。
- **直方图加速**是 LightGBM 训练快于 sklearn GBDT 的核心原因——分割点搜索从排序复杂度降到桶查找。
- `n_jobs=-1` 在 LightGBM 中是安全的——它并行的是直方图构建和特征扫描，而非基学习器训练。

## 3. 预测细节

### `model.predict(X_test_s)` — 硬预测

```
300 棵树加权累加（每类一个分数）
    → softmax（分数转为概率分布）
    → argmax（概率最大的类）
    → {0, 1, 2, 3}
```

### `model.predict_proba(X_test_s)` — 软预测

```
300 棵树加权累加（每类一个分数）
    → softmax（分数转为概率分布）
    → [p₀, p₁, p₂, p₃]，∑p = 1.0
```

### 参数速览

| 方法 | 输入形状 | 输出形状 | 输出含义 |
|---|---|---|---|
| `predict(X)` | `(n, 20)` | `(n,)` | 预测类别标号——$\{0, 1, 2, 3\}$ |
| `predict_proba(X)` | `(n, 20)` | `(n, 4)` | 每类 softmax 概率——4 列和恒为 1.0 |

### 理解重点

- LightGBM 的预测接口与 sklearn `GradientBoostingClassifier` 完全兼容——`predict` 返回类别，`predict_proba` 返回概率。
- 与 Bagging 的投票聚合不同——Bagging 用等权投票（每棵树一票），LightGBM 用加权累加（每棵树 × 学习率 + softmax）。
- `predict_proba` 在 LightGBM 中始终可用——流水线未使用 `hasattr` 条件检查。

## 4. 标准化：训练/测试分离

### 参数速览

| 操作 | 方法 | 数据 | 目的 |
|---|---|---|---|
| 训练集 | `scaler.fit_transform(X_train)` | 训练集 `(800, 20)` | 计算 $\mu_j$ 和 $\sigma_j$，同时变换 |
| 测试集 | `scaler.transform(X_test)` | 测试集 `(200, 20)` | 使用训练集的 $\mu_j$ 和 $\sigma_j$ 变换 |

### 理解重点

- `fit_transform` 在训练集上**同时**计算统计量和变换——一步完成。
- `transform` 在测试集上**只**应用变换——使用训练集的统计量，防止信息泄露。
- LightGBM 基于直方图的树本身对特征缩放不敏感——但标准化保持与 GBDT 和 Bagging 流水线的一致性。

## 常见坑

1. 在缺少 `lightgbm` 的环境中直接运行流水线——会抛出 `ImportError`，需先 `pip install lightgbm`。
2. 误以为 `max_depth=-1` 表示树可以无限深——`num_leaves=31` 已限制最大叶子数，但单叶可能很深。
3. 在测试集上使用 `fit_transform` 而非 `transform`——标准信息泄露。
4. 忽略 `stratify=y` 的重要性——四分类数据中某个类别可能在测试集中缺失。

## 小结

- LightGBM 流水线的 7 步流程与 Bagging/GBDT 共享相同的数据准备段（切分 + 标准化），差异在训练和评估阶段。
- `fit()` 的核心流程：串行训练 300 棵 Leaf-wise 直方图树，每棵树拟合前序负梯度，由 `num_leaves=31` 控制复杂度，行/列双采样增强多样性，`learning_rate=0.05` 收缩修正幅度。
- `predict()` 和 `predict_proba()` 的接口与 sklearn 完全兼容——加权累加 → softmax → 类别/概率。
