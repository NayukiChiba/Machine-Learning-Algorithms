---
title: XGBoost — 训练与预测
outline: deep
---

# 训练与预测

## 本章目标

1. 理解 `pipelines/ensemble/xgboost.py` 的 `run()` 流水线——回归任务下的端到端流程（无标准化、无分层抽样）。
2. 理解 XGBoost 的 `fit()` 训练过程——二阶目标近似 + 显式正则化 + 加权分位数草图。
3. 理解回归预测的输出——连续房价预测值与残差分析。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `run()` | 函数 | 回归流水线编排——6 步串联数据拆分、训练、预测和两项评估 |
| `model.fit(X_train, y_train)` | 方法 | 训练 300 棵二阶近似正则化回归树——列块并行 + 加权分位数草图 |
| `model.predict(X_test)` | 方法 | 300 棵树加权累加——输出连续房价预测值 |
| `plot_residuals(y_test, y_pred, ...)` | 函数 | 绘制预测残差散点图和分布图——回归专用评估 |
| `plot_feature_importance(model, feature_names, ...)` | 函数 | 绘制特征重要性柱状图 |

## 1. 完整流水线流程

### 流程概述

```
xgboost_data.copy()
    │
    ├─ ① X = data.drop(columns=["price"]), y = data["price"]
    ├─ ② feature_names = list(X.columns)
    ├─ ③ X_train, X_test, y_train, y_test = train_test_split(test_size=0.2)
    ├─ ④ model = train_model(X_train, y_train)  # 无标准化，含 ImportError 检查
    ├─ ⑤ y_pred = model.predict(X_test)
    └─ ⑥ 两项评估可视化
```

### 参数速览

| 步骤 | 操作 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| 复制数据 | `xgboost_data.copy()` | 全局 `DataFrame` | 本地 `DataFrame`，`(20640, 9)` | 避免修改全局变量 |
| 分离 X/y | `data.drop(columns=["price"])` + `data["price"]` | `DataFrame` | `(DataFrame, Series)` | 特征 8 列 + 连续目标 1 列 |
| 提取特征名 | `list(X.columns)` | `DataFrame` | `list[str]`，长度 8 | 供特征重要性图表使用 |
| 切分数据 | `train_test_split(test_size=0.2)` | `(X, y)` | `(X_train, X_test, y_train, y_test)` | 16512 训练 / 4128 测试 |
| 训练 | `train_model(X_train, y_train)` | `(DataFrame, Series)` | `XGBRegressor` | 300 棵二阶正则化树 |
| 预测 | `model.predict(X_test)` | `DataFrame`，`(4128, 8)` | `ndarray`，`(4128,)` | 连续房价预测值 |
| 残差图 | `plot_residuals(y_test, y_pred, ...)` | `(Series, ndarray)` | PNG 文件 | 残差散点图 + 分布图 |
| 特征重要性 | `plot_feature_importance(model, feature_names, ...)` | `(model, list)` | PNG 文件 | 8 个特征排序柱状图 |

### 理解重点

- 这是四个集成模型中最简洁的流水线——6 步（vs Bagging 7 步、GBDT 9 步、LightGBM 7 步），少了标准化步骤。
- 与分类集成流水线的关键差异：无 `StandardScaler`、无 `stratify`、无 `predict_proba`、无混淆矩阵、无 ROC。
- 目标列名为 `price`（不是 `label`）——这是回归任务与分类任务在命名上的明确区分。

## 2. 训练细节：`model.fit(X_train, y_train)`

### 训练过程（300 棵树串行，含列块并行）

1. **第 1 棵树**：在原始房价标签上训练——初始预测为训练集均值
2. **第 $m$ 棵树**（$m = 2, \dots, 300$）：计算一阶梯度 $g_i$ 和二阶 Hessian $h_i$（回归下 $h_i=1$），对目标函数做二阶泰勒展开
3. **分裂点搜索**：对 8 个特征分别用加权分位数草图找候选分裂点，计算分裂增益 $\text{Gain} = \frac{1}{2}[\dots] - \gamma$，选最大增益分裂
4. **列采样**：`colsample_bytree=0.9`——每棵树随机选约 7 个特征
5. **行采样**：`subsample=0.9`——每轮随机保留 90% 样本
6. **正则化约束**：$w_j^* = -\frac{G_j}{H_j + \lambda}$（叶子权重 L2 压缩）+ $\gamma$ 门槛检查
7. **学习率收缩**：每棵树的输出乘以 `learning_rate=0.05`

### 参数速览

| 参数名 | 当前取值 | 训练中的作用 |
|---|---|---|
| `n_estimators` | `300` | 串行训练的弱学习器数量 |
| `learning_rate` | `0.05` | 每棵树输出的收缩乘数 |
| `max_depth` | `6` | 每棵树的最大深度——可以分裂 6 次（最多 64 个叶子） |
| `min_child_weight` | `1` | 叶子节点的最小 Hessian 和——回归下等价于最小样本数 1 |
| `subsample` | `0.9` | 行采样比例——每轮随机保留 90% 训练样本 |
| `colsample_bytree` | `0.9` | 列采样比例——每棵树随机选 90% 特征（≈7/8） |
| `gamma` | `0.0` | 分裂最低增益——当前不设门槛 |
| `reg_lambda` | `1.0` | L2 正则化——压缩叶子权重 |
| `reg_alpha` | `0.0` | L1 正则化——当前不启用 |
| `n_jobs` | `-1` | 列块并行——各特征分裂点搜索可并行 |

### 理解重点

- XGBoost 的训练**在概念上**仍是串行 Boosting——但每棵树内部的列块分裂搜索是并行的（`n_jobs=-1`）。
- `reg_lambda=1.0` 使得每片叶子的权重被压缩——$w_j^* = -\frac{G_j}{H_j + 1}$，分母恒加 1 防止权重过大。
- 在回归任务中 $h_i=1$，Hessian 恒为常数——二阶展开的信息增量为零，但闭式解和正则化仍有效。

## 3. 预测细节

### `model.predict(X_test)` — 输出连续值

```
300 棵树加权累加（每棵 × learning_rate）
    → 连续实数（房价预测值，单位：10 万美元）
```

### 参数速览

| 方法 | 输入形状 | 输出形状 | 输出含义 |
|---|---|---|---|
| `predict(X)` | `(n, 8)` | `(n,)` | 连续房价预测值——$\in \mathbb{R}$ |

### 理解重点

- 与分类模型的根本不同：`predict()` 返回连续实数，不是类别标号。
- 没有 `predict_proba()`——回归模型只输出一个标量预测值。
- 预测值 = 训练集初始均值 + $\sum_{m=1}^{300} 0.05 \times f_m(\mathbf{x})$。

## 4. 与 Bagging/GBDT/LightGBM 流水线对比

| 步骤 | Bagging | GBDT | LightGBM | XGBoost |
|---|---|---|---|---|
| 标准化 | 有 | 有 | 有 | **无** |
| 分层抽样 | 有 | 有 | 有 | **无** |
| `predict_proba` | 有（条件检查） | 有 | 有 | **无** |
| 混淆矩阵 | 有 | 有 | 有 | **无** |
| ROC 曲线 | 有（条件可用） | 有 | 有 | **无** |
| 残差图 | 无 | 无 | 无 | **有** |
| 学习曲线 | 无 | 有 | 无 | 无 |

### 理解重点

- XGBoost 流水线与其他集成模型的差异根源于任务类型——回归 vs 分类导致评估手段完全不同。
- 残差图是回归模型的标准诊断——它回答"预测值和真实值的偏差在哪些区域较大、有没有系统偏差"。

## 常见坑

1. 在回归场景下调用 `model.predict_proba()`——`XGBRegressor` 没有此方法，只有 `predict()`。
2. 误以为需要标准化——树模型基于分裂点比较，对特征尺度不变，标准化既非必须也无帮助。
3. 在 `train_test_split` 中传入 `stratify=y`——回归任务的连续目标没有类别可分层。
4. 在缺少 `xgboost` 的环境中直接运行流水线——会触发 `ImportError`。

## 小结

- XGBoost 流水线是最简洁的集成模型流水线——6 步完成数据拆分、训练、预测和两项评估，无标准化、无分层。
- `fit()` 的核心流程：二阶泰勒展开 $g_i + \frac{1}{2}h_i f^2$ → 正则化目标 + 叶子权重闭式解 $w_j^* = -\frac{G_j}{H_j+\lambda}$ → 加权分位数草图 + 列块并行 → 300 棵树串行累加。
- `predict()` 输出连续实数——与分类集成模型的 softmax + argmax 预测路径在本质上不同。
