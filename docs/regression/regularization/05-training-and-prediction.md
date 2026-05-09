---
title: 正则化回归 — 训练与预测
outline: deep
---

# 训练与预测

## 本章目标

1. 理解正则化回归流水线的完整执行顺序——从数据加载到残差图输出。
2. 理解三种模型的训练过程——坐标下降（Lasso/EN）vs 闭式解（Ridge）。
3. 理解预测阶段的统一循环——三模型共用同一份标准化测试数据。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `loadRegularizationDataset()` | 方法 | 加载 diabetes + 共线 + 噪声特征——返回 `(442, 22)` DataFrame |
| `StandardScaler` | 预处理 | Z-score 标准化——正则化回归训练前的强制步骤 |
| `trainRegularizationModels(...)` | 函数 | 构建并 `fit` 三个正则化模型——返回模型字典 |
| `model.predict(X_test_s)` | 方法 | 对标准化测试集做回归预测——$\hat{y} = \mathbf{X}\mathbf{w} + b$ |
| `plot_residuals(...)` | 函数 | 为每个模型绘制残差诊断图 |

## 1. 完整流水线流程

### 流程概述

```
loadRegularizationDataset()
    │
    ├─ ① X = data.drop(columns=["price"]), y = data["price"]
    ├─ ② X_train, X_test, y_train, y_test = train_test_split(test_size=0.2)
    ├─ ③ scaler = StandardScaler(); X_train_s = scaler.fit_transform(X_train)
    ├─ ④ X_test_s = scaler.transform(X_test)
    ├─ ⑤ models = trainRegularizationModels(X_train_s, y_train)
    ├─ ⑥ for name, model in models.items():
    │       y_pred = model.predict(X_test_s)
    │       plot_residuals(y_test, y_pred, ...)
    └─ ⑦ plot_feature_importance(model, feature_names)  — 对每个模型
```

### 参数速览

| 步骤 | 操作 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| 加载数据 | `loadRegularizationDataset` | — | `DataFrame`，`(442, 22)` | diabetes + 共线 + 噪声 |
| 特征标签拆分 | `drop` + 列选择 | `DataFrame` | `X(442,21)`, `y(442,)` | 标签列 `price` |
| 数据切分 | `train_test_split` | `X`, `y` | `X_train(353,21)`, `X_test(89,21)` | `test_size=0.2` |
| 标准化 | `StandardScaler` | `X_train`, `X_test` | `X_train_s`, `X_test_s` | **正则化回归必需** |
| 训练 | `trainRegularizationModels` | `X_train_s`, `y_train` | `dict[lasso/ridge/elasticnet]` | 一次训练三个模型 |
| 预测 | `model.predict` | `X_test_s` | 各模型 `y_pred(89,)` | 循环三次 |
| 残差图 | `plot_residuals` | `y_test`, `y_pred` | PNG 图像 | 每个模型一张 |
| 特征重要性 | `plot_feature_importance` | `model`, `feature_names` | PNG 图像 | 系数柱状图 |

### 理解重点

- 正则化回归流水线比线性回归多两步——标准化（③④）和特征重要性图（⑦），但没有学习曲线。
- 标准化在切分**之后**执行——先在训练集上 `fit_transform`，再对测试集仅 `transform`。
- 三个模型共享完全相同的训练/测试数据——对比的公平性由统一的切分和标准化保证。

## 2. 标准化：正则化回归训练的关键前置步骤

### 参数速览

适用 API：`StandardScaler().fit_transform(X_train)` / `StandardScaler().transform(X_test)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `ndarray(353, 21)` | 未标准化的训练特征 | 原始 diabetes + 构造特征 |
| `X_train_s` | `ndarray(353, 21)` | 标准化后——每列均值 0、标准差 1 | — |
| `X_test_s` | `ndarray(89, 21)` | 使用训练集统计量标准化 | — |

### 示例代码

```python
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)  # 计算 μ, σ 并变换
X_test_s = scaler.transform(X_test)         # 仅变换——使用训练集的 μ, σ
```

### 理解重点

- 正则化回归**必须标准化**——L1/L2 惩罚对系数量级敏感。未标准化的特征会导致惩罚不均匀：量级大的特征被过度惩罚，量级小的特征惩罚不足。
- 这是正则化回归与线性回归、决策树回归最关键的工程差异——两者在 `PipelineSpec` 中的预处理分别为 `None` 和 `"standardScaler"`。
- `fit_transform` vs `transform` 的区别是防止数据泄露——测试集的均值和标准差不应影响模型。

## 3. 训练细节：三种算法，三种求解路径

### 参数速览

| 模型 | 求解算法 | 是否有闭式解 | 是否需要迭代 | 收敛判断 |
|---|---|---|---|---|
| Ridge | SVD / 闭式解 | **是**——$\mathbf{w}^* = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$ | 否 | 不需要 |
| Lasso | 坐标下降 | 否 | **是**——`max_iter=10000` | 对偶间隙 < `tol` |
| ElasticNet | 坐标下降 | 否 | **是**——`max_iter=10000` | 对偶间隙 < `tol` |

### 理解重点

- Ridge 训练是**瞬时**的——21 维特征的闭式解计算量极小。
- Lasso 和 ElasticNet 是**迭代**的——坐标下降逐维度优化，`max_iter=10000` 确保充分收敛。
- 三种模型都保证找到全局最优解——Ridge 的目标函数是强凸的，Lasso/EN 是凸的（坐标下降收敛到全局最优）。

## 4. 预测细节：统一的线性公式

三种模型的预测公式完全相同：

$$
\hat{y} = \mathbf{X}_{\text{test}} \mathbf{w} + b
$$

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_test_s` | `ndarray(89, 21)` | 标准化后的测试特征 | — |
| `model.coef_` | `ndarray(21,)` | 各模型的系数向量——形态因正则化策略而异 | Lasso：部分零 / Ridge：全非零 |
| `model.intercept_` | `float` | 截距项——不受正则化惩罚 | — |
| `y_pred` | `ndarray(89,)` | 预测值 | — |

### 理解重点

- 预测公式与普通线性回归完全一致——都是 $\mathbf{X}\mathbf{w} + b$ 的矩阵乘法。
- 差异不在预测公式，而在 $\mathbf{w}$ 的形态——Lasso 的 $\mathbf{w}$ 有零分量，Ridge 的 $\mathbf{w}$ 整体收缩。
- 预测复杂度 $O(N_{\text{test}} \cdot d) = O(89 \times 21)$——几乎瞬时。

## 5. 多模型预测循环

### 示例代码

```python
models = trainRegularizationModels(X_train_s, y_train, randomState=42)

for name, model in models.items():
    y_pred = model.predict(X_test_s)
    plot_residuals(
        y_test, y_pred,
        title=f"{name} 残差分析",
        dataset_name="regularization",
        model_name=name,
    )
    plot_feature_importance(
        model, feature_names,
        title=f"{name} 系数",
        dataset_name="regularization",
        model_name=name,
    )
```

### 理解重点

- 循环结构是正则化回归流水线的独特特征——其他回归模型只训练和评估一个模型。
- 每个模型都生成独立的残差图和系数图——便于横向对比三种正则化策略的差异。
- `model_name=name` 使输出文件自动按模型名命名（`lasso_residual.png`、`ridge_coefficients.png` 等）。

## 6. 正则化回归 vs 线性回归 vs 决策树回归 训练对比

| 训练维度 | 线性回归 | 决策树回归 | 正则化回归 |
|---|---|---|---|
| 数据 | 合成 `(200, 3)` | 真实 `(20640, 8)` | **真实+构造 `(442, 21)`** |
| 标准化 | 无 | 无 | **`StandardScaler`——强制** |
| 训练算法 | SVD 闭式解 | CART 贪心递归 | **坐标下降（Lasso/EN）+ 闭式解（Ridge）** |
| 训练模型数 | 1 | 1 | **3（并行训练）** |
| 收敛判断 | 不需要 | `max_depth` 等早停 | **`max_iter=10000`（Lasso/EN）** |
| 预测 | $\hat{y} = \mathbf{X}\mathbf{w} + b$ | 沿树走到叶子返回均值 | **$\hat{y} = \mathbf{X}\mathbf{w} + b$（同线性回归）** |
| 评估可视化 | 残差图 + 学习曲线 | 残差图 + 特征重要性 + 学习曲线 + 树结构 | **残差图 + 特征重要性——无学习曲线** |

## 常见坑

1. 训练时传 `X_train_s`，预测时却传了未标准化的 `X_test`——预测结果会严重偏离。
2. 忘记标准化必须在切分之后——先标准化再切分会导致测试集信息泄露。
3. 期待正则化回归也有学习曲线——当前 `PipelineSpec` 中学可视化列表为 `[]`，无学习曲线。
4. 以为 Lasso 和 Ridge 的 `alpha` 可以直接比较——`alpha=0.15`（Lasso）和 `alpha=2.0`（Ridge）量级不同，因为 L1 和 L2 的惩罚尺度不同。

## 小结

- 正则化回归流水线为 8 步：加载 → 拆分 → 切分 → 标准化 → 训练三模型 → 循环预测 → 残差图 → 系数图。
- 标准化是正则化回归区别于线性回归和决策树回归的最关键工程差异——惩罚项对尺度敏感。
- 训练阶段一次产出三个模型——Ridge 用闭式解瞬间完成，Lasso/EN 用坐标下降迭代求解。
- 预测阶段与线性回归完全相同（$\mathbf{X}\mathbf{w} + b$）——差异在 $\mathbf{w}$ 的形态而非预测公式。
