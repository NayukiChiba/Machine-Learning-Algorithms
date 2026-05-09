---
title: 线性回归 — 训练与预测
outline: deep
---

# 训练与预测

## 本章目标

1. 理解线性回归流水线的完整执行顺序——从数据加载到残差图和学习曲线输出。
2. 理解 OLS 的训练过程——SVD 闭式求解，无需迭代，无收敛判断。
3. 理解 `predict` 的预测方式——简单的矩阵乘法 $\hat{y} = \mathbf{X}\mathbf{w} + b$。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `trainLinearRegressionModel(...)` | 函数 | 构建并训练线性回归模型——基于 SVD 的闭式求解 |
| `model.fit(X_train, y_train)` | 方法 | 求解 $\min_{\mathbf{w},b} \|\mathbf{y} - \mathbf{X}\mathbf{w} - b\|^2$——一次计算完成 |
| `model.predict(X_test)` | 方法 | 对测试集做矩阵乘法预测——$\hat{y} = \mathbf{X}\mathbf{w} + b$ |
| `plot_residuals(...)` | 函数 | 绘制预测-真实散点图 + 残差分布图 |
| `plot_learning_curve(...)` | 函数 | 绘制训练/验证 R² 随样本量变化的曲线 |

## 1. 完整流水线流程

### 流程概述

```
loadLinearRegressionDataset()
    │
    ├─ ① X = data.drop(columns=["price"]), y = data["price"]
    ├─ ② X_train, X_test, y_train, y_test = train_test_split(test_size=0.2)
    ├─ ③ model = trainLinearRegressionModel(X_train, y_train)
    ├─ ④ y_pred = model.predict(X_test)
    ├─ ⑤ plot_residuals(y_test, y_pred)
    └─ ⑥ plot_learning_curve(LinearRegression(), X_train, y_train, scoring="r2")
```

### 参数速览

| 步骤 | 操作 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| 加载数据 | `loadLinearRegressionDataset` | — | `DataFrame`，`(200, 4)` | 手工合成线性房价 |
| 特征标签拆分 | `drop` + 列选择 | `DataFrame` | `X` `(200, 3)`、`y` `(200,)` | 标签列 `price` |
| 数据切分 | `train_test_split` | `X`、`y` | `X_train` `(160, 3)`、`X_test` `(40, 3)` | `test_size=0.2`，无标准化 |
| 训练 | `trainLinearRegressionModel` | `X_train`、`y_train` | `LinearRegression` | SVD 闭式求解——瞬间完成 |
| 预测 | `model.predict` | `X_test` | `y_pred` `(40,)` | 矩阵乘法 |
| 残差图 | `plot_residuals` | `y_test`、`y_pred` | PNG 图像 | 误差分布诊断 |
| 学习曲线 | `plot_learning_curve` | 新 `LinearRegression()`、`X_train`、`y_train` | PNG 图像 | 样本量-得分趋势 |

### 理解重点

- 这是本仓库**最简流水线**——仅 6 步，无标准化、无特征重要性、无树结构，聚焦于系数解释和残差诊断。
- 训练步骤耗时极短（毫秒级）——3 个特征 × 160 样本的 SVD 求解计算量极小。
- 与决策树回归流水线的对比：决策树多出特征重要性和树结构图两步，训练为贪心递归而非闭式求解。

## 2. 训练细节：SVD 闭式求解

### 算法流程

```
输入 X_train (160, 3), y_train (160,)
    ↓
① 构建设计矩阵: X̃ = [1, X_train] → (160, 4)
② 对 X̃ 做奇异值分解: X̃ = U Σ V^T
③ 计算: w̃* = V Σ^{-1} U^T y_train
④ 返回: coef_ = w̃*[1:], intercept_ = w̃*[0]
```

### 理解重点

- scikit-learn 的 `LinearRegression` 使用 `scipy.linalg.lstsq`（基于 SVD 或 QR 分解）求解——比直接计算 $(\mathbf{X}^T\mathbf{X})^{-1}$ 的数值稳定性更好。
- 训练是**一次性**的——没有迭代、没有收敛判断、没有 `n_iter` 或 `tol` 参数。
- 这是 OLS 与所有迭代式训练算法（EM、Baum-Welch、梯度下降）的根本区别——OLS 保证找到全局最优解，且一步到位。

## 3. 预测细节：矩阵乘法

对测试样本矩阵 $\mathbf{X}_{\text{test}}$：

$$
\hat{\mathbf{y}} = \mathbf{X}_{\text{test}} \mathbf{w} + b = \tilde{\mathbf{X}}_{\text{test}} \tilde{\mathbf{w}}
$$

### 理解重点

- 预测完全不涉及训练数据——模型参数 $\mathbf{w}$ 和 $b$ 已经固化在 `coef_` 和 `intercept_` 中。
- 预测复杂度为 $O(N_{\text{test}} \cdot d) = O(40 \times 3)$——几乎瞬时。
- 与决策树回归的预测对比：线性回归做矩阵乘法（全局统一公式），决策树沿树走到叶子（局部 if-else 路径）。

## 4. 与决策树回归训练流程的对比

| 步骤 | 线性回归 | 决策树回归 |
|---|---|---|
| 数据 | 手工合成 `(200, 3)` | 真实数据 `(20640, 8)` |
| 标准化 | 无 | 无 |
| 训练算法 | SVD 闭式解——一次性完成 | CART 贪心递归——逐层分裂 |
| 训练复杂度 | $O(d^3 + Nd^2)$——极快 | $O(d \cdot N \log N)$——快 |
| 是否需要 `random_state` | 否——确定性解 | **是——分裂涉及随机性** |
| 收敛判断 | 不需要——闭式解一次到位 | **需要 `max_depth`/`min_samples_split` 等早停** |
| 预测 | $\hat{y} = \mathbf{X}\mathbf{w} + b$（矩阵乘法） | **沿树走到叶子 → 返回叶子均值** |
| 评估可视化 | 残差图 + 学习曲线 | **残差图 + 特征重要性 + 学习曲线 + 树结构** |

## 常见坑

1. 在 `LinearRegression()` 上期待看到 `n_iter` 或训练耗时——它是一次性闭式求解，没有迭代过程。
2. 把 `plot_learning_curve` 传入已训练的 `model`——学习曲线需要未训练的模型实例做交叉验证。
3. 在 200 样本上期待看到学习曲线中训练/验证得分的巨大差异——线性回归参数少（4 个），小样本下也不容易过拟合。

## 小结

- 线性回归流水线为最简 6 步：加载 → 拆分 → 切分 → 训练 → 预测 → 残差图 + 学习曲线——无标准化、无特征重要性、无树结构。
- `fit()` 的核心是 SVD 闭式求解——一次计算，无迭代，无收敛判断，是 OLS 区别于所有迭代式算法的最本质特征。
- `predict()` 是简单的矩阵乘法——测试样本与固定参数做线性组合，计算量极小。
