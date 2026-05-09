---
title: SVR 支持向量回归 — 训练与预测
outline: deep
---

# 训练与预测

## 本章目标

1. 理解 SVR 流水线的完整执行顺序——从数据加载到残差图和学习曲线输出。
2. 理解 SVR 的训练过程——SMO 迭代求解对偶问题，不可见但关键。
3. 理解 SVR 的预测方式——支持向量与测试点的核函数加权求和，而非矩阵乘法。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `loadSvrDataset()` | 方法 | 生成 Friedman1 非线性数据——返回 `(200, 11)` DataFrame |
| `StandardScaler` | 预处理 | Z-score 标准化——RBF 核距离计算的前置条件 |
| `trainSvrRegressionModel(...)` | 函数 | 构建并 `fit` SVR 模型——SMO 迭代求解 |
| `model.predict(X_test_s)` | 方法 | 支持向量与测试点的核函数加权求和预测 |
| `plot_residuals(...)` | 函数 | 残差诊断图 |
| `plot_learning_curve(...)` | 函数 | 学习曲线——使用新 `SVR(...)` 实例 |

## 1. 完整流水线流程

### 流程概述

```
loadSvrDataset()
    │
    ├─ ① X = data.drop(columns=["price"]), y = data["price"]
    ├─ ② X_train, X_test, y_train, y_test = train_test_split(test_size=0.2)
    ├─ ③ scaler = StandardScaler(); X_train_s = scaler.fit_transform(X_train)
    ├─ ④ X_test_s = scaler.transform(X_test)
    ├─ ⑤ model = trainSvrRegressionModel(X_train_s, y_train)
    ├─ ⑥ y_pred = model.predict(X_test_s)
    ├─ ⑦ plot_residuals(y_test, y_pred)
    └─ ⑧ plot_learning_curve(SVR(C=10.0, epsilon=0.1, kernel='rbf', gamma='scale'), X_train_s, y_train, scoring='r2')
```

### 参数速览

| 步骤 | 操作 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| 加载数据 | `loadSvrDataset` | — | `DataFrame`，`(200, 11)` | Friedman1 非线性数据 |
| 特征标签拆分 | `drop` + 列选择 | `DataFrame` | `X(200,10)`, `y(200,)` | 标签列 `price` |
| 数据切分 | `train_test_split` | `X`, `y` | `X_train(160,10)`, `X_test(40,10)` | `test_size=0.2` |
| 标准化 | `StandardScaler` | `X_train`, `X_test` | `X_train_s`, `X_test_s` | **SVR（RBF 核）必需** |
| 训练 | `trainSvrRegressionModel` | `X_train_s`, `y_train` | `SVR` 模型 | SMO 迭代求解 |
| 预测 | `model.predict` | `X_test_s` | `y_pred(40,)` | 核函数加权求和 |
| 残差图 | `plot_residuals` | `y_test`, `y_pred` | PNG 图像 | 误差分布诊断 |
| 学习曲线 | `plot_learning_curve` | 新 `SVR(...)`, `X_train_s`, `y_train` | PNG 图像 | 样本量-得分趋势 |

### 理解重点

- SVR 流水线为 8 步——与线性回归（6 步）相比多了标准化（③④），与正则化回归（8 步）步骤数相同但无多模型循环。
- 标准化必须在切分之后——与正则化回归一致。
- SVR 没有特征重要性可视化——`PipelineSpec` 中训练后诊断列表为 `[]`。RBF 核的权重在对偶空间无法映射回原始特征。

## 2. 标准化：RBF 核训练的关键前置

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `ndarray(160, 10)` | 未标准化的训练特征 | 原始 Friedman1 各列 |
| `X_train_s` | `ndarray(160, 10)` | 标准化后——每列 μ=0, σ=1 | — |
| `X_test_s` | `ndarray(40, 10)` | 使用训练集统计量标准化 | — |

### 理解重点

- RBF 核计算 $\exp(-\gamma\|\mathbf{x}_i - \mathbf{x}_j\|^2)$——欧氏距离对尺度极敏感。若某特征量级为 100 而其他为 0.1，该特征将完全主导核计算。
- 标准化使所有特征在核距离中平等——这是 RBF 核 SVR 必须标准化的数学原因。
- `fit_transform` 仅用于训练集，`transform` 用于测试集——这是数据泄露的基本防护。

## 3. 训练细节：SMO 迭代求解

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| 优化算法 | — | SMO（Sequential Minimal Optimization） | scikit-learn 内部实现 |
| 优化变量 | — | $\alpha_i, \alpha_i^*$——共 $2 \times 160 = 320$ 个变量 | — |
| 目标 | — | 最大化对偶问题——凸二次规划 | — |
| 终止条件 | — | 对偶间隙 < `tol`（默认 1e-3） | `tol=1e-3` |

### 理解重点

- SVR 的训练是**不可见的迭代过程**——不像决策树那样可以观察分裂步骤，也不像线性回归的一步到位。
- 训练复杂度约 $O(N^2 \cdot d)$ 到 $O(N^3)$——在 160 样本上毫秒级完成。
- 训练过程是**确定性**的——对偶问题为凸优化，给定相同数据必然收敛到相同解。因此 `SVR` 无需 `random_state` 参数。
- 训练完成后，大部分样本的 $\alpha_i - \alpha_i^* = 0$——它们被 ε-管道"忽视"，不参与预测。

## 4. 预测细节：核函数加权求和

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_test_s` | `ndarray(40, 10)` | 标准化后的测试特征 | — |
| `model.support_` | `ndarray(nSV,)` | 支持向量索引——仅这些样本参与预测 | — |
| `model.dual_coef_` | `ndarray(1, nSV)` | $\alpha_i - \alpha_i^*$ 的值——支持向量的权重 | — |
| `y_pred` | `ndarray(40,)` | 预测值 | — |

$$
f(\mathbf{x}) = \sum_{i \in \text{SV}} (\alpha_i - \alpha_i^*) K(\mathbf{x}_i, \mathbf{x}) + b
$$

### 理解重点

- 预测**不是** $\mathbf{X}\mathbf{w} + b$ 的矩阵乘法——而是与支持向量逐一计算核函数再加权求和。
- 预测复杂度 $O(nSV \cdot N_{\text{test}} \cdot d)$——支持向量越多预测越慢。
- 支持向量占比（nSV / N_train）直接决定了预测成本——通常为 30%~60%。
- 这与线性回归形成鲜明对比——线性回归的预测是固定 $O(N_{\text{test}} \cdot d)$，与训练集大小无关。

## 5. SVR 预测 vs 线性回归预测 对比

| 预测维度 | 线性回归 | SVR（RBF 核） |
|---|---|---|
| 公式 | $\hat{y} = \mathbf{X}\mathbf{w} + b$ | **$\hat{y} = \sum(\alpha_i - \alpha_i^*)K(\mathbf{x}_i, \mathbf{x}) + b$** |
| 复杂度 | $O(N_{\text{test}} \cdot d)$ | **$O(nSV \cdot N_{\text{test}} \cdot d)$** |
| 依赖训练集 | 否——参数已固化为 $\mathbf{w}$ | **是——需要存储支持向量集** |
| 参与计算的样本 | 无——仅用参数 | **仅支持向量——管道内样本被忽略** |
| 内存占用 | $O(d)$——仅存储系数 | **$O(nSV \cdot d)$——存储支持向量** |

## 6. SVR vs 线性回归 vs 正则化回归 训练对比

| 训练维度 | 线性回归 | 正则化回归 | SVR |
|---|---|---|---|
| 数据 | 合成 `(200, 3)` | 真实+构造 `(442, 21)` | **合成非线性 `(200, 10)`** |
| 标准化 | 无 | **`StandardScaler`** | **`StandardScaler`** |
| 训练算法 | SVD 闭式解 | 坐标下降（Lasso/EN）+ 闭式解（Ridge） | **SMO——序列最小优化** |
| 训练模型数 | 1 | 3（并行） | **1** |
| 收敛判断 | 不需要 | `max_iter`（Lasso/EN） | **对偶间隙 < tol** |
| 预测 | $\mathbf{X}\mathbf{w} + b$ | $\mathbf{X}\mathbf{w} + b$ | **$\sum(\alpha_i - \alpha_i^*)K(\mathbf{x}_i, \mathbf{x}) + b$** |
| 评估可视化 | 残差图 + 学习曲线 | 残差图 + 特征重要性 | **残差图 + 学习曲线** |
| 独有诊断 | coef_ 对照真实公式 | 近零系数计数 | **支持向量数量 + 占比** |

## 常见坑

1. 期待 SVR（RBF 核）输出 `coef_`——线性核才有 `coef_`，RBF 核的权重在 `dual_coef_` 中，不可直接解释。
2. 预测时传未标准化的 `X_test`——RBF 核的欧氏距离对尺度敏感，未标准化会导致预测结果严重偏离。
3. 忽略支持向量数量——如果 nSV 接近 N_train（如 > 90%），说明模型近乎"记住"了所有训练样本，可能严重过拟合。
4. 认为 SVR 也是一步到位——训练过程是 SMO 迭代，虽然对 160 样本几乎瞬时，但对大规模数据会显著变慢。

## 小结

- SVR 流水线为 8 步：加载 → 拆分 → 切分 → 标准化 → 训练 → 预测 → 残差图 → 学习曲线。无特征重要性。
- 标准化是 SVR（RBF 核）的硬性要求——欧氏距离对特征尺度敏感。
- 训练使用 SMO 迭代求解凸二次规划——确定性过程，220 个变量（2N）在小样本上瞬时完成。
- 预测是支持向量与测试点的核函数加权求和——与线性回归的矩阵乘法本质不同，复杂度依赖支持向量数量。
