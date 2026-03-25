# GBDT 梯度提升决策树 (Gradient Boosting Decision Tree)

## 核心思想

GBDT 是一种 **Boosting** 集成方法：通过**逐步添加新的弱学习器**来纠正前一轮残差。与 Bagging 的并行独立不同，GBDT 是串行依赖的，旨在**降低偏差**。

## 前向分布加法模型

最终模型为 $T$ 棵树的叠加：

$$
F_T(\mathbf{x}) = \sum_{t=1}^{T} \eta \cdot h_t(\mathbf{x})
$$

其中 $\eta$ 为学习率（缩减系数），$h_t$ 为第 $t$ 棵回归树。

每一步贪心地添加使损失最小的树：

$$
h_t = \arg\min_{h} \sum_{i=1}^{N} L\big(y_i, F_{t-1}(\mathbf{x}_i) + h(\mathbf{x}_i)\big)
$$

## 负梯度作为伪残差

直接优化上式很困难。**梯度提升**的关键洞察：用损失函数对当前模型预测值的**负梯度**作为第 $t$ 棵树的拟合目标：

$$
r_{ti} = -\frac{\partial L(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)} \Bigg|_{F = F_{t-1}}
$$

| 损失函数 | $L(y, F)$ | 负梯度 $r_{ti}$ |
|---------|-----------|-----------------|
| 平方损失 | $\frac{1}{2}(y-F)^2$ | $y_i - F_{t-1}(\mathbf{x}_i)$ (真实残差) |
| 绝对损失 | $\lvert y-F\rvert$ | $\text{sign}(y_i - F_{t-1}(\mathbf{x}_i))$ |
| 对数损失 (分类) | $-[y\ln p + (1-y)\ln(1-p)]$ | $y_i - p_i$ |

可以看到，**当使用平方损失时，负梯度恰好就是残差本身**，这正是 GBDT 名称中"残差"的由来。

## 完整算法流程

1. 初始化 $F_0(\mathbf{x}) = \arg\min_c \sum_{i=1}^N L(y_i, c)$
2. 对 $t = 1, 2, \dots, T$：
   1. 计算伪残差 $r_{ti}$
   2. 用回归树拟合 $\{(\mathbf{x}_i, r_{ti})\}$，得到 $h_t$
   3. 对每个叶节点区域 $R_{tm}$，计算最优输出值：$\gamma_{tm} = \arg\min_\gamma \sum_{\mathbf{x}_i \in R_{tm}} L(y_i, F_{t-1}(\mathbf{x}_i) + \gamma)$
   4. 更新 $F_t(\mathbf{x}) = F_{t-1}(\mathbf{x}) + \eta \sum_m \gamma_{tm} \mathbb{1}(\mathbf{x} \in R_{tm})$

## 正则化手段

- **学习率** $\eta \in (0, 1]$：缩小每棵树的贡献
- **子采样**：每轮只用部分样本（随机梯度提升）
- **树复杂度**：限制深度、叶节点数、最小分裂样本数

## 代码对应

```bash
python -m pipelines.ensemble.gbdt
```
