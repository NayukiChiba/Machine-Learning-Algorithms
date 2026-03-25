# Bagging 与随机森林 (Random Forest)

## Bagging (Bootstrap Aggregating)

### 核心思想

Bagging 通过**自举采样** (Bootstrap) 构建多个独立的基学习器，再通过投票（分类）或平均（回归）进行集成，以**降低方差**。

### Bootstrap 采样

从大小为 $N$ 的训练集中**有放回地**抽取 $N$ 个样本，形成一个 Bootstrap 样本集。每个样本被抽中的概率为：

$$
P(\text{被选中}) = 1 - \left(1 - \frac{1}{N}\right)^N \xrightarrow{N \to \infty} 1 - \frac{1}{e} \approx 0.632
$$

约 36.8% 的样本未被选中（称为 **OOB, Out-Of-Bag** 样本），可用于评估。

### 方差缩减

假设 $T$ 个基学习器 $h_1, \dots, h_T$ 的预测方差均为 $\sigma^2$，两两相关系数为 $\rho$，集成后方差：

$$
\text{Var}\left[\frac{1}{T}\sum_{t=1}^T h_t\right] = \rho\sigma^2 + \frac{1-\rho}{T}\sigma^2
$$

- 第一项 $\rho\sigma^2$ 不可消除（受相关性限制）
- 第二项随 $T$ 增大而趋近于零

## 随机森林 (Random Forest)

### 改进：特征随机化

在 Bagging 的基础上，随机森林在每次节点分裂时，**只从随机抽取的 $m$ 个特征中选择最优分裂特征**，进一步降低基学习器的相关性 $\rho$。

推荐值：

| 任务 | $m$ 的推荐值 |
|------|-------------|
| 分类 | $m = \lfloor \sqrt{d} \rfloor$ |
| 回归 | $m = \lfloor d/3 \rfloor$ |

### 特征重要性

基于不纯度（MDI）：特征 $j$ 在所有树中被用于分裂时带来的不纯度减少之和。

$$
\text{Imp}(j) = \sum_{t=1}^{T} \sum_{\text{node } v \text{ splits on } j} \Delta \text{Gini}(v)
$$

## 代码对应

```bash
python -m pipelines.classification.random_forest
python -m pipelines.ensemble.bagging
```
