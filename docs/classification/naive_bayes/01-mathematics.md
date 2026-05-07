---
title: GaussianNB 高斯朴素贝叶斯 — 数学原理
outline: deep
---

# 数学原理

## 本章目标

1. 理解朴素贝叶斯如何用贝叶斯公式从先验和似然推算后验概率——这是一种生成式分类思路。
2. 理解"朴素"条件独立假设 $P(\mathbf{x}\vert Y) = \prod P(x_j\vert Y)$ 的数学含义、简化效果和现实代价。
3. 理解 `GaussianNB` 对连续特征的高斯建模方式——每个类别每个特征各拟合一个 $\mathcal{N}(\mu_{kj}, \sigma_{kj}^2)$。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 贝叶斯公式 $P(Y\vert\mathbf{x}) \propto P(\mathbf{x}\vert Y)P(Y)$ | 基础公式 | 从先验 $P(Y)$ 和似然 $P(\mathbf{x}\vert Y)$ 推到后验概率 |
| 条件独立假设 $\prod P(x_j\vert Y)$ | 核心简化 | 把高维联合分布拆成单特征条件概率乘积，大幅减少参数 |
| MAP 决策 $\arg\max_c P(c)\prod P(x_j\vert c)$ | 分类规则 | 选择后验概率最大的类别作为预测输出 |
| 高斯似然 $\mathcal{N}(\mu_{kj}, \sigma_{kj}^2)$ | 概率模型 | 当前实现中对每类每个连续特征的单高斯建模 |
| `var_smoothing` | 超参数 | 向方差中加入极小平滑项 $\epsilon$，防止 $\sigma_{kj}^2 \to 0$ 时数值崩溃 |
| 类别先验 $P(Y=c_k) = n_k/N$ | 概率项 | 各类别在训练集中的基础比例 |

## 1. 朴素贝叶斯的核心思想

朴素贝叶斯是一种生成式分类器：它不直接学习分类边界，而是先对"每个类别下数据长什么样"建模（$P(\mathbf{x}\vert Y)$），再结合各类别的先验概率（$P(Y)$），通过贝叶斯公式推算后验概率。

### 贝叶斯公式

$$
P(Y = c_k \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid Y = c_k) \, P(Y = c_k)}{P(\mathbf{x})}
$$

其中：
- $P(Y = c_k)$：先验概率——在不知道特征值之前，样本属于类别 $c_k$ 的概率
- $P(\mathbf{x} \mid Y = c_k)$：似然——如果样本属于类别 $c_k$，看到特征 $\mathbf{x}$ 的概率有多大
- $P(Y = c_k \mid \mathbf{x})$：后验概率——在看到特征 $\mathbf{x}$ 后，样本属于类别 $c_k$ 的最终概率
- $P(\mathbf{x})$：证据项——对所有类别相同，分类时通常忽略

### 理解重点

- 朴素贝叶斯是生成式模型：先对 $P(\mathbf{x}, Y)$ 建模，再由贝叶斯公式推出 $P(Y\vert\mathbf{x})$。
- 这与逻辑回归（直接对 $P(Y\vert\mathbf{x})$ 用 Sigmoid 建模）的判别式思路有本质区别——逻辑回归不关心数据是怎么"生成"的。
- 证据项 $P(\mathbf{x})$ 在分类决策时可忽略，因为它对所有类别相同。

## 2. 为什么叫"朴素"：条件独立假设

如果不对似然 $P(\mathbf{x} \mid Y = c_k)$ 做任何简化，需要对 $d$ 维连续特征估计一个完整的 $d$ 维联合分布——这在样本有限时极难做到。

朴素贝叶斯的关键假设是：**在给定类别后，各个特征之间条件独立**。

$$
P(\mathbf{x} \mid Y = c_k) = P(x_1, x_2, \dots, x_d \mid Y = c_k) = \prod_{j=1}^{d} P(x_j \mid Y = c_k)
$$

这个假设让参数估计从 $\mathcal{O}(\text{指数级})$ 降到 $\mathcal{O}(d)$。代价是：如果特征之间存在强依赖关系（如 $x_2 \approx 2x_1$），模型会双倍计算同一条信息，导致概率估计偏差。

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `priors` | `array_like` 或 `None` | 类别的先验概率 $P(Y=c_k)$。`None` 时从训练数据估计：$P(Y=c_k) = n_k / N$。可手动传入数组来覆盖从数据估计的先验。默认为 `None` | `None`、`[0.3, 0.3, 0.4]` |

### 理解重点

- "朴素"不是指算法粗糙，而是对特征关系做了强简化——假设条件独立。
- 这个假设在真实数据里往往不严格成立，但好处是：联合分布建模的难度大幅下降，参数估计直接、训练极快。
- 即使特征不完全独立，朴素贝叶斯在实际应用中仍常给出不错的效果——特别是在高维文本分类中。

## 3. 分类决策：最大后验概率（MAP）

联合贝叶斯公式和条件独立假设，得到分类规则（对数形式，避免连乘下溢）：

$$
\hat{y} = \arg\max_{c_k} \left[ \ln P(Y = c_k) + \sum_{j=1}^{d} \ln P(x_j \mid Y = c_k) \right]
$$

直观理解：
- 第一项 $\ln P(Y=c_k)$：类别本身有多常见（先验）
- 第二项 $\sum \ln P(x_j\vert Y=c_k)$：当前特征值在类别 $c_k$ 下有多"自然"（似然之和）

两项相加，得分最高的类别就是预测输出。

### 理解重点

- 这是纯粹的代数计算，不涉及梯度下降或迭代优化——因此训练极快。
- 两项之间没有权重超参数调节——模型对先验和似然的信任程度完全由数据决定。
- 对数形式是工程实现的必须——$d$ 个 $(0,1)$ 区间概率连乘会迅速下溢到浮点数零。

## 4. GaussianNB：连续特征的高斯建模

`GaussianNB` 假设：在每个类别 $c_k$ 内，每个连续特征 $x_j$ 服从高斯（正态）分布。

$$
P(x_j \mid Y = c_k) = \frac{1}{\sqrt{2\pi \sigma_{kj}^2}} \exp\left(-\frac{(x_j - \mu_{kj})^2}{2\sigma_{kj}^2}\right)
$$

其中 $\mu_{kj}$ 和 $\sigma_{kj}^2$ 是从训练数据中该类该特征的最大似然估计：

$$
\mu_{kj} = \frac{1}{n_k} \sum_{i: y_i = c_k} x_{ij}, \quad
\sigma_{kj}^2 = \frac{1}{n_k} \sum_{i: y_i = c_k} (x_{ij} - \mu_{kj})^2
$$

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `var_smoothing` | `float` | 方差平滑项 $\epsilon$。实际计算使用 $\sigma_{kj}^2 + \epsilon \cdot \sigma_{\max}^2$，其中 $\sigma_{\max}^2$ 是所有特征所有类别中最大的方差。防止方差异常小时 $\frac{1}{\sqrt{2\pi\sigma^2}} \to \infty$ 导致数值问题。默认为 `1e-9` | `1e-9`、`1e-8`、`1e-7` |

### 示例代码

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB(var_smoothing=1e-9)
model.fit(X_train_s, y_train)
# model.theta_   → 各类别各特征的均值 μ_kj，形状 (n_classes, n_features)
# model.var_     → 各类别各特征的方差 σ_kj²（应用平滑后的），形状 (n_classes, n_features)
```

### 理解重点

- iris 数据的 4 个特征都是连续值（萼片长宽、花瓣长宽），因此用高斯分布建模是自然选择。
- 高斯似然的参数（$\mu_{kj}$、$\sigma_{kj}^2$）通过简单的统计公式一步得到——不涉及迭代优化。
- 如果某些类别下某些特征的方差极小（所有样本取值几乎相同），$\sigma_{kj}^2 \approx 0$ 会让高斯概率密度值趋于无穷——`var_smoothing` 就是为此设置的数值保险。

## 5. 为什么 GaussianNB 不需要像逻辑回归那样迭代优化

GaussianNB 的参数估计全部是解析解（闭式解）：

- 先验：$P(Y=c_k) = n_k / N$（计数除总数）
- 均值：$\mu_{kj} = \frac{1}{n_k} \sum x_{ij}$（样本均值）
- 方差：$\sigma_{kj}^2 = \frac{1}{n_k} \sum (x_{ij} - \mu_{kj})^2$（样本方差）

没有需要迭代优化的损失函数，没有梯度计算，没有收敛判断。这使得 GaussianNB 的 `fit()` 是所有分类模型中最快的之一。

### 理解重点

- GaussianNB 的训练等价于"统计各类别下各特征的均值和方差"——纯粹的数据扫描。
- 这与逻辑回归（`lbfgs` 迭代优化交叉熵）、KNN（虽然不优化但需建索引）、决策树（递归贪心搜分裂点）在计算特征上有本质区别。
- 代价是：高斯假设 + 条件独立假设在复杂真实数据上可能偏离较大，需要权衡模型假设的合理性与计算效率。

## 6. 数学原理如何映射到当前源码

以下表格将本章涉及的数学概念与当前仓库的代码实现一一对应：

| 数学概念 | 数学符号/公式 | 代码实现 |
|---|---|---|
| 贝叶斯公式 | $P(Y\vert\mathbf{x}) = \frac{P(\mathbf{x}\vert Y)P(Y)}{P(\mathbf{x})}$ | `model.predict_proba(X)` 内部 |
| 条件独立假设 | $P(\mathbf{x}\vert Y) = \prod_j P(x_j\vert Y)$ | `GaussianNB` 算法核心假设 |
| MAP 决策 | $\hat{y} = \arg\max_c [\ln P(c) + \sum \ln P(x_j\vert c)]$ | `model.predict(X)` |
| 类别先验 | $P(Y=c_k) = n_k / N$ | `model.class_prior_` |
| 类别样本数 | $n_k$ | `model.class_count_` |
| 高斯均值 | $\mu_{kj}$ | `model.theta_`，形状 `(n_classes, n_features)` |
| 高斯方差 | $\sigma_{kj}^2$ | `model.var_`（平滑后），形状 `(n_classes, n_features)` |
| 方差平滑 | $\sigma^2 + \epsilon \cdot \sigma_{\max}^2$ | `var_smoothing=1e-9` |
| 平滑绝对值 | $\epsilon \cdot \sigma_{\max}^2$ | `model.epsilon_` |
| 类别标签 | $\{c_1, \dots, c_K\}$ | `model.classes_` |

## 常见坑

1. 把朴素贝叶斯理解成"简单版分类器"，而忽略它是生成式概率模型——对 $P(\mathbf{x}\vert Y)$ 建模，而非直接拟合边界。
2. 把"条件独立"误解成特征在原始数据中必须完全独立——这个假设是为计算可行性做的简化，现实中很少严格成立，但模型常常仍然有效。
3. 混淆 GaussianNB（连续特征，高斯似然）与其他朴素贝叶斯变体——MultinomialNB（离散计数）、BernoulliNB（二元特征）、ComplementNB（不平衡文本）。
4. 忽略 `var_smoothing` 的数值稳定作用——特别是当某些类别样本极少、特征方差接近 0 时。

## 小结

- 朴素贝叶斯的核心数学链：贝叶斯公式 $P(Y\vert\mathbf{x}) \propto P(\mathbf{x}\vert Y)P(Y)$ → 条件独立 $\prod P(x_j\vert Y)$ → 高斯似然 $\mathcal{N}(\mu_{kj}, \sigma_{kj}^2)$ → MAP 决策（对数形式）→ `var_smoothing` 数值保护。
- 所有参数（先验、均值、方差）都是解析解——不需要迭代优化，训练极快。
- 当前源码使用 `GaussianNB(var_smoothing=1e-9)`，对应连续特征的高斯建模——与 iris 数据的 4 个连续特征天然匹配。
