# 决策树 (Decision Tree)

## 核心思想

决策树通过递归地选择最优特征并按其取值将数据集分割为子集，构建一棵树状判别结构。每个内部节点是一个特征判断，每个叶节点是一个类别标签（分类）或回归值。

## 信息论基础

### 信息熵 (Shannon Entropy)

对于随机变量 $Y$ 取 $K$ 个类别，分布为 $p_k = P(Y=k)$，信息熵定义为：

$$
H(Y) = -\sum_{k=1}^{K} p_k \log_2 p_k
$$

- 当所有类别等概率时，$H$ 取最大值 $\log_2 K$
- 当所有样本属于同一类别时，$H = 0$

### 条件熵

给定特征 $A$ 将数据划分为 $V$ 个子集 $D_1, D_2, \dots, D_V$：

$$
H(Y \mid A) = \sum_{v=1}^{V} \frac{|D_v|}{|D|} H(Y_{D_v})
$$

## ID3 算法：信息增益

**信息增益** = 划分前后熵的减少量：

$$
\text{Gain}(D, A) = H(D) - H(D \mid A)
$$

ID3 算法选择信息增益最大的特征进行分割。

::: warning 缺陷
信息增益偏向取值数目较多的特征。例如"身份证号"的信息增益极高（每个值对应唯一样本），但毫无泛化能力。
:::

## C4.5 算法：信息增益率

为修正 ID3 的偏好，C4.5 引入**特征固有值**（Intrinsic Value）：

$$
\text{IV}(A) = -\sum_{v=1}^{V} \frac{|D_v|}{|D|} \log_2 \frac{|D_v|}{|D|}
$$

**信息增益率**：

$$
\text{Gain\_ratio}(D, A) = \frac{\text{Gain}(D, A)}{\text{IV}(A)}
$$

## CART 算法：基尼系数

**基尼不纯度** (Gini Impurity)：

$$
\text{Gini}(D) = 1 - \sum_{k=1}^{K} p_k^2
$$

对于特征 $A$ 的划分：

$$
\text{Gini}(D, A) = \sum_{v=1}^{V} \frac{|D_v|}{|D|} \text{Gini}(D_v)
$$

CART 选择划分后基尼不纯度最小的特征。

### 基尼系数 vs 信息熵

二分类时，令 $p$ 为正类比例：

- 熵：$H = -p \log_2 p - (1-p) \log_2 (1-p)$
- 基尼：$\text{Gini} = 2p(1-p)$

两者趋势一致，但基尼系数计算更快（无需对数运算）。

## 剪枝策略

### 预剪枝

在构建树的过程中，提前终止分裂。常用条件：
- 节点样本数低于阈值
- 树深度达到上限
- 信息增益低于阈值

### 后剪枝（代价复杂度剪枝）

CART 使用代价复杂度参数 $\alpha$ 控制树的复杂度：

$$
C_\alpha(T) = C(T) + \alpha |T|
$$

其中 $C(T)$ 是训练误差、$|T|$ 是叶节点数。$\alpha$ 越大，惩罚越多，树越简单。

## 回归树

回归树的分割准则为**平方误差最小化**：

$$
\min_{j, s} \left[ \min_{c_1} \sum_{x_i \in R_1(j,s)} (y_i - c_1)^2 + \min_{c_2} \sum_{x_i \in R_2(j,s)} (y_i - c_2)^2 \right]
$$

其中 $c_1, c_2$ 分别为两个区域的均值。

## 代码对应

```bash
python -m pipelines.classification.decision_tree     # 分类
python -m pipelines.regression.decision_tree          # 回归
```
