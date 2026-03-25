# 决策树回归 (Decision Tree Regression)

## 核心思想

决策树回归通过递归地将特征空间划分为若干矩形区域，每个区域内以常数（均值）作为预测值。分割准则为**最小化平方误差**。

详细的树构建原理、信息度量、剪枝策略请参阅 [决策树分类](/classification/decision_tree) 页面。本页聚焦回归特有的数学细节。

## 分割准则

### 平方误差最小化

对于特征 $j$ 和分割点 $s$，定义左右子区域：

$$
R_1(j, s) = \{\mathbf{x} \mid x_j \leq s\}, \quad R_2(j, s) = \{\mathbf{x} \mid x_j > s\}
$$

选择最优 $(j, s)$ 使得总平方误差最小：

$$
\min_{j, s} \left[\sum_{\mathbf{x}_i \in R_1} (y_i - \hat{c}_1)^2 + \sum_{\mathbf{x}_i \in R_2} (y_i - \hat{c}_2)^2 \right]
$$

其中 $\hat{c}_m = \text{mean}(y_i : \mathbf{x}_i \in R_m)$。

### 高效搜索

对每个特征 $j$，将其取值排序后遍历所有可能的分割点，总复杂度为 $O(d \cdot N \log N)$。

## 正则化

- **最大深度**：限制树生长层数
- **最小叶节点样本数**：避免过分细化
- **后剪枝**：同分类决策树的代价复杂度剪枝

$$
C_\alpha(T) = \sum_{m=1}^{|T|} N_m \cdot \text{MSE}_m + \alpha |T|
$$

## 代码对应

```bash
python -m pipelines.regression.decision_tree
```
