# XGBoost

## 核心思想

XGBoost (eXtreme Gradient Boosting) 在 GBDT 基础上引入**二阶泰勒展开**来近似目标函数，并加入树结构的正则化项，使训练更快、更稳定。

## 正则化目标函数

$$
\text{Obj} = \sum_{i=1}^{N} L(y_i, \hat{y}_i) + \sum_{t=1}^{T} \Omega(h_t)
$$

其中树的正则项：

$$
\Omega(h) = \gamma \cdot |\text{叶子数}| + \frac{1}{2}\lambda \sum_{j=1}^{J} w_j^2
$$

$J$ 为叶节点数，$w_j$ 为叶节点权重。

## 二阶泰勒展开推导

在第 $t$ 轮，目标函数对第 $t$ 棵树 $h_t$：

$$
\text{Obj}^{(t)} = \sum_{i=1}^{N} L(y_i, \hat{y}_i^{(t-1)} + h_t(\mathbf{x}_i)) + \Omega(h_t)
$$

对 $L$ 在 $\hat{y}_i^{(t-1)}$ 处做二阶泰勒展开：

$$
L(y_i, \hat{y}_i^{(t-1)} + h_t) \approx L(y_i, \hat{y}_i^{(t-1)}) + g_i h_t(\mathbf{x}_i) + \frac{1}{2} h_i h_t^2(\mathbf{x}_i)
$$

其中：

$$
g_i = \frac{\partial L(y_i, \hat{y})}{\partial \hat{y}} \Bigg|_{\hat{y}=\hat{y}_i^{(t-1)}}, \quad
h_i = \frac{\partial^2 L(y_i, \hat{y})}{\partial \hat{y}^2} \Bigg|_{\hat{y}=\hat{y}_i^{(t-1)}}
$$

去掉与 $h_t$ 无关的常数项，目标函数化为：

$$
\tilde{\text{Obj}}^{(t)} = \sum_{i=1}^{N} \left[g_i h_t(\mathbf{x}_i) + \frac{1}{2} h_i h_t^2(\mathbf{x}_i)\right] + \Omega(h_t)
$$

## 叶节点权重最优解

定义叶节点 $j$ 的样本集合 $I_j = \{i : \mathbf{x}_i \in \text{leaf}_j\}$，则 $h_t(\mathbf{x}_i) = w_j$。

代入目标函数：

$$
\tilde{\text{Obj}}^{(t)} = \sum_{j=1}^{J} \left[G_j w_j + \frac{1}{2}(H_j + \lambda)w_j^2\right] + \gamma J
$$

其中 $G_j = \sum_{i \in I_j} g_i$，$H_j = \sum_{i \in I_j} h_i$。

对 $w_j$ 求导令其为零：

$$
\boxed{w_j^* = -\frac{G_j}{H_j + \lambda}}
$$

代回目标函数：

$$
\boxed{\tilde{\text{Obj}}^* = -\frac{1}{2}\sum_{j=1}^{J} \frac{G_j^2}{H_j + \lambda} + \gamma J}
$$

## 分裂增益

对节点分裂的增益：

$$
\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma
$$

只有 $\text{Gain} > 0$ 时才分裂，$\gamma$ 起到预剪枝的作用。

## XGBoost 的工程优化

- **列采样**：类似随机森林的特征子集采样
- **加权分位数草图**：高效近似分割点搜索
- **缓存感知访问**：利用 CPU 缓存加速列遍历
- **稀疏感知**：自动处理缺失值

## 代码对应

```bash
python -m pipelines.ensemble.xgboost
```
