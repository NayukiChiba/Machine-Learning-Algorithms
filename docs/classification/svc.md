# SVM 支持向量机 (Support Vector Machine)

## 核心思想

SVM 寻找一个**最大间隔超平面**将不同类别的数据分开。核心直觉是：在所有能正确分类训练数据的超平面中，间隔最大的那个泛化能力最强。

## 硬间隔线性 SVM

### 超平面与函数间隔

超平面定义为：

$$
\mathbf{w}^T \mathbf{x} + b = 0
$$

样本 $(\mathbf{x}_i, y_i)$ 到超平面的**几何间隔**：

$$
\gamma_i = y_i \cdot \frac{\mathbf{w}^T \mathbf{x}_i + b}{\|\mathbf{w}\|}
$$

### 最大间隔优化问题

令所有样本的最小几何间隔为 $\gamma = \min_i \gamma_i$，最大化 $\gamma$：

$$
\max_{\mathbf{w}, b} \frac{2}{\|\mathbf{w}\|} \quad \text{s.t.} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \quad \forall i
$$

等价于：

$$
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \quad \forall i
$$

### 拉格朗日对偶

引入拉格朗日乘子 $\alpha_i \geq 0$，构造拉格朗日函数：

$$
\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2} \|\mathbf{w}\|^2 - \sum_{i=1}^{N} \alpha_i \Big[ y_i(\mathbf{w}^T \mathbf{x}_i + b) - 1 \Big]
$$

对 $\mathbf{w}$ 和 $b$ 求偏导并令其为零：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 0 \implies \mathbf{w} = \sum_{i=1}^N \alpha_i y_i \mathbf{x}_i
$$

$$
\frac{\partial \mathcal{L}}{\partial b} = 0 \implies \sum_{i=1}^N \alpha_i y_i = 0
$$

代回得到**对偶问题**：

$$
\max_{\boldsymbol{\alpha}} \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j
$$

$$
\text{s.t.} \quad \alpha_i \geq 0, \quad \sum_{i=1}^N \alpha_i y_i = 0
$$

### KKT 条件

互补松弛条件：

$$
\alpha_i \Big[ y_i(\mathbf{w}^T \mathbf{x}_i + b) - 1 \Big] = 0, \quad \forall i
$$

这意味着只有**支持向量**（$y_i(\mathbf{w}^T \mathbf{x}_i + b) = 1$ 的样本）对应的 $\alpha_i > 0$。

## 软间隔 SVM

引入松弛变量 $\xi_i \geq 0$ 和惩罚参数 $C > 0$：

$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{N} \xi_i
$$

$$
\text{s.t.} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
$$

对偶问题变为约束 $0 \leq \alpha_i \leq C$。

## 核函数 (Kernel Trick)

对偶问题中的 $\mathbf{x}_i^T \mathbf{x}_j$ 可替换为核函数 $K(\mathbf{x}_i, \mathbf{x}_j)$，实现**隐式映射到高维空间**而无需显式计算：

| 核函数 | 公式 |
|--------|------|
| 线性核 | $K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T \mathbf{z}$ |
| 多项式核 | $K(\mathbf{x}, \mathbf{z}) = (\gamma \mathbf{x}^T \mathbf{z} + r)^d$ |
| RBF (高斯核) | $K(\mathbf{x}, \mathbf{z}) = \exp\left(-\gamma \|\mathbf{x} - \mathbf{z}\|^2\right)$ |
| Sigmoid 核 | $K(\mathbf{x}, \mathbf{z}) = \tanh(\gamma \mathbf{x}^T \mathbf{z} + r)$ |

**Mercer 定理**：$K$ 必须是正半定的，即对任意样本集，核矩阵 $\mathbf{K}_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$ 满足 $\mathbf{K} \succeq 0$。

### RBF 核的直觉

RBF 核相当于将样本映射到无穷维空间。参数 $\gamma$ 越大，决策边界越"弯曲"、越灵活（更容易过拟合）。

## SMO 算法简述

序列最小优化（Sequential Minimal Optimization）每次选取两个 $\alpha_i, \alpha_j$ 进行解析更新，利用约束 $\alpha_i y_i + \alpha_j y_j = \text{const}$ 将二变量问题化为一元问题，得到闭式解。

## 代码对应

```bash
python -m pipelines.classification.svc
```
