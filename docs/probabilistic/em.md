# EM 算法与高斯混合模型 (GMM)

## 核心思想

EM (Expectation-Maximization) 是一种用于**含隐变量的概率模型**的参数估计迭代算法。GMM 是 EM 最经典的应用：用多个高斯分布的加权和来建模数据分布。

## 高斯混合模型

### 模型定义

$$
p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

- $\pi_k$：第 $k$ 个高斯的**混合系数**，$\sum_k \pi_k = 1$，$\pi_k \geq 0$
- $\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$：多元高斯密度

$$
\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{d/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\right)
$$

### 隐变量

引入隐变量 $z_i \in \{1, \dots, K\}$ 表示样本 $\mathbf{x}_i$ 来自哪个分量。

## 为什么不能直接 MLE？

对数似然：

$$
\ln L = \sum_{i=1}^{N} \ln \left[\sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)\right]
$$

**对数内有求和**，无法分解，直接求导没有闭式解。

## EM 算法的理论基础

### Jensen 不等式

对于凹函数 $\ln$：

$$
\ln \left(\sum_k q_k \frac{p_k}{q_k}\right) \geq \sum_k q_k \ln \frac{p_k}{q_k}
$$

构造对数似然的**下界** (ELBO)，EM 通过交替最大化下界来逼近似然极大值。

### ELBO (Evidence Lower Bound)

$$
\ln L \geq \sum_{i=1}^N \sum_{k=1}^K \gamma_{ik} \ln \frac{\pi_k \, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\gamma_{ik}} = \text{ELBO}
$$

当 $\gamma_{ik} = P(z_i = k \mid \mathbf{x}_i)$（后验概率）时，下界取等号。

## E 步 (Expectation)

计算隐变量的后验概率（责任度）：

$$
\gamma_{ik} = \frac{\pi_k \, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K} \pi_j \, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}
$$

## M 步 (Maximization)

利用 $\gamma_{ik}$ 更新参数：

$$
N_k = \sum_{i=1}^N \gamma_{ik}
$$

$$
\boldsymbol{\mu}_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} \, \mathbf{x}_i
$$

$$
\boldsymbol{\Sigma}_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k^{\text{new}})(\mathbf{x}_i - \boldsymbol{\mu}_k^{\text{new}})^T
$$

$$
\pi_k^{\text{new}} = \frac{N_k}{N}
$$

## 收敛性

每次迭代 $\ln L$ 单调不减。EM 算法收敛到**局部极大值**（不保证全局最优）。

## 代码对应

```bash
python -m pipelines.probabilistic.em
```
