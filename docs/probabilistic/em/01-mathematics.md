---
title: EM 与 GMM — 数学原理
outline: deep
---

# 数学原理

> 对应代码：`model_training/probabilistic/em.py`
>  
> 相关对象：`GaussianMixture`、`train_model(...)`

## 本章目标

1. 理解 EM 为什么适合用来估计含隐变量的 GMM 参数。
2. 理解 GMM、ELBO、E 步、M 步与收敛性之间的数学关系。
3. 把这些公式和当前源码中的 `n_components`、`covariance_type`、`lower_bound_` 对应起来。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| GMM | 概率模型 | 用多个高斯分布的加权和建模数据 |
| EM | 迭代算法 | 交替估计隐变量和参数 |
| ELBO | 下界目标 | 把难直接优化的对数似然转成可交替优化的形式 |
| 责任度 | 后验概率 | 表示样本属于各分量的概率 |
| `lower_bound_` | 源码属性 | 当前训练收敛后的平均对数似然下界 |

## 1. 核心思想

EM（Expectation-Maximization）是一种用于含隐变量概率模型的参数估计迭代算法。GMM 是 EM 最经典的应用：用多个高斯分布的加权和来建模数据分布。

### 理解重点

- 当前源码中的 `GaussianMixture(...)`，本质上就是在做 GMM 参数估计。
- 文档里的“隐变量”，在当前分册中对应“每个样本到底来自哪个高斯分量”。
- EM 之所以重要，是因为这个分量归属在训练时并不可见。

## 2. 高斯混合模型

### 参数速览（本节）

适用模型：GMM

| 参数名 | 数学含义 | 在源码中的对应 |
|---|---|---|
| $K$ | 高斯分量数 | `n_components` |
| $\pi_k$ | 第 $k$ 个分量的混合系数 | 分量权重 |
| $\boldsymbol{\mu}_k$ | 第 $k$ 个分量的均值 | 分量中心 |
| $\boldsymbol{\Sigma}_k$ | 第 $k$ 个分量的协方差矩阵 | 协方差结构 |

### 模型定义

$$
p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

- $\pi_k$：第 $k$ 个高斯的混合系数，满足 $\sum_k \pi_k = 1$，且 $\pi_k \geq 0$
- $\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$：多元高斯密度

$$
\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{d/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\right)
$$

### 理解重点

- 当前分册中的二维混合高斯数据，正好对应这个模型的最直观实例。
- `n_components=3` 在源码里就是对公式中 $K=3$ 的具体实现。
- `covariance_type='full'` 则对应对协方差结构的完整建模能力。

## 3. 隐变量视角

引入隐变量 $z_i \in \{1, \dots, K\}$ 表示样本 $\mathbf{x}_i$ 来自哪个分量。

### 理解重点

- 当前训练数据里虽然有 `true_label`，但那只是数据生成时记录下来的参考标签，训练时并不会传给模型。
- 从模型视角看，真正需要估计的就是这个不可见的分量归属。
- 这正是 EM 要解决的问题。

## 4. 为什么不能直接做普通 MLE

对数似然为：

$$
\ln L = \sum_{i=1}^{N} \ln \left[\sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)\right]
$$

对数内部带有求和项，因此无法像线性模型那样直接展开并得到简单闭式解。

### 理解重点

- 困难点不在高斯分布本身，而在“一个样本可能来自多个分量的混合表达”。
- 正是这个“对数内的求和”使直接最大似然变得不方便。
- EM 的价值，就是把这个难问题转换成一个可交替求解的问题。

## 5. EM 算法的理论基础

### Jensen 不等式

对于凹函数 $\ln$：

$$
\ln \left(\sum_k q_k \frac{p_k}{q_k}\right) \geq \sum_k q_k \ln \frac{p_k}{q_k}
$$

构造对数似然的下界（ELBO），EM 通过交替最大化下界来逼近似然极大值。

### ELBO（Evidence Lower Bound）

$$
\ln L \geq \sum_{i=1}^N \sum_{k=1}^K \gamma_{ik} \ln \frac{\pi_k \, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\gamma_{ik}} = \text{ELBO}
$$

当 $\gamma_{ik} = P(z_i = k \mid \mathbf{x}_i)$（后验概率）时，下界取等号。

### 理解重点

- ELBO 的作用，是把原本难处理的对数似然变成一个可以交替优化的目标。
- 当前源码虽然没有显式写出 ELBO 公式，但训练后打印的 `lower_bound_` 与这个下界直接相关。
- 这就是数学下界与工程日志之间最直接的连接点。

## 6. E 步（Expectation）

计算隐变量的后验概率，也就是责任度：

$$
\gamma_{ik} = \frac{\pi_k \, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K} \pi_j \, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}
$$

### 理解重点

- 责任度表示样本 $i$ 属于第 $k$ 个分量的概率。
- 这正是 GMM 训练过程属于“软聚类”的根本原因。
- 当前流水线最终虽然输出硬标签，但训练核心仍然是这组软责任度。

## 7. M 步（Maximization）

利用责任度更新参数：

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

### 理解重点

- M 步的本质，是把 E 步得到的软归属概率重新汇总成新的模型参数。
- 更新后的均值、协方差和权重会再次进入下一轮 E 步。
- 这就是 EM 交替迭代的核心闭环。

## 8. 收敛性

每次迭代 $\ln L$ 单调不减。EM 算法收敛到局部极大值，不保证全局最优。

### 理解重点

- 当前源码中的 `max_iter=200`，就是给这种迭代过程设置一个上限。
- `random_state=42` 则帮助固定初始化与局部解路径，使结果更可复现。
- 训练日志中的 `lower_bound_`，可以理解为当前实现里最接近“收敛下界”观测值的工程输出。

## 9. 数学原理如何映射到当前源码

### 理解重点

- 公式里的 $K$，在工程里对应 `n_components`。
- 协方差矩阵建模方式，在工程里对应 `covariance_type`。
- ELBO/对数似然下界，在工程里对应 `model.lower_bound_` 的训练日志输出。
- E 步和 M 步本身没有在仓库里手写实现，而是由 `GaussianMixture.fit(...)` 内部完成。

## 常见坑

1. 把 `true_label` 误当成 E 步需要的输入，实际上 EM 训练不依赖这个参考列。
2. 只记住 E 步和 M 步公式，却不把它们和 `n_components`、`covariance_type`、`lower_bound_` 这些源码对象对应起来。
3. 误以为 EM 能保证全局最优，忽略了它通常只收敛到局部极大值。

## 小结

- EM / GMM 的数学核心，是在隐变量不可见的前提下，通过 E 步和 M 步交替优化对数似然下界。
- GMM 提供概率模型结构，EM 提供可执行的参数估计过程。
- 当前源码中的 `GaussianMixture` 参数设定和 `lower_bound_` 日志，正是这些数学思想在工程层面的直接映射。
