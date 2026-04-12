---
title: SVR 支持向量回归 — 数学原理
outline: deep
---

# 数学原理

> 对应代码：`model_training/regression/svr.py`
>  
> 相关对象：`train_model(...)`、`sklearn.svm.SVR`

## 本章目标

1. 理解 SVR 的 `epsilon`-不敏感损失与间隔控制思想。
2. 看清目标函数中正则项与松弛变量的作用分工。
3. 将数学符号与仓库中的 `train_model(...)` 参数逐一对应。

## 重点概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `epsilon`-不敏感损失 | 数学概念 | 控制多大范围内的误差不计入损失 |
| `C` | 超参数 | 控制模型平滑度与误差惩罚之间的权衡 |
| 核函数 `K(x_i, x)` | 数学概念 | 把线性内积扩展到非线性特征空间 |
| `kernel='rbf'` | 源码默认配置 | 当前实现默认采用的非线性核 |

## 1. `epsilon`-不敏感损失函数

SVR 的核心思想，是构造一个以 `epsilon` 为半宽的“管道”。当预测误差落在这个管道内部时，不计损失；只有超出管道的部分才会被线性惩罚。

$$
L_\epsilon(y, f(\mathbf{x})) = \max(0, |y - f(\mathbf{x})| - \epsilon)
$$

### 理解重点

- 当误差在 `[-epsilon, +epsilon]` 内时，损失为 0。
- `epsilon` 越大，模型容忍的小误差范围越宽，通常会让模型更平滑。

## 2. 原始优化问题

$$
\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{N} (\xi_i + \xi_i^*)
$$

$$
\text{s.t.} \quad
\begin{cases}
y_i - \mathbf{w}^T\mathbf{x}_i - b \leq \epsilon + \xi_i \\
\mathbf{w}^T\mathbf{x}_i + b - y_i \leq \epsilon + \xi_i^* \\
\xi_i, \xi_i^* \geq 0
\end{cases}
$$

### 理解重点

- `\frac{1}{2}\|\mathbf{w}\|^2` 控制模型平滑度。
- `C\sum(\xi_i + \xi_i^*)` 控制超出管道样本的惩罚强度。
- `\xi_i` 和 `\xi_i^*` 分别表示落在管道上方和下方之外的偏离量。

## 3. 对偶问题与核技巧

引入拉格朗日乘子 `\alpha_i, \alpha_i^*` 后，可以得到对偶形式：

$$
\max_{\boldsymbol{\alpha}, \boldsymbol{\alpha}^*} -\frac{1}{2}\sum_{i,j}(\alpha_i - \alpha_i^*)(\alpha_j - \alpha_j^*)\mathbf{x}_i^T\mathbf{x}_j
- \epsilon\sum_i(\alpha_i + \alpha_i^*) + \sum_i y_i(\alpha_i - \alpha_i^*)
$$

$$
\text{s.t.} \quad \sum_i(\alpha_i - \alpha_i^*) = 0, \quad 0 \leq \alpha_i, \alpha_i^* \leq C
$$

把内积 `\mathbf{x}_i^T\mathbf{x}_j` 换成核函数 `K(\mathbf{x}_i, \mathbf{x}_j)` 后，SVR 就可以处理非线性关系。

### 理解重点

- 核技巧让模型不必显式构造高维特征，也能拟合复杂非线性关系。
- 当前仓库默认 `kernel='rbf'`，正是为了适配 `svr_data` 的非线性结构。

## 4. 预测函数

$$
f(\mathbf{x}) = \sum_{i=1}^{N}(\alpha_i - \alpha_i^*)K(\mathbf{x}_i, \mathbf{x}) + b
$$

### 理解重点

- 真正参与预测的样本主要是支持向量。
- 这也是为什么源码训练结束后会特别打印“支持向量数量”。

## 5. 数学符号与源码参数映射

### 参数速览（本节）

适用函数：`train_model(X_train, y_train, C=10.0, epsilon=0.1, kernel='rbf', gamma='scale', degree=3, coef0=0.0)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `C` | `10.0` | 正则化强度的倒数，越大越强调拟合训练样本 |
| `epsilon` | `0.1` | 不敏感区间半宽，对应 `epsilon`-tube |
| `kernel` | `'rbf'` | 决定使用何种核函数 |
| `gamma` | `'scale'` | 控制核函数影响范围 |
| `degree` | `3` | 多项式核阶数，仅 `kernel='poly'` 时主要起作用 |
| `coef0` | `0.0` | 多项式核与 sigmoid 核常数项 |
| 返回值 | `SVR` 模型对象 | 已完成 `fit` 的回归器 |

### 理解重点

- 数学上最关键的两个量是 `C` 和 `epsilon`，源码中也正是通过这两个参数直接暴露给用户。
- `gamma` 决定核函数的局部性，对 RBF 核效果影响很大。

## 常见坑

1. 只记公式，不和源码里的默认参数对应起来。
2. 误以为 `C` 越大一定越好，实际上它会带来更强的拟合倾向和更高的过拟合风险。
3. 忽略核函数与数据结构的匹配关系。

## 小结

- SVR 的数学核心，是“平滑度”和“误差容忍度”之间的权衡。
- 在本仓库里，这种权衡主要通过 `C`、`epsilon`、`kernel`、`gamma` 体现。
- 数学理解和源码阅读结合起来看，最容易形成稳定的调参直觉。
