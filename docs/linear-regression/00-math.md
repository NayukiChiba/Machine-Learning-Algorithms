# 线性回归数学基础与推导

这一章只讲原理，不涉及代码。你读完之后，再看后面的代码会更顺畅。

---

## 1. 什么是线性回归

线性回归是一种**监督学习**的回归算法，用来预测连续值。它假设：

- 输出变量 `y` 可以用输入特征的**线性组合**表示。
- 通过最小化误差来估计参数。

直观理解：
- 一元线性回归就是“拟合一条直线”。
- 多元线性回归就是“在高维空间拟合一个超平面”。

---

## 2. 模型形式

### 2.1 一元线性回归

\[
\hat{y} = \beta_0 + \beta_1 x
\]

- \(\beta_0\)：截距（常数项）
- \(\beta_1\)：斜率（特征权重）

### 2.2 多元线性回归

\[
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p
\]

---

## 3. 矩阵表示

为了推导方便，我们把偏置项 \(\beta_0\) 合并进参数向量：

\[
\mathbf{X} = \begin{bmatrix}
1 & x_{11} & x_{12} & \cdots & x_{1p} \\
1 & x_{21} & x_{22} & \cdots & x_{2p} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & x_{n2} & \cdots & x_{np}
\end{bmatrix}
\]

\[
\boldsymbol{\beta} = \begin{bmatrix}\beta_0 \\ \beta_1 \\ \vdots \\ \beta_p\end{bmatrix}
\]

预测写成：

\[
\hat{\mathbf{y}} = \mathbf{X}\boldsymbol{\beta}
\]

---

## 4. 损失函数（最小二乘）

我们希望预测值 \(\hat{y}\) 与真实值 \(y\) 的差越小越好。

### 4.1 SSE / MSE

**SSE（误差平方和）**：
\[
\text{SSE} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
\]

**MSE（均方误差）**：
\[
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
\]

向量形式：
\[
J(\boldsymbol{\beta}) = \frac{1}{n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2
\]

---

## 5. 正规方程推导（闭式解）

### 5.1 目标函数

\[
J(\boldsymbol{\beta}) = \frac{1}{n}(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})
\]

### 5.2 对参数求导

\[
\frac{\partial J}{\partial \boldsymbol{\beta}} = -\frac{2}{n}\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})
\]

### 5.3 令导数为 0

\[
\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}
\]

### 5.4 解出参数

\[
\boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
\]

> 这就是正规方程（Normal Equation）。

**注意**：如果 \(\mathbf{X}^T\mathbf{X}\) 不可逆，可以用伪逆或 SVD 求解。

---

## 6. 梯度下降（数值解）

当数据规模大或矩阵不可逆时，用梯度下降更稳：

### 6.1 更新公式

\[
\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \alpha \cdot \nabla J(\boldsymbol{\beta})
\]

\[
\nabla J(\boldsymbol{\beta}) = -\frac{2}{n}\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})
\]

### 6.2 学习率 \(\alpha\)

- 太小：收敛慢
- 太大：可能发散

---

## 7. 线性回归的概率视角

如果假设误差 \(\epsilon\) 服从高斯分布：

\[
\epsilon \sim \mathcal{N}(0, \sigma^2)
\]

则最大似然估计会导出最小二乘解。也就是说：

> 最小二乘 = 高斯噪声下的最大似然估计。

---

## 8. 评价指标（用于代码评估部分）

**MSE**：
\[
\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2
\]

**RMSE**：
\[
\text{RMSE} = \sqrt{\text{MSE}}
\]

**MAE**：
\[
\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|
\]

**R²**：
\[
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
\]

---

## 9. 线性回归常见假设

- **线性关系**：特征与目标之间线性可表示
- **误差独立**：样本之间不相关
- **同方差**：误差的方差稳定
- **误差正态**：误差近似高斯分布
- **无严重多重共线性**：特征之间不能高度相关

> 这些假设不满足时，模型可信度会下降。

---

## 10. 特征缩放的意义

线性回归的闭式解对缩放不敏感，但以下情况推荐标准化：

- 梯度下降优化
- 需要比较系数大小时
- 特征量纲差异过大时

标准化公式：

\[
 x' = \frac{x - \mu}{\sigma}
\]

---

## 11. 过拟合与欠拟合

- **欠拟合**：模型太简单，训练集表现差
- **过拟合**：训练集很好，测试集差

检查方式：训练集 R² 与测试集 R² 差距过大。

---

## 12. 正则化（延伸阅读）

### 12.1 Ridge（L2）

\[
J(\boldsymbol{\beta}) = \text{MSE} + \lambda \sum \beta_i^2
\]

### 12.2 Lasso（L1）

\[
J(\boldsymbol{\beta}) = \text{MSE} + \lambda \sum |\beta_i|
\]

- Ridge：更稳定，降低方差
- Lasso：可做特征选择

---

## 13. 你在这个项目里会用到什么

- 使用 `sklearn.linear_model.LinearRegression` 拟合参数
- 用 `MSE / RMSE / MAE / R²` 评估模型
- 可视化帮助诊断线性关系和残差结构

读完这一章后，继续看代码部分即可。
