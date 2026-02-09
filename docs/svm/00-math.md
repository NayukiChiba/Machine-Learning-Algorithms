# 支持向量机数学基础与核心概念

这一章只讲原理，不涉及代码。读完之后再看后面的实现会更顺畅。

---

## 1. SVM 要解决什么问题

SVM（Support Vector Machine）是**监督学习**中的经典分类算法，核心目标是：

- 找到一个分类超平面
- 让两类样本之间的“间隔”最大
- 只由少数关键样本（支持向量）决定模型

---

## 2. 线性可分的最大间隔

假设数据线性可分，超平面形式：

$$
	ext{f}(x) = \mathbf{w}^T \mathbf{x} + b
$$

要求所有样本满足：

$$
 y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1
$$

最大间隔等价于最小化：

$$
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2
$$

---

## 3. 软间隔与惩罚系数 C

现实数据往往不可完全线性可分，引入松弛变量 $\xi_i$：

$$
 y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0
$$

优化目标变成：

$$
\min_{\mathbf{w}, b, \xi} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^n \xi_i
$$

- `C` 越大：更强调训练集正确，容易过拟合
- `C` 越小：更强调间隔，可能欠拟合

---

## 4. 对偶问题与支持向量

对偶形式：

$$
\max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j y_i y_j \langle x_i, x_j \rangle
$$

约束：

$$
0 \le \alpha_i \le C, \quad \sum_{i=1}^n \alpha_i y_i = 0
$$

- $\alpha_i > 0$ 的样本就是**支持向量**
- 分类超平面只由这些样本决定

---

## 5. 核技巧（Kernel Trick）

当数据不可线性分割时，引入核函数：

$$
K(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle
$$

常见核函数：

- 线性核 `linear`
- 高斯核 `rbf`
- 多项式核 `poly`
- Sigmoid 核 `sigmoid`

使用核函数后，决策函数变成：

$$
 f(x) = \text{sign}\left(\sum_i \alpha_i y_i K(x_i, x) + b\right)
$$

---

## 6. RBF 核的意义

RBF（高斯核）是最常用的非线性核：

$$
K(x_1, x_2) = \exp(-\gamma \|x_1 - x_2\|^2)
$$

- `gamma` 越大：模型更“弯曲”，更容易过拟合
- `gamma` 越小：模型更平滑，可能欠拟合

---

## 7. 为什么要标准化

SVM 对特征尺度非常敏感：

- 不同量纲会导致距离计算失真
- 影响支持向量的位置与决策边界形状

因此在代码中必须做标准化。

---

## 8. 你在这个项目里会用到什么

- `make_moons` 生成非线性二分类数据
- `SVC` 训练 SVM 分类器
- `C`、`kernel`、`gamma` 控制模型复杂度
- 用 Accuracy / Precision / Recall / F1 做评估
- 可视化决策边界和支持向量

读完这一章后，继续看代码部分即可。
