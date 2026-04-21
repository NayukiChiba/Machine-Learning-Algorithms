---
title: NumPy 综合实战
outline: deep
---

# NumPy 综合实战

## 本章目标

1. 把前 11 章的知识串起来，解决完整的小任务。
2. 学会在真实场景中组合使用索引、统计、线代、广播。
3. 建立"问题 → NumPy 算子"的映射能力。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `np.random.randint(...)` | 函数 | 生成整数随机样本 |
| `np.random.normal(...)` | 函数 | 生成正态分布样本 |
| `arr.sum(...)` / `arr.mean(...)` | 方法 | 聚合统计 |
| `arr.std(...)` / `arr.var(...)` | 方法 | 标准差 / 方差 |
| `arr.min(...)` / `arr.max(...)` | 方法 | 极值 |
| `np.argmax(...)` / `np.argmin(...)` | 函数 | 极值索引 |
| `np.argsort(...)` | 函数 | 排序索引，配合 `[::-1]` 得降序 |
| `np.column_stack(...)` | 函数 | 按列拼接多个一维数组成设计矩阵 |
| `np.linalg.solve(...)` | 函数 | 解线性方程组 |
| `np.rot90(...)` | 函数 | 二维数组旋转 90° |
| `arr.astype(...)` | 方法 | dtype 转换 |
| `np.percentile(...)` | 函数 | 分位数统计 |
| `np.histogram(...)` | 函数 | 区间频数统计 |

## 案例一：学生成绩分析

### 问题描述

给定 5 名学生在 3 门课程上的成绩（`5 × 3` 矩阵），计算每个学生总分与平均分、每门课程的统计指标，并给出总分排名。

### 涉及方法

- `np.sum(axis=1)` / `np.mean(axis=1)`: 每个学生聚合
- `np.mean/std/max/min(axis=0)`: 每门课程聚合
- `np.argmax` / `np.argmin`: 找最高 / 最低分
- `np.argsort(...)[::-1]`: 倒序排名

### 核心代码

```python
import numpy as np

np.random.seed(42)
grades = np.random.randint(60, 101, size=(5, 3))
students = ["学生A", "学生B", "学生C", "学生D", "学生E"]
courses = ["数学", "英语", "物理"]

# 每个学生统计
total_scores = np.sum(grades, axis=1)
avg_scores = np.mean(grades, axis=1)

# 每门课程统计
course_mean = np.mean(grades, axis=0)
course_std = np.std(grades, axis=0)
course_max = np.max(grades, axis=0)
course_min = np.min(grades, axis=0)

# 排名
best_idx = np.argmax(total_scores)
worst_idx = np.argmin(total_scores)
rank_indices = np.argsort(total_scores)[::-1]

print(f"成绩矩阵:\n{grades}")
print(f"总分: {total_scores}")
print(f"平均分: {avg_scores}")
print(f"课程均值: {course_mean}")
print(f"课程标准差: {course_std.round(1)}")
print(f"总分最高: {students[best_idx]} ({total_scores[best_idx]})")
print(f"总分最低: {students[worst_idx]} ({total_scores[worst_idx]})")
print(f"排名: {[students[i] for i in rank_indices]}")
```

### 输出

```text
成绩矩阵:
[[98 88 74]
 [67 80 98]
 [78 82 70]
 [70 83 95]
 [99 83 62]]
总分: [260 245 230 248 244]
平均分: [86.66666667 81.66666667 76.66666667 82.66666667 81.33333333]
课程均值: [82.4 83.2 79.8]
课程标准差: [13.6  2.6 14.2]
总分最高: 学生A (260)
总分最低: 学生C (230)
排名: ['学生A', '学生D', '学生B', '学生E', '学生C']
```

### 理解重点

- `axis=1` 是"沿列方向走" → 每一行得到一个聚合值（学生维度）。
- `axis=0` 是"沿行方向走" → 每一列得到一个聚合值（课程维度）。
- `argsort(...)[::-1]` 是得到降序索引的标准写法。

## 案例二：线性回归（正规方程）

### 问题描述

给定带噪声的一维特征 $x$ 和目标 $y$，用最小二乘法拟合 $y = w_1 x + w_0$。

### 数学推导

最小二乘解：
$$
w = (X^T X)^{-1} X^T y
$$

为保证数值稳定性，**不**直接求逆，而是解正规方程：
$$
(X^T X)\, w = X^T y \quad\Rightarrow\quad w = \text{solve}(X^T X,\ X^T y)
$$

### 涉及方法

- `np.column_stack([np.ones(n), x])`: 构造设计矩阵（含偏置列）
- `X.T @ X`、`X.T @ y`: 矩阵乘法
- `np.linalg.solve(A, b)`: 解 $Aw = b$
- $R^2$ 与 RMSE: 自定义指标计算

### 核心代码

```python
import numpy as np

np.random.seed(42)
n = 50
x = np.linspace(0, 10, n)
true_slope, true_intercept = 2, 1
noise = np.random.normal(0, 1, n)
y = true_slope * x + true_intercept + noise

# 正规方程
X = np.column_stack([np.ones(n), x])
w = np.linalg.solve(X.T @ X, X.T @ y)
intercept, slope = w

# 预测与指标
y_pred = slope * x + intercept
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - ss_res / ss_tot
rmse = np.sqrt(np.mean((y - y_pred) ** 2))

print(f"估计参数: y = {slope:.4f}x + {intercept:.4f}")
print(f"斜率误差: {abs(slope - true_slope):.4f}")
print(f"截距误差: {abs(intercept - true_intercept):.4f}")
print(f"R² = {r2:.4f}")
print(f"RMSE = {rmse:.4f}")
```

### 输出

```text
估计参数: y = 1.9420x + 1.0644
斜率误差: 0.0580
截距误差: 0.0644
R² = 0.9754
RMSE = 0.9084
```

### 理解重点

- **永远不要**写 `np.linalg.inv(X.T @ X) @ X.T @ y`——应改用 `solve`，更快更稳定。
- 设计矩阵第一列是 `np.ones(n)`，对应截距项 $w_0$。
- $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$，越接近 1 越好；RMSE 以目标变量的单位度量误差。

## 案例三：图像矩阵操作（8×8 灰度）

### 问题描述

用 `uint8` 数组模拟一张灰度图像，演示切片翻转、旋转、裁剪、归一化等常见操作。

### 涉及方法

- `np.random.randint(0, 256, size=(8, 8), dtype=np.uint8)`: 生成灰度图
- 切片 `[:, ::-1]` / `[::-1, :]`: 水平 / 垂直翻转
- `np.rot90(m, k=1, axes=(0, 1))`: 逆时针旋转 90°
- `arr[2:6, 2:6]`: 中心裁剪
- `arr.astype(np.float64) / 255.0`: 归一化到 `[0, 1]`

### 核心代码

```python
import numpy as np

np.random.seed(42)
image = np.random.randint(0, 256, size=(8, 8), dtype=np.uint8)

# 统计
print(f"min={image.min()}, max={image.max()}, mean={image.mean():.1f}, std={image.std():.1f}")

# 变换
flipped_h = image[:, ::-1]
flipped_v = image[::-1, :]
rotated = np.rot90(image)
cropped = image[2:6, 2:6]
normalized = image.astype(np.float64) / 255.0

print(f"水平翻转前三行:\n{flipped_h[:3]}")
print(f"旋转 90°:\n{rotated}")
print(f"裁剪 [2:6, 2:6]:\n{cropped}")
print(f"归一化后前两行:\n{normalized[:2].round(2)}")
```

### 输出

```text
min=3, max=245, mean=137.3, std=73.7
裁剪 [2:6, 2:6]:
[[ 99 187  71 212]
 [ 65 153  20  44]
 [240  39 121  24]
 [239  39 214 244]]
归一化后前两行:
[[0.4  0.86 0.88 0.37 0.7  0.24 0.92 0.8 ]
 [0.36 0.01 0.38 0.95 0.05 0.58 0.96 0.18]]
```

### 理解重点

- `uint8` 溢出会回绕（`255 + 1 → 0`）；做运算前先 `astype(np.float64)`。
- 归一化时务必用浮点除以 `255.0`（而不是 `255`），避免整数除法截断。
- 切片（翻转、裁剪）返回**视图**，修改会影响原数组；需要独立数据用 `.copy()`。

## 案例四：统计分析与直方图

### 问题描述

从 $\mathcal{N}(100, 15^2)$ 采样 1000 个样本，计算常用统计量、分位数，并做 10 个区间的直方图统计。

### 涉及方法

- `np.random.normal(loc, scale, size)`: 生成正态样本
- `arr.mean()` / `arr.std()`: 均值 / 标准差
- `np.percentile(a, q)`: 分位数
- `np.histogram(a, bins)`: 区间频数

### 核心代码

```python
import numpy as np

np.random.seed(42)
data = np.random.normal(loc=100, scale=15, size=1000)

print(f"均值: {data.mean():.2f}, 标准差: {data.std():.2f}")
print(f"范围: [{data.min():.2f}, {data.max():.2f}]")

# 分位数
for p in [25, 50, 75, 90, 95, 99]:
    print(f"P{p}: {np.percentile(data, p):.2f}")

# 直方图
hist, edges = np.histogram(data, bins=10)
for i in range(len(hist)):
    print(f"[{edges[i]:.1f}, {edges[i+1]:.1f}): {hist[i]}")
```

### 输出

```text
均值: 100.29, 标准差: 14.68
范围: [51.38, 157.79]
P25: 90.29
P50: 100.38
P75: 109.72
P90: 119.58
P95: 125.15
P99: 134.74
[51.4, 62.0): 4
[62.0, 72.7): 22
[72.7, 83.3): 96
[83.3, 93.9): 228
[93.9, 104.6): 272
[104.6, 115.2): 226
[115.2, 125.9): 104
[125.9, 136.5): 38
[136.5, 147.1): 9
[147.1, 157.8): 1
```

### 理解重点

- 样本均值 / 标准差会围绕真实参数波动；样本量越大越接近真值。
- `np.histogram` 返回 `(counts, edges)`，`edges` 长度比 `counts` **多 1**（区间端点）。
- 画直方图时配合 matplotlib 的 `plt.hist(data, bins=10)` 更直观，但 `np.histogram` 便于编程处理。

## 学习建议

1. 把四个案例**分别拆成函数**，自己重写一遍。
2. 改变随机种子后，观察指标的稳定性变化。
3. 将线性回归案例扩展到**多特征**版本（`X` 变为 `(n, p)` 形状）。
4. 给图像案例加上**阈值分割**（`np.where(image > 128, 255, 0)`）和二值化练习。
5. 统计案例试试换成偏态分布（如 `np.random.lognormal`），观察分位数的变化。

## 小结

- 本章的重点**不是某个具体 API**，而是**组合能力**——把前 11 章的基础算子拼成完整流程。
- 成绩分析靠聚合 + 排序；线性回归靠矩阵运算 + `solve`；图像操作靠切片 + 类型转换；统计分析靠采样 + `histogram`。
- 当你能用 NumPy 实现完整的"数据加载 → 计算 → 统计 → 输出"流水线，NumPy 就真正用起来了。
- 继续学习 Pandas、SciPy、scikit-learn，它们底层都建立在 NumPy 之上，本章掌握的思维方式会一直受用。
