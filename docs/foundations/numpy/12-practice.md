---
title: NumPy 综合练习
outline: deep
---

# NumPy 综合练习

> 对应脚本：`Basic/Numpy/12_practice.py`  
> 运行方式：`python Basic/Numpy/12_practice.py`

## 本章目标

1. 把前 11 章知识串起来，解决完整小任务。
2. 学会在真实场景中组合使用索引、统计、线代、广播。
3. 建立“问题到 NumPy 算子”的映射能力。

## 案例 1：学生成绩分析

### 涉及方法

- `np.sum(..., axis=1)`：每个学生总分
- `np.mean(..., axis=1)`：每个学生平均分
- `np.mean/std/max/min(..., axis=0)`：每门课程统计
- `np.argmax` / `np.argmin`：找最高分/最低分
- `np.argsort(...)[::-1]`：倒序排名

### 参数速览（本节）

适用 API（分项）：

1. `np.random.randint(low, high=None, size=None, dtype=int)`
2. `np.sum(a, axis=...)`
3. `np.mean(a, axis=...)`
4. `np.std(a, axis=...)`
5. `np.argsort(a, axis=-1)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `low` / `high` / `size`（`random.randint`） | `60` / `101` / `(5,3)` | 生成 5 名学生 3 门课程的随机成绩 |
| `axis`（`sum/mean/std`） | `1`、`0` | `axis=1` 按学生聚合，`axis=0` 按课程聚合 |
| `axis`（`argsort`） | `-1`（默认） | 返回升序索引，配合 `[::-1]` 得到降序排名 |
### 核心代码

```python
import numpy as np

np.random.seed(42)
grades = np.random.randint(60, 101, size=(5, 3))

total_scores = np.sum(grades, axis=1)
avg_scores = np.mean(grades, axis=1)

course_mean = np.mean(grades, axis=0)
course_std = np.std(grades, axis=0)

rank_indices = np.argsort(total_scores)[::-1]
```

### 结果输出（关键）

```text
成绩矩阵:
----------------
[[98 88 74]
 [67 80 98]
 [78 82 70]
 [70 83 95]
 [99 83 62]]
----------------
总分: [260 245 230 248 244]
----------------
平均分: [86.7 81.7 76.7 82.7 81.3]
----------------
课程均值: [82.4 83.2 79.8]
----------------
课程标准差: [13.6  2.6 14.2]
----------------
总分最高: 学生A (260)
----------------
总分最低: 学生C (230)
```

## 案例 2：线性回归（正规方程）

### 涉及方法

- `np.column_stack`：构建设计矩阵 $X$
- `@`：矩阵乘法
- `np.linalg.solve`：求解正规方程
- 自定义指标：$R^2$、RMSE

### 数学形式

$$
w = (X^T X)^{-1} X^T y
$$

脚本中为了数值稳定与实现简洁，写成：

$$
X^TX\,w = X^Ty \quad \Rightarrow \quad w = \text{solve}(X^TX, X^Ty)
$$

### 参数速览（本节）

适用 API（分项）：

1. `np.linspace(start, stop, num=50, endpoint=True)`
2. `np.random.normal(loc=0, scale=1, size=n)`
3. `np.column_stack(tup)`
4. `np.linalg.solve(a, b)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `start` / `stop` / `num`（`linspace`） | `0` / `10` / `50` | 生成 50 个等间距特征点 |
| `loc` / `scale` / `size`（`random.normal`） | `0` / `1` / `50` | 生成高斯噪声 |
| `tup`（`column_stack`） | `[np.ones(n), x]` | 拼接偏置列和特征列形成设计矩阵 |
| `a` / `b`（`linalg.solve`） | `X.T @ X` / `X.T @ y` | 求解正规方程组 `a @ w = b` |
### 核心代码

```python
import numpy as np

np.random.seed(42)
n = 50
x = np.linspace(0, 10, n)
y = 2 * x + 1 + np.random.normal(0, 1, n)

X = np.column_stack([np.ones(n), x])
w = np.linalg.solve(X.T @ X, X.T @ y)

intercept, slope = w
```

### 结果输出（关键）

```text
估计参数: y = 1.9420x + 1.0644
----------------
斜率误差: 0.0580
----------------
截距误差: 0.0644
----------------
R²: 0.9754
----------------
RMSE: 0.9084
----------------
x=5  -> 预测 10.77, 真实 11.00
----------------
x=10 -> 预测 20.48, 真实 21.00
----------------
x=15 -> 预测 30.19, 真实 31.00
```

## 案例 3：图像矩阵操作（8x8 灰度）

### 涉及方法

- 切片：翻转、裁剪
- `np.rot90`：旋转
- `astype(np.float64) / 255.0`：归一化
- `min/max/mean/std`：像素统计

### 参数速览（本节）

适用 API/语法（分项）：

1. `np.random.randint(low, high=None, size=None, dtype=int)`
2. `np.rot90(m, k=1, axes=(0, 1))`
3. `arr.astype(dtype, copy=True)`
4. 切片 `start:stop:step`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `low` / `high` / `size` / `dtype`（`random.randint`） | `0` / `256` / `(8,8)` / `np.uint8` | 生成 8x8 灰度图像矩阵 |
| `k` / `axes`（`rot90`） | `1` / `(0,1)` | 在二维平面上逆时针旋转 90 度 |
| `dtype` / `copy`（`astype`） | `np.float64` / `True` | 转浮点后便于做 `0~1` 归一化 |
| 切片位（切片语法） | `[:, ::-1]`、`[2:6, 2:6]` | 实现水平翻转与中心裁剪 |
### 核心代码

```python
import numpy as np

np.random.seed(42)
image = np.random.randint(0, 256, size=(8, 8), dtype=np.uint8)

flipped_h = image[:, ::-1]
rotated = np.rot90(image)
cropped = image[2:6, 2:6]
normalized = image.astype(np.float64) / 255.0
```

### 结果输出（关键）

```text
图像统计:
----------------
min=3, max=245, mean=137.3, std=73.7
----------------
裁剪 [2:6, 2:6]:
----------------
[[ 99 187  71 212]
 [ 65 153  20  44]
 [240  39 121  24]
 [239  39 214 244]]
----------------
归一化后前两行:
----------------
[[0.4  0.86 0.88 0.37 0.7  0.24 0.92 0.8 ]
 [0.36 0.01 0.38 0.95 0.05 0.58 0.96 0.18]]
```

## 案例 4：统计分析与直方图

### 涉及方法

- `np.random.normal`：模拟正态样本
- `np.percentile`：分位数
- `np.histogram`：区间统计

### 参数速览（本节）

适用 API（分项）：

1. `np.random.normal(loc=100, scale=15, size=1000)`
2. `np.percentile(a, q, axis=None, method='linear', keepdims=False)`
3. `np.histogram(a, bins=10, range=None, density=False, weights=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `loc` / `scale` / `size`（`random.normal`） | `100` / `15` / `1000` | 生成模拟分数样本 |
| `q`（`percentile`） | `25, 50, 75, 90, 95, 99` | 计算多个分位点统计量 |
| `method` / `axis`（`percentile`） | `'linear'` / `None` | 默认线性插值，按整体数据计算 |
| `bins`（`histogram`） | `10` | 把样本划分为 10 个区间并计数 |
### 核心代码

```python
import numpy as np

np.random.seed(42)
data = np.random.normal(loc=100, scale=15, size=1000)

percentiles = [25, 50, 75, 90, 95, 99]
values = [np.percentile(data, p) for p in percentiles]

hist, bin_edges = np.histogram(data, bins=10)
```

### 结果输出（关键）

```text
样本均值: 100.29
----------------
样本标准差: 14.68
----------------
最小值: 51.38
----------------
最大值: 157.79
----------------
分位数:
----------------
P25=90.29, P50=100.38, P75=109.72,
----------------
P90=119.58, P95=125.15, P99=134.74
----------------
直方图区间计数（10 bins）:
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

## 学习建议

1. 把四个案例分别拆成函数，自己重写一遍。
2. 改随机种子后，观察指标稳定性变化。
3. 把线性回归案例改成多特征版本（二维以上 `X`）。
4. 给图像案例加阈值分割和二值化练习。

## 小结

- 这章的重点不是某个 API，而是“组合能力”。
- 当你能把基础算子拼成完整流程，NumPy 就真正用起来了。