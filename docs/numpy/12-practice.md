# 综合实战

> 对应代码: [12_practice.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Numpy/12_practice.py)

## 学习目标

- 综合运用所学的 NumPy 知识
- 解决实际问题
- 提高编程能力

## 实战项目

| 项目         | 涉及知识点                     |
| ------------ | ------------------------------ |
| 学生成绩分析 | 数组创建、统计运算、排序、索引 |
| 线性回归实现 | 线性代数、矩阵运算、统计分析   |
| 图像操作模拟 | 数组变形、切片、翻转           |
| 统计分析     | 随机数、百分位数、直方图       |

---

## 项目 1: 学生成绩分析

### 任务目标

分析 5 名学生 3 门课程的成绩：

- 计算每个学生的总分和平均分
- 计算每门课程的统计信息
- 找出最高分和最低分的学生
- 按总分排名

### 示例代码

```python
np.random.seed(42)

# 创建成绩数据 (5 学生 x 3 课程)
grades = np.random.randint(60, 101, size=(5, 3))
students = ['学生A', '学生B', '学生C', '学生D', '学生E']
courses = ['数学', '英语', '物理']

# 计算每个学生的总分和平均分
total_scores = np.sum(grades, axis=1)
avg_scores = np.mean(grades, axis=1)

# 计算每门课程的统计信息
course_mean = np.mean(grades, axis=0)
course_std = np.std(grades, axis=0)

# 找出总分最高的学生
best_idx = np.argmax(total_scores)
print(f"总分最高: {students[best_idx]}")

# 按总分排名
rank_indices = np.argsort(total_scores)[::-1]
for rank, idx in enumerate(rank_indices, 1):
    print(f"第{rank}名: {students[idx]}")
```

---

## 项目 2: 线性回归实现

### 任务目标

使用正规方程实现最小二乘法线性回归：

- 生成带噪声的线性数据
- 计算回归系数
- 评估模型性能 (R², RMSE)

### 正规方程

$$
\hat{\mathbf{w}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

### 示例代码

```python
np.random.seed(42)

# 生成数据: y = 2x + 1 + 噪声
n = 50
x = np.linspace(0, 10, n)
y = 2 * x + 1 + np.random.normal(0, 1, n)

# 构建设计矩阵
X = np.column_stack([np.ones(n), x])

# 正规方程求解
XTX = X.T @ X
XTy = X.T @ y
w = np.linalg.solve(XTX, XTy)

intercept, slope = w[0], w[1]
print(f"估计: y = {slope:.4f}x + {intercept:.4f}")

# 计算 R²
y_pred = slope * x + intercept
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - ss_res / ss_tot
print(f"R²: {r_squared:.4f}")
```

---

## 项目 3: 图像操作模拟

### 任务目标

模拟基本图像操作：

- 图像翻转（水平、垂直）
- 图像旋转
- 图像裁剪
- 图像归一化

### 示例代码

```python
# 创建模拟图像 (8x8 灰度图)
image = np.random.randint(0, 256, size=(8, 8), dtype=np.uint8)

# 水平翻转
flipped_h = image[:, ::-1]

# 垂直翻转
flipped_v = image[::-1, :]

# 旋转 90 度
rotated = np.rot90(image)

# 裁剪
cropped = image[2:6, 2:6]

# 归一化到 [0, 1]
normalized = image.astype(np.float64) / 255.0
```

---

## 项目 4: 统计分析

### 任务目标

对正态分布数据进行统计分析：

- 计算基本统计量
- 计算百分位数
- 生成直方图

### 示例代码

```python
np.random.seed(42)

# 生成正态分布数据
data = np.random.normal(loc=100, scale=15, size=1000)

# 基本统计量
print(f"均值: {data.mean():.2f}")
print(f"标准差: {data.std():.2f}")
print(f"最小值: {data.min():.2f}")
print(f"最大值: {data.max():.2f}")

# 百分位数
for p in [25, 50, 75, 90, 95]:
    print(f"第{p}百分位: {np.percentile(data, p):.2f}")

# 直方图
hist, bin_edges = np.histogram(data, bins=10)
```

---

## 总结：常用技巧速查

```python
# 快速创建
np.zeros((3, 4))          # 全零
np.ones((3, 4))           # 全一
np.arange(0, 10, 2)       # 等差数列
np.linspace(0, 1, 5)      # 等分数列

# 形状操作
arr.reshape(3, 4)         # 变形
arr.flatten()             # 展平（副本）
arr.T                     # 转置

# 统计运算
arr.sum(axis=0)           # 按列求和
arr.mean(axis=1)          # 按行求均值
arr.max(), arr.argmax()   # 最大值及索引

# 条件筛选
arr[arr > 5]              # 布尔索引
np.where(arr > 5)         # 返回索引
np.where(arr > 5, 1, 0)   # 条件替换
```

---

## 进阶学习建议

1. **深入学习**：
   - NumPy 官方文档：https://numpy.org/doc/
   - 结构化数组（Structured Arrays）
   - 内存布局（C-order vs Fortran-order）

2. **相关库学习**：
   - **Pandas**：数据分析和处理
   - **Matplotlib**：数据可视化
   - **Scikit-learn**：机器学习
   - **SciPy**：科学计算

3. **实践项目**：
   - 图像处理（图像本质是 NumPy 数组）
   - 数据清洗和预处理
   - 简单的机器学习算法实现

---

🎉 **恭喜你完成了 NumPy 学习教程！**

运行完整演示：

```bash
python Basic/Numpy/12_practice.py
```
