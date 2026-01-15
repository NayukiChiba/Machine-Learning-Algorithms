# 数组创建

> 对应代码: [02_creation.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Numpy/02_creation.py)

## 学习目标

- 掌握各种创建 NumPy 数组的方法
- 理解不同创建方法的适用场景
- 学会创建特殊类型的数组

## 创建方法一览

| 方法                  | 说明                 | 示例                            |
| --------------------- | -------------------- | ------------------------------- |
| `np.array()`          | 从列表创建数组       | `np.array([1,2,3])`             |
| `np.zeros()`          | 创建全零数组         | `np.zeros((3,4))`               |
| `np.ones()`           | 创建全一数组         | `np.ones((2,3))`                |
| `np.eye()`            | 创建单位矩阵         | `np.eye(3)`                     |
| `np.full()`           | 创建填充指定值的数组 | `np.full((2,2), 7)`             |
| `np.arange()`         | 创建等差数列         | `np.arange(0, 10, 2)`           |
| `np.linspace()`       | 创建等间距数列       | `np.linspace(0, 1, 5)`          |
| `np.random.rand()`    | 创建 0-1 随机数组    | `np.random.rand(2,3)`           |
| `np.random.randint()` | 创建随机整数数组     | `np.random.randint(0,10,(3,3))` |

## 从列表创建 (np.array)

```python
# 一维数组
arr_1d = np.array([1, 2, 3, 4, 5])

# 二维数组
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])

# 指定数据类型
arr_float = np.array([1, 2, 3], dtype=np.float64)
```

### 参数说明

| 参数     | 说明                                      |
| -------- | ----------------------------------------- |
| `object` | 输入数据，可以是列表、元组等可迭代对象    |
| `dtype`  | 指定数据类型，如 `np.int32`、`np.float64` |
| `copy`   | 是否复制数据，默认 True                   |

## 特殊数组

### 全零数组 (np.zeros)

```python
np.zeros((3, 4))        # 3x4 全零数组
np.zeros(5)             # 长度为 5 的一维全零数组
np.zeros((2, 3, 4))     # 三维全零数组
```

### 全一数组 (np.ones)

```python
np.ones((2, 3))         # 2x3 全一数组
np.ones(5, dtype=int)   # 整数类型的全一数组
```

### 单位矩阵 (np.eye)

```python
np.eye(3)               # 3x3 单位矩阵
np.eye(3, k=1)          # 上对角线偏移 1
np.eye(3, k=-1)         # 下对角线偏移 1
np.eye(2, 3)            # 2x3 非方阵
```

### 填充数组 (np.full)

```python
np.full((2, 2), 7)      # 2x2 填充 7
np.full((3, 3), np.pi)  # 3x3 填充 π
```

## 序列数组

### 等差数列 (np.arange)

```python
np.arange(10)           # [0, 1, 2, ..., 9]
np.arange(1, 10)        # [1, 2, 3, ..., 9]
np.arange(1, 10, 2)     # [1, 3, 5, 7, 9]
np.arange(10, 0, -1)    # [10, 9, 8, ..., 1]
```

### 等间距数列 (np.linspace)

```python
np.linspace(0, 1, 5)           # [0, 0.25, 0.5, 0.75, 1]
np.linspace(0, 2*np.pi, 10)    # 0 到 2π 的 10 个等间距点
np.linspace(0, 1, 5, endpoint=False)  # 不包含终点
```

> [!TIP]
> `arange` 需要指定步长，`linspace` 需要指定点数。绑图时常用 `linspace` 创建 x 轴。

## 随机数组

### 设置随机种子

```python
np.random.seed(42)  # 保证结果可复现
```

### 均匀分布

```python
np.random.rand(2, 3)           # [0, 1) 均匀分布
np.random.random(size=(2, 3))  # 同上，参数用 size
```

### 随机整数

```python
np.random.randint(0, 10, (3, 3))  # [0, 10) 随机整数
np.random.randint(5, size=10)     # [0, 5) 随机整数
```

### 正态分布

```python
np.random.randn(5)                     # 标准正态分布 (μ=0, σ=1)
np.random.normal(loc=10, scale=2, size=5)  # 指定均值和标准差
```

## 随机函数对比

| 函数                 | 分布             | 范围                   |
| -------------------- | ---------------- | ---------------------- |
| `rand()`             | 均匀分布         | [0, 1)                 |
| `random()`           | 均匀分布         | [0, 1)                 |
| `randint(low, high)` | 均匀分布（整数） | [low, high)            |
| `randn()`            | 标准正态分布     | (-∞, +∞)               |
| `normal(loc, scale)` | 正态分布         | 均值 loc，标准差 scale |

## 练习

运行代码文件查看完整演示：

```bash
python Basic/Numpy/02_creation.py
```
