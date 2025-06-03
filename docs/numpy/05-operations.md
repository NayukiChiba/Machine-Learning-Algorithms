# 数组运算

> 对应代码: [05_operations.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Numpy/05_operations.py)

## 学习目标

- 掌握 NumPy 数组的基本数学运算
- 理解向量化操作的优势
- 学会使用比较运算和逻辑运算

## 运算类型

| 类型     | 运算符/函数                         | 说明         |
| -------- | ----------------------------------- | ------------ |
| 算术运算 | `+`, `-`, `*`, `/`, `**`, `//`, `%` | 元素级运算   |
| 比较运算 | `>`, `<`, `>=`, `<=`, `==`, `!=`    | 返回布尔数组 |
| 逻辑运算 | `&`, `\|`, `~`, `np.logical_and()`  | 布尔数组运算 |
| 统计运算 | `sum()`, `mean()`, `std()`, `var()` | 聚合运算     |

## 向量化的优势

```python
# Python 循环方式（慢）
result = []
for i in range(len(a)):
    result.append(a[i] + b[i])

# NumPy 向量化方式（快）
result = a + b
```

> [!TIP]
> 向量化操作比 Python 循环快 10-100 倍！

## 算术运算

```python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

a + b   # [6, 8, 10, 12]
a - b   # [-4, -4, -4, -4]
a * b   # [5, 12, 21, 32]
a / b   # [0.2, 0.33, 0.43, 0.5]

a ** 2    # [1, 4, 9, 16] (平方)
a ** 0.5  # [1, 1.41, 1.73, 2] (平方根)
a // 2    # [0, 1, 1, 2] (整除)
a % 2     # [1, 0, 1, 0] (取余)
```

## 比较运算

```python
a = np.array([1, 2, 3, 4])
b = np.array([4, 3, 2, 1])

# 元素级比较
a == b  # [False, False, False, False]
a != b  # [True, True, True, True]
a > b   # [False, False, True, True]
a < b   # [True, True, False, False]

# 数组整体比较
np.array_equal(a, b)  # False
np.any(a == b)        # False (任意元素相同)
np.all(a != b)        # True (所有元素不同)
```

## 统计运算

| 函数                   | 说明            |
| ---------------------- | --------------- |
| `sum()`                | 求和            |
| `mean()`               | 均值            |
| `std()`                | 标准差          |
| `var()`                | 方差            |
| `min()`, `max()`       | 最小/最大值     |
| `argmin()`, `argmax()` | 最小/最大值索引 |
| `cumsum()`             | 累积和          |
| `cumprod()`            | 累积积          |

```python
arr = np.array([1, 2, 3, 4, 5])

arr.sum()     # 15
arr.mean()    # 3.0
arr.std()     # 1.41
arr.var()     # 2.0
arr.min()     # 1
arr.max()     # 5
arr.argmin()  # 0
arr.argmax()  # 4
arr.cumsum()  # [1, 3, 6, 10, 15]
```

## 沿轴运算 (axis)

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# axis=None: 所有元素
arr.sum()          # 45

# axis=0: 沿行方向（按列计算）
arr.sum(axis=0)    # [12, 15, 18]
arr.mean(axis=0)   # [4, 5, 6]

# axis=1: 沿列方向（按行计算）
arr.sum(axis=1)    # [6, 15, 24]
arr.mean(axis=1)   # [2, 5, 8]
```

## 数学函数

### 三角函数

```python
angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])

np.sin(angles)  # 正弦
np.cos(angles)  # 余弦
np.tan(angles)  # 正切
```

### 指数和对数

```python
arr = np.array([1, 2, 3, 4, 5])

np.exp(arr)     # e^x
np.log(arr)     # 自然对数
np.log10(arr)   # 以 10 为底
np.log2(arr)    # 以 2 为底
```

### 取整函数

```python
arr = np.array([1.2, 2.5, 3.7, -1.2, -2.5])

np.floor(arr)   # 向下取整
np.ceil(arr)    # 向上取整
np.round(arr)   # 四舍五入
np.abs(arr)     # 绝对值
```

## 逻辑运算

```python
a = np.array([True, True, False, False])
b = np.array([True, False, True, False])

np.logical_and(a, b)  # [True, False, False, False]
np.logical_or(a, b)   # [True, True, True, False]
np.logical_not(a)     # [False, False, True, True]
np.logical_xor(a, b)  # [False, True, True, False]
```

## 练习

运行代码文件查看完整演示：

```bash
python Basic/Numpy/05_operations.py
```
