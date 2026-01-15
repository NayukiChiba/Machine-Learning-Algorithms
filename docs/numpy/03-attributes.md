# 数组属性

> 对应代码: [03_attributes.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Numpy/03_attributes.py)

## 学习目标

- 掌握 NumPy 数组的重要属性
- 理解数组形状、维度、数据类型等概念
- 学会查看和修改数组属性

## 重要属性一览

| 属性       | 说明                     | 示例结果  |
| ---------- | ------------------------ | --------- |
| `shape`    | 数组的形状（各维度大小） | `(3, 4)`  |
| `ndim`     | 数组的维度数             | `2`       |
| `size`     | 数组的元素总数           | `12`      |
| `dtype`    | 数组的数据类型           | `float64` |
| `itemsize` | 每个元素的字节大小       | `8`       |
| `nbytes`   | 数组总字节数             | `96`      |

## 形状属性

```python
arr = np.random.random((3, 4))

# 形状 shape - 返回各维度大小的元组
print(arr.shape)      # (3, 4)
print(arr.shape[0])   # 3 (行数)
print(arr.shape[1])   # 4 (列数)

# 维度数 ndim
print(arr.ndim)       # 2

# 元素总数 size
print(arr.size)       # 12
```

## 内存属性

```python
arr = np.random.random((3, 4))

# 数据类型
print(arr.dtype)      # float64

# 每个元素的字节大小
print(arr.itemsize)   # 8

# 总字节数
print(arr.nbytes)     # 96 = 12 * 8
```

## 常用数据类型

### 整数类型

| 类型    | 说明      | 范围           |
| ------- | --------- | -------------- |
| `int8`  | 8 位整数  | -128 ~ 127     |
| `int16` | 16 位整数 | -32768 ~ 32767 |
| `int32` | 32 位整数 | -2³¹ ~ 2³¹-1   |
| `int64` | 64 位整数 | -2⁶³ ~ 2⁶³-1   |

### 浮点类型

| 类型      | 说明        | 精度           |
| --------- | ----------- | -------------- |
| `float16` | 16 位浮点数 | 半精度         |
| `float32` | 32 位浮点数 | 单精度         |
| `float64` | 64 位浮点数 | 双精度（默认） |

### 其他类型

| 类型         | 说明                  |
| ------------ | --------------------- |
| `bool`       | 布尔类型 (True/False) |
| `complex64`  | 复数 (2 个 float32)   |
| `complex128` | 复数 (2 个 float64)   |

## 数据类型转换 (astype)

```python
# 创建浮点数组
arr_float = np.array([1.5, 2.7, 3.2, 4.8])
print(arr_float.dtype)  # float64

# 转换为整数（截断）
arr_int = arr_float.astype(np.int32)
print(arr_int)  # [1, 2, 3, 4]

# 转换为字符串
arr_str = arr_float.astype(str)
print(arr_str)  # ['1.5', '2.7', '3.2', '4.8']
```

> [!WARNING]
> `astype()` 返回新数组，不会修改原数组。浮点转整数会截断小数部分。

## 布尔数组

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 通过比较运算创建布尔数组
bool_arr = arr > 5
print(bool_arr)       # [False False False False False  True  True  True  True  True]
print(bool_arr.dtype) # bool

# 布尔数组可用于索引
print(arr[bool_arr])  # [6, 7, 8, 9, 10]

# 统计 True 的数量
print(bool_arr.sum()) # 5
```

## 查看数据类型信息

```python
# 整数类型信息
print(np.iinfo(np.int32))  # 显示范围

# 浮点类型信息
print(np.finfo(np.float64))  # 显示精度
```

## 练习

运行代码文件查看完整演示：

```bash
python Basic/Numpy/03_attributes.py
```
