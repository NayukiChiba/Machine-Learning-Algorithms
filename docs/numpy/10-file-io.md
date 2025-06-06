# 文件操作

> 对应代码: [10_file_io.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Numpy/10_file_io.py)

## 学习目标

- 掌握 NumPy 数组的保存和加载方法
- 学会读写文本文件
- 理解不同文件格式的特点

## 文件格式对比

| 格式        | 函数                  | 特点                     |
| ----------- | --------------------- | ------------------------ |
| `.npy`      | `save()/load()`       | 二进制，快速，保留 dtype |
| `.npz`      | `savez()/load()`      | 压缩的多数组文件         |
| `.txt/.csv` | `savetxt()/loadtxt()` | 文本格式，可读性好       |

## 二进制文件 (.npy)

### 保存单个数组

```python
arr = np.random.random((3, 4))

# 保存
np.save('data.npy', arr)

# 加载
loaded = np.load('data.npy')

# 验证
np.array_equal(arr, loaded)  # True
```

### 优点

- ✅ 速度快
- ✅ 文件小
- ✅ 完整保留数据类型
- ✅ 支持任意维度

## 多数组文件 (.npz)

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([[1, 2], [3, 4]])

# 保存多个数组
np.savez('data.npz', a=arr1, b=arr2)

# 加载
data = np.load('data.npz')
print(data.files)  # ['a', 'b']
print(data['a'])   # [1, 2, 3]
print(data['b'])   # [[1, 2], [3, 4]]
```

### 压缩版本

```python
# 使用 gzip 压缩
np.savez_compressed('data.npz', a=arr1, b=arr2)
```

## 文本文件 (.txt/.csv)

### 保存为文本

```python
arr = np.random.random((3, 4))

# 默认格式
np.savetxt('data.txt', arr)

# 自定义格式
np.savetxt('data.csv', arr,
           delimiter=',',    # 分隔符
           fmt='%.4f',       # 格式
           header='A,B,C,D', # 表头
           comments='')      # 不添加注释符

```

### 格式说明符

| 格式     | 说明              | 示例         |
| -------- | ----------------- | ------------ |
| `%.2f`   | 2 位小数          | `3.14`       |
| `%.4f`   | 4 位小数          | `3.1416`     |
| `%d`     | 整数              | `3`          |
| `%.2e`   | 科学计数法        | `3.14e+00`   |
| `%10.4f` | 宽度 10，4 位小数 | `    3.1416` |

### 加载文本文件

```python
# 基本加载
arr = np.loadtxt('data.txt')

# 指定分隔符
arr = np.loadtxt('data.csv', delimiter=',')

# 跳过表头
arr = np.loadtxt('data.csv', delimiter=',', skiprows=1)

# 指定数据类型
arr = np.loadtxt('data.txt', dtype=np.int32)
```

## savetxt 参数详解

```python
np.savetxt(fname,           # 文件路径
           X,               # 要保存的数组
           fmt='%.18e',     # 格式字符串
           delimiter=' ',   # 分隔符
           newline='\n',    # 行分隔符
           header='',       # 文件头
           footer='',       # 文件尾
           comments='# ')   # 注释前缀
```

## loadtxt 参数详解

```python
np.loadtxt(fname,           # 文件路径
           dtype=float,     # 数据类型
           delimiter=None,  # 分隔符（None=空白）
           skiprows=0,      # 跳过开头行数
           usecols=None,    # 读取的列
           unpack=False)    # 是否转置
```

## 使用场景

| 场景              | 推荐方法      |
| ----------------- | ------------- |
| 临时存储/快速 I/O | `.npy`        |
| 存储多个数组      | `.npz`        |
| 与其他程序交换    | `.csv`/`.txt` |
| 需要人工查看      | `.csv`/`.txt` |

> [!TIP]
> 对于大型数组，二进制格式 (`.npy`) 比文本格式快 10 倍以上。

## 练习

运行代码文件查看完整演示：

```bash
python Basic/Numpy/10_file_io.py
```
