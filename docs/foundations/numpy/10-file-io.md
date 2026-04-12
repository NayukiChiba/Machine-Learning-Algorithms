---
title: NumPy 文件 IO
outline: deep
---

# NumPy 文件 IO

> 对应脚本：`Basic/Numpy/10_file_io.py`  
> 运行方式：`python Basic/Numpy/10_file_io.py`

## 本章目标

1. 掌握二进制保存/加载：`save`、`load`、`savez`。
2. 掌握文本保存/加载：`savetxt`、`loadtxt`。
3. 掌握格式控制参数：`delimiter`、`fmt`、`header`、`skiprows`。

## 重点方法速览

| 方法 | 用途 |
|---|---|
| `np.save(file, arr)` | 保存单个 `.npy` |
| `np.load(file)` | 加载 `.npy` / `.npz` |
| `np.savez(file, **kwargs)` | 保存多个数组到 `.npz` |
| `np.savetxt(file, arr, ...)` | 保存文本文件 |
| `np.loadtxt(file, ...)` | 从文本读取数组 |

## 1. `.npy`：`save` + `load`

### 参数速览（本节）

适用 API（分项）：

1. `np.save(file, arr, allow_pickle=True, fix_imports=...)`
2. `np.load(file, mmap_mode=None, allow_pickle=False, ...)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `file` / `arr`（`save`） | `"array.npy"` / `arr(3x4)` | 保存单个数组为 `.npy` |
| `allow_pickle`（`save`） | `True`（默认） | 控制 object 数组序列化 |
| `file`（`load`） | `"array.npy"` | 读取 `.npy` 文件 |
| `mmap_mode` / `allow_pickle`（`load`） | `None` / `False` | 默认普通读取且更安全 |
### 示例代码

```python
import numpy as np

np.random.seed(42)
arr = np.random.random((3, 4))

np.save("array.npy", arr)
loaded = np.load("array.npy")

print(arr)
print(loaded)
print(np.array_equal(arr, loaded))
```

### 结果输出

```text
[[0.37454012 0.95071431 0.73199394 0.59865848]
 [0.15601864 0.15599452 0.05808361 0.86617615]
 [0.60111501 0.70807258 0.02058449 0.96990985]]
----------------
[[0.37454012 0.95071431 0.73199394 0.59865848]
 [0.15601864 0.15599452 0.05808361 0.86617615]
 [0.60111501 0.70807258 0.02058449 0.96990985]]
----------------
True
```

### 理解重点

- `.npy` 速度快、保真高，适合 NumPy 内部数据存储。

## 2. `.npz`：`savez` 保存多个数组

### 参数速览（本节）

适用 API（分项）：

1. `np.savez(file, *args, **kwds)`
2. `np.load(file)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `file`（`savez`） | `"arrays.npz"` | 保存多个数组到压缩包容器 |
| `**kwds`（`savez`） | `a=arr1, b=arr2, c=arr3` | 具名键便于后续按键读取 |
| `*args`（`savez`） | 未使用 | 若用匿名参数会自动命名 `arr_0` 等 |
| `file`（`load`） | `"arrays.npz"` | 返回可按键访问的容器对象 |
### 示例代码

```python
import numpy as np

arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2], [3, 4]])
arr3 = np.arange(10)

np.savez("arrays.npz", a=arr1, b=arr2, c=arr3)
data = np.load("arrays.npz")

print(list(data.keys()))
print(data["a"])
print(data["b"])
print(data["c"])
```

### 结果输出

```text
['a', 'b', 'c']
----------------
[1 2 3 4 5]
----------------
[[1 2]
 [3 4]]
----------------
[0 1 2 3 4 5 6 7 8 9]
```

## 3. 文本读写：`savetxt` 与 `loadtxt`

### 参数速览（本节）

适用 API（分项）：

1. `np.savetxt(fname, X, fmt='%.18e', delimiter=' ', header='', comments='# ', ...)`
2. `np.loadtxt(fname, dtype=float, delimiter=None, skiprows=0, usecols=None, ...)`
3. `np.allclose(a, b, rtol=1e-05, atol=1e-08)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `fname` / `X`（`savetxt`） | `"array.txt"` / `arr` | 保存文本数组 |
| `delimiter` / `fmt`（`savetxt`） | `","` / `"%.4f"` | 写 CSV 时指定逗号分隔与保留 4 位小数 |
| `header` / `comments`（`savetxt`） | `"col1,col2,col3,col4"` / `""` | 写入表头并取消默认注释前缀 |
| `fname` / `delimiter`（`loadtxt`） | `"array.txt"` / `None` | 按默认空白分隔读取文本 |
| `skiprows` / `usecols`（`loadtxt`） | `0` / `None` | 不跳行且读取全部列 |
| `a` / `b`（`allclose`） | `arr` / `loaded` | 判断浮点数组近似相等 |
### 示例代码

```python
import numpy as np

np.random.seed(42)
arr = np.random.random((3, 4))

np.savetxt("array.txt", arr)
np.savetxt(
    "array.csv",
    arr,
    delimiter=",",
    fmt="%.4f",
    header="col1,col2,col3,col4",
    comments="",
)

loaded = np.loadtxt("array.txt")
print(np.allclose(arr, loaded))
```

### `array.csv` 示例内容

```text
col1,col2,col3,col4
0.3745,0.9507,0.7320,0.5987
0.1560,0.1560,0.0581,0.8662
0.6011,0.7081,0.0206,0.9699
```

### 理解重点

- 文本格式便于查看和交换，但精度与速度不如二进制。
- `allclose` 常用于验证浮点读写误差。

## 4. `fmt` 格式参数

### 参数速览（本节）

适用参数：`savetxt` 的 `fmt`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `fmt` | `"%.2f"` | 保留 2 位小数（定点小数） |
| `fmt` | `"%.4f"` | 保留 4 位小数（定点小数） |
| `fmt` | `"%d"` | 以整数格式输出（去除小数部分显示） |
| `fmt` | `"%.2e"` | 科学计数法，2 位小数 |
| `fmt` | `"%10.4f"` | 宽度 10，4 位小数（对齐输出列宽） |

脚本演示了常见格式：

- `%.2f`：2 位小数
- `%.4f`：4 位小数
- `%d`：整数
- `%.2e`：科学计数法
- `%10.4f`：宽度 10，4 位小数

### 示例输出（一行）

```text
%.2f  -> 1.23 2.35
----------------
%.4f  -> 1.2346 2.3457
----------------
%d    -> 1 2
----------------
%.2e  -> 1.23e+00 2.35e+00
----------------
%10.4f->     1.2346     2.3457
```

## 5. 带表头文件与 `skiprows`

### 参数速览（本节）

适用 API（分项）：

1. `np.savetxt(..., header='A,B,C', comments='')`
2. `np.savetxt(..., delimiter=',', fmt='%d')`
3. `np.loadtxt(..., delimiter=',', skiprows=1)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `header` / `comments` | `"A,B,C"` / `""` | 写表头并取消默认 `# ` 前缀 |
| `delimiter` / `fmt` | `","` / `"%d"` | 以 CSV 整数格式写入 |
| `delimiter` / `skiprows` | `","` / `1` | 与写入分隔符一致并跳过第一行表头 |
### 示例代码

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
np.savetxt("with_header.csv", arr, delimiter=",", fmt="%d", header="A,B,C", comments="")

loaded = np.loadtxt("with_header.csv", delimiter=",", skiprows=1)
print(loaded)
```

### 结果输出

```text
[[1. 2. 3.]
 [4. 5. 6.]
 [7. 8. 9.]]
```

## 常见坑

1. `loadtxt` 遇到表头会报错，记得 `skiprows=1`。
2. CSV 读写分隔符要一致。
3. 文本保存精度由 `fmt` 决定，过低会丢失信息。

## 小结

- 训练过程建议用 `.npy` / `.npz`，数据交换可用 `.csv`。
- 写入时明确格式、读取时明确解析规则，避免隐式错误。
