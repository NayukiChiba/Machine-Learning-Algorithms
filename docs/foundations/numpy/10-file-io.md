---
title: NumPy 文件 IO
outline: deep
---

# NumPy 文件 IO

## 本章目标

1. 掌握二进制保存/加载：`save`、`load`、`savez`、`savez_compressed`
2. 掌握文本保存/加载：`savetxt`、`loadtxt`
3. 掌握 `fmt` / `delimiter` / `header` / `skiprows` 等格式控制参数
4. 理解 `.npy` / `.npz` / `.csv` 三种格式的选择场景

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `np.save(...)` | 函数 | 保存单个数组到 `.npy` 二进制文件 |
| `np.savez(...)` | 函数 | 保存多个数组到 `.npz` 容器 |
| `np.savez_compressed(...)` | 函数 | 同 `savez`，额外压缩存储 |
| `np.load(...)` | 函数 | 加载 `.npy` / `.npz` 文件 |
| `np.savetxt(...)` | 函数 | 保存数组为可读文本（`.txt` / `.csv`） |
| `np.loadtxt(...)` | 函数 | 从文本文件读取数组 |

## 1. 二进制：`.npy` 单数组

### `np.save`

#### 作用

将单个 NumPy 数组保存为 `.npy` 二进制文件，保真度高、速度快，是 NumPy 内部数据的首选格式。文件自动追加 `.npy` 后缀。

#### 重点方法

```python
np.save(file, arr, allow_pickle=True, fix_imports=True)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `file` | `str` 或 `file-like` | 文件路径；不带 `.npy` 后缀会自动追加 | `"array.npy"` |
| `arr` | `array_like` | 要保存的数组 | —— |
| `allow_pickle` | `bool` | 是否允许保存 object 数组（依赖 pickle，存在安全风险），默认为 `True` | `False` |
| `fix_imports` | `bool` | Python 2/3 兼容选项，默认为 `True` | —— |

### `np.load`

#### 作用

加载 `.npy` 或 `.npz` 文件。`.npy` 返回单个数组；`.npz` 返回可按键访问的容器对象。大文件可配合 `mmap_mode` 内存映射加载。

#### 重点方法

```python
np.load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `file` | `str` 或 `file-like` | 文件路径 | `"array.npy"` |
| `mmap_mode` | `str` 或 `None` | 内存映射模式：`'r'` / `'r+'` / `'c'` / `'w+'`；大文件避免全量加载 | `'r'` |
| `allow_pickle` | `bool` | 是否允许反序列化 object 数组，默认为 `False`（安全） | `True` |
| `encoding` | `str` | object 数组读取时的字符编码，默认为 `'ASCII'` | `'latin1'` |

### 综合示例

#### 示例代码

```python
import numpy as np

np.random.seed(42)
arr = np.random.random((3, 4))

np.save("array.npy", arr)
loaded = np.load("array.npy")

print(f"原数组:\n{arr}")
print(f"加载后:\n{loaded}")
print(f"相等: {np.array_equal(arr, loaded)}")
```

#### 输出

```text
原数组:
[[0.37454012 0.95071431 0.73199394 0.59865848]
 [0.15601864 0.15599452 0.05808361 0.86617615]
 [0.60111501 0.70807258 0.02058449 0.96990985]]
加载后:
[[0.37454012 0.95071431 0.73199394 0.59865848]
 [0.15601864 0.15599452 0.05808361 0.86617615]
 [0.60111501 0.70807258 0.02058449 0.96990985]]
相等: True
```

#### 理解重点

- `.npy` 保存完整的 `dtype` / `shape` / 内存布局，读写往返完全无损
- 大文件用 `mmap_mode='r'` 实现按需读取，不占内存

## 2. 二进制：`.npz` 多数组

### `np.savez`

#### 作用

将多个数组保存到一个 `.npz` 容器中。用关键字参数命名，后续按键读取。

#### 重点方法

```python
np.savez(file, *args, **kwds)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `file` | `str` 或 `file-like` | 文件路径，自动追加 `.npz` 后缀 | `"arrays.npz"` |
| `*args` | `array_like` | 位置参数保存的数组，自动命名为 `arr_0`、`arr_1`... | —— |
| `**kwds` | `array_like` | 关键字参数保存的数组，按给定名称存取 | `a=arr1, b=arr2` |

### `np.savez_compressed`

#### 作用

语义与 `savez` 完全相同，但内部压缩存储，文件体积更小，适合分发或归档。

#### 重点方法

```python
np.savez_compressed(file, *args, **kwds)
```

### 综合示例

#### 示例代码

```python
import numpy as np

arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2], [3, 4]])
arr3 = np.arange(10)

np.savez("arrays.npz", a=arr1, b=arr2, c=arr3)
data = np.load("arrays.npz")

print(f"包含的数组: {list(data.keys())}")
print(f"data['a']: {data['a']}")
print(f"data['b']:\n{data['b']}")
print(f"data['c']: {data['c']}")
```

#### 输出

```text
包含的数组: ['a', 'b', 'c']
data['a']: [1 2 3 4 5]
data['b']:
[[1 2]
 [3 4]]
data['c']: [0 1 2 3 4 5 6 7 8 9]
```

## 3. 文本：`savetxt` / `loadtxt`

### `np.savetxt`

#### 作用

将一维或二维数组保存为可读文本（`.txt` / `.csv`）。支持自定义分隔符、格式化、表头。

#### 重点方法

```python
np.savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n',
           header='', footer='', comments='# ', encoding=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `fname` | `str` 或 `file-like` | 文件路径 | `"array.csv"` |
| `X` | `array_like` | 一维或二维数组 | `arr(3, 4)` |
| `fmt` | `str` | 格式字符串，遵循 C 的 `printf` 语法，默认为 `'%.18e'` | `'%.4f'`、`'%d'` |
| `delimiter` | `str` | 列分隔符，默认为 `' '`（空格） | `','` |
| `newline` | `str` | 行间分隔符，默认为 `'\n'` | —— |
| `header` | `str` | 文件头部附加字符串，默认为 `''` | `"col1,col2,col3"` |
| `footer` | `str` | 文件末尾附加字符串，默认为 `''` | —— |
| `comments` | `str` | `header` / `footer` 前的注释前缀，默认为 `'# '`；写纯 CSV 头时必须设为 `''` | `''` |
| `encoding` | `str` 或 `None` | 写入编码，默认为 `None` | `'utf-8'` |

#### `fmt` 常见写法

| `fmt` | 含义 | 示例：`1.23456` |
|---|---|---|
| `'%.2f'` | 保留 2 位小数 | `1.23` |
| `'%.4f'` | 保留 4 位小数 | `1.2346` |
| `'%d'` | 整数格式 | `1` |
| `'%.2e'` | 科学计数法 2 位小数 | `1.23e+00` |
| `'%10.4f'` | 宽度 10，4 位小数 | <code>&nbsp;&nbsp;&nbsp;1.2346</code> |
| `'%s'` | 字符串格式 | — |

### `np.loadtxt`

#### 作用

从文本文件读取数值数组。支持自定义分隔符、跳行、选取列等。

#### 重点方法

```python
np.loadtxt(fname, dtype=float, comments='#', delimiter=None,
           skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `fname` | `str` 或 `file-like` | 文件路径 | `"array.csv"` |
| `dtype` | `dtype` | 结果数据类型，默认为 `float` | `int` |
| `comments` | `str` | 以此字符开头的行视为注释跳过，默认为 `'#'` | —— |
| `delimiter` | `str` 或 `None` | 列分隔符，默认为 `None`（任意空白字符） | `','` |
| `skiprows` | `int` | 跳过文件头部行数，常用于跳过 CSV 表头，默认为 `0` | `1` |
| `usecols` | `tuple[int, ...]` | 指定读取哪些列（从 0 开始），默认为 `None`（全部） | `(0, 2)` |
| `unpack` | `bool` | `True` 时返回按列解包的结果，便于 `x, y = loadtxt(...)` | `True` |
| `ndmin` | `int` | 结果最小维度，避免单行/单列被降维为一维，默认为 `0` | `2` |
| `encoding` | `str` | 读取编码，默认为 `'bytes'` | `'utf-8'` |

### 综合示例

#### 示例代码

```python
import numpy as np

np.random.seed(42)
arr = np.random.random((3, 4))

# CSV 格式，带表头
np.savetxt(
    "array.csv", arr,
    delimiter=",", fmt="%.4f",
    header="col1,col2,col3,col4",
    comments="",
)

# 加载（跳过表头）
loaded = np.loadtxt("array.csv", delimiter=",", skiprows=1)
print(f"加载一致: {np.allclose(arr, loaded)}")
```

#### `array.csv` 文件内容

```text
col1,col2,col3,col4
0.3745,0.9507,0.7320,0.5987
0.1560,0.1560,0.0581,0.8662
0.6011,0.7081,0.0206,0.9699
```

#### 理解重点

- `comments=''` 不可省略：默认 `'# '` 会让表头变成 `# col1,col2,...`，`loadtxt` 将其识别为注释
- 文本读写有精度损失（由 `fmt` 决定）；需无损保存用 `.npy`
- 验证读写往返用 `np.allclose`，不用 `==`（浮点精度可能微差）

## 4. 格式选择指南

| 场景 | 格式 | API |
|---|---|---|
| 训练/实验中间结果（单数组） | `.npy` | `save` / `load` |
| 多数组（权重、配置、数据） | `.npz` | `savez` / `load` |
| 分发或归档（多数组 + 压缩） | `.npz` | `savez_compressed` / `load` |
| 与 Excel/pandas/其他语言交换 | `.csv` | `savetxt` / `loadtxt` |
| 大文件（超出内存） | `.npy` + mmap | `load(mmap_mode='r')` |

## 常见坑

1. `np.loadtxt` 遇到表头行会报 `ValueError`——加 `skiprows=1` 跳过
2. 写 CSV 时忘记 `comments=''`，表头被加 `#` 前缀变成注释
3. `fmt` 精度过低会丢失有效数字——科学计算默认用 `'%.18e'`
4. `np.load` 的 `allow_pickle=False` 是默认值——加载旧版 object 数组文件时须手动开 `True`（仅在信任来源时）
5. CSV 分隔符可能是 `,` / `;` / `\t`——读写必须前后一致
6. 内存紧张场景用 `mmap_mode='r'` 做内存映射，避免全量加载

## 小结

- 训练过程优先 `.npy`（单数组）或 `.npz`（多数组）——无损且快速
- 跨语言/跨工具交换数据用 `.csv`，明确指定 `delimiter` 和 `fmt`
- 写入端明确格式、读取端明确解析规则——避免隐式错误
- 大文件或内存紧张时用 `mmap_mode='r'` 按需读取
