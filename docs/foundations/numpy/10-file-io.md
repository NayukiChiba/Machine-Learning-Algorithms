---
title: NumPy 文件 IO
outline: deep
---

# NumPy 文件 IO

## 本章目标

1. 掌握二进制保存 / 加载：`save`、`load`、`savez`。
2. 掌握文本保存 / 加载：`savetxt`、`loadtxt`。
3. 掌握 `fmt` / `delimiter` / `header` / `skiprows` 等格式控制参数。
4. 理解 `.npy` / `.npz` / `.csv` 三种格式的选择场景。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `np.save(...)` | 函数 | 保存**单个**数组到 `.npy` 二进制文件 |
| `np.savez(...)` | 函数 | 保存**多个**数组到 `.npz` 容器 |
| `np.savez_compressed(...)` | 函数 | 同 `savez`，额外压缩存储 |
| `np.load(...)` | 函数 | 加载 `.npy` / `.npz` |
| `np.savetxt(...)` | 函数 | 保存数组为可读文本（`.txt` / `.csv`） |
| `np.loadtxt(...)` | 函数 | 从文本读取数组 |

## 二进制：`.npy` 单数组

### `np.save`

#### 作用

将单个 NumPy 数组保存为 `.npy` 二进制文件。保真度高、速度快，是 NumPy 内部数据的首选格式。

#### 重点方法

```python
np.save(file, arr, allow_pickle=True, fix_imports=True)
```

#### 参数

| 参数名         | 本例取值          | 说明                                                           |
| -------------- | ----------------- | -------------------------------------------------------------- |
| `file`         | `"array.npy"`     | 文件路径或类文件对象；不带 `.npy` 后缀会自动追加               |
| `arr`          | 任意 `ndarray`    | 要保存的数组                                                   |
| `allow_pickle` | `True`（默认）    | 是否允许保存 object 数组（依赖 pickle，存在反序列化安全风险）  |
| `fix_imports`  | `True`（默认）    | Python 2/3 兼容选项，现代代码可保持默认                        |

### `np.load`

#### 作用

加载 `.npy` 或 `.npz` 文件。`.npy` 返回单个数组；`.npz` 返回可按键访问的容器对象。

#### 重点方法

```python
np.load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
```

#### 参数

| 参数名         | 本例取值         | 说明                                                                 |
| -------------- | ---------------- | -------------------------------------------------------------------- |
| `file`         | `"array.npy"`    | 文件路径或类文件对象                                                 |
| `mmap_mode`    | `None`（默认）   | 内存映射模式：`'r'` / `'r+'` / `'c'` / `'w+'`；大文件避免全量加载    |
| `allow_pickle` | `False`（默认）  | 是否允许反序列化 object 数组；默认禁用以提高安全性                   |
| `encoding`     | `'ASCII'`（默认）| object 数组读取时的字符编码                                          |

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

- `.npy` 保存完整的 `dtype` / `shape` / 内存布局，读写往返**完全无损**。
- 大文件配合 `mmap_mode='r'` 可按需读取不占内存。

## 二进制：`.npz` 多数组

### `np.savez`

#### 作用

将多个数组保存到一个 `.npz` 容器中。使用**关键字参数**命名，后续按键读取。

#### 重点方法

```python
np.savez(file, *args, **kwds)
```

#### 参数

| 参数名  | 本例取值                     | 说明                                                                  |
| ------- | ---------------------------- | --------------------------------------------------------------------- |
| `file`  | `"arrays.npz"`               | 文件路径或类文件对象；自动追加 `.npz`                                 |
| `*args` | —                            | 位置参数保存的数组，自动命名为 `arr_0`、`arr_1`...                    |
| `**kwds`| `a=arr1, b=arr2, c=arr3`     | 关键字参数保存的数组，按给定名称存取                                  |

### `np.savez_compressed`

#### 作用

语义与 `savez` 相同，但对内容**压缩存储**。牺牲一些速度换取更小文件体积，适合分发或归档。

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

## 文本：`savetxt` / `loadtxt`

### `np.savetxt`

#### 作用

将一维或二维数组保存为可读文本（如 `.txt` / `.csv`）。支持自定义分隔符、格式化、表头。

#### 重点方法

```python
np.savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n',
           header='', footer='', comments='# ', encoding=None)
```

#### 参数

| 参数名      | 本例取值                | 说明                                                                |
| ----------- | ----------------------- | ------------------------------------------------------------------- |
| `fname`     | `"array.csv"`           | 文件路径                                                            |
| `X`         | `arr(3, 4)`             | 一维或二维数组                                                      |
| `fmt`       | `'%.18e'`（默认）、`'%.4f'`、`'%d'` | 格式字符串，遵循 C 的 `printf` 语法（见下表）             |
| `delimiter` | `' '`（默认）、`','`    | 列之间的分隔符                                                      |
| `newline`   | `'\n'`（默认）          | 行间分隔符                                                          |
| `header`    | `''`（默认）、`"A,B,C"` | 文件头部附加的字符串                                                |
| `footer`    | `''`（默认）            | 文件末尾附加的字符串                                                |
| `comments`  | `'# '`（默认）、`''`    | `header` / `footer` 前自动添加的注释前缀；写纯 CSV 时设为 `''`      |
| `encoding`  | `None`（默认）          | 写入文件使用的编码                                                  |

### `np.loadtxt`

#### 作用

从文本读取数值数组。支持自定义分隔符、跳行、选取列等。

#### 重点方法

```python
np.loadtxt(fname, dtype=float, comments='#', delimiter=None, converters=None,
           skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes')
```

#### 参数

| 参数名       | 本例取值           | 说明                                                       |
| ------------ | ------------------ | ---------------------------------------------------------- |
| `fname`      | `"array.csv"`      | 文件路径                                                   |
| `dtype`      | `float`（默认）    | 结果数据类型                                               |
| `comments`   | `'#'`（默认）      | 以此字符开头的行视为注释跳过                               |
| `delimiter`  | `None`（默认）、`','`| 列分隔符，`None` 表示任意空白字符                        |
| `converters` | `None`（默认）     | 字典，指定按列自定义转换函数                               |
| `skiprows`   | `0`（默认）、`1`   | 跳过文件头部多少行（常用于跳过表头）                       |
| `usecols`    | `None`（默认）、`(0, 2)` | 指定读取哪些列                                         |
| `unpack`     | `False`（默认）    | `True` 时返回按列解包的结果，便于 `x, y = loadtxt(...)`    |
| `ndmin`      | `0`（默认）        | 结果最小维度，避免单行 / 单列被降维为一维                  |

### `fmt` 常见写法

| `fmt` 写法 | 含义                 | 示例输入 → 输出        |
| ---------- | -------------------- | ---------------------- |
| `'%.2f'`   | 保留 2 位小数        | `1.23456` → `1.23`     |
| `'%.4f'`   | 保留 4 位小数        | `1.23456` → `1.2346`   |
| `'%d'`     | 整数格式             | `1.23` → `1`           |
| `'%.2e'`   | 科学计数法 2 位小数  | `123.45` → `1.23e+02`  |
| `'%10.4f'` | 宽度 10，4 位小数    | `1.23` → `    1.2300`  |
| `'%s'`     | 字符串格式           | —                      |

### 综合示例

#### 示例代码

```python
import numpy as np

np.random.seed(42)
arr = np.random.random((3, 4))

# 默认格式
np.savetxt("array.txt", arr)

# CSV 格式，带表头
np.savetxt(
    "array.csv",
    arr,
    delimiter=",",
    fmt="%.4f",
    header="col1,col2,col3,col4",
    comments="",
)

# 加载
loaded = np.loadtxt("array.txt")
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

- `comments=''` 不可少；默认 `'# '` 会让表头变成 `# col1,col2,col3,col4`，`loadtxt` 将其识别为注释行。
- 文本读写有**精度损失**（`fmt` 决定）；需要无损保留用 `.npy`。
- `np.allclose` 用于浮点近似比较，是读写往返验证的标准做法。

## 带表头 CSV 的完整流程

### 示例代码

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 写：CSV + 表头
np.savetxt(
    "with_header.csv",
    arr,
    delimiter=",",
    fmt="%d",
    header="A,B,C",
    comments="",
)

# 读：跳过表头
loaded = np.loadtxt("with_header.csv", delimiter=",", skiprows=1)
print(f"加载结果:\n{loaded}")
```

### 输出

```text
加载结果:
[[1. 2. 3.]
 [4. 5. 6.]
 [7. 8. 9.]]
```

### 理解重点

- 写入 `header` 用的**分隔符**和 `delimiter` 要一致，否则 `loadtxt` 解析会错位。
- `loadtxt` 默认以 `float` 读取；需要整数用 `dtype=int`。

## 常见坑

1. `np.loadtxt` 遇到表头行会报 `ValueError`，加 `skiprows=1` 跳过。
2. 写 CSV 时忘了 `comments=''`，`header` 会被注释前缀污染。
3. `fmt` 精度过低会丢失信息，科学计算默认用 `'%.18e'`。
4. `np.load` 的 `allow_pickle=False` 是默认值，加载含 object 数组的旧文件会报错；**仅在信任来源时**手动开启。
5. `.npz` 用 `np.load` 得到的是懒加载对象，别忘了 `.close()` 或用 `with` 上下文管理；或直接 `dict(np.load(...))`。
6. CSV 分隔符可能是 `,` / `;` / `\t`；读写要**前后一致**。

## 小结

- 训练 / 实验过程优先用 `.npy`（单数组）或 `.npz`（多数组），无损又快速。
- 需要与 Excel / pandas / 其他语言交换数据时用 `.csv`，明确 `delimiter` 和 `fmt`。
- 大文件或内存紧张场景用 `mmap_mode='r'` 做内存映射加载。
- 写入端明确格式、读取端明确解析规则，避免隐式错误。
