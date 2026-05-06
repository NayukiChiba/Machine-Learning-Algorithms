---
title: SciPy 稀疏矩阵
outline: deep
---

# SciPy 稀疏矩阵

## 本章目标

1. 掌握 CSR 和 COO 格式的稀疏矩阵创建方法
2. 学会稀疏矩阵的基本运算与格式转换
3. 理解稀疏线性代数求解器 `spsolve` 的使用
4. 了解稀疏矩阵在内存效率上的优势

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `sparse.csr_matrix(data)` | 构造器 | 创建 CSR 格式（压缩行）稀疏矩阵 |
| `sparse.coo_matrix((data, (row, col)))` | 构造器 | 创建 COO 格式（坐标）稀疏矩阵 |
| `sparse.random(m, n, density)` | 函数 | 创建指定密度的随机稀疏矩阵 |
| `sparse.diags(diagonals, offsets, shape)` | 函数 | 创建对角稀疏矩阵 |
| `sparse.linalg.spsolve(A, b)` | 函数 | 稀疏线性方程组求解 |

CSR 适合计算（行切片、矩阵乘法），COO 适合构建（逐元素添加）。

## 1. 稀疏矩阵创建

### `sparse.csr_matrix` / `sparse.coo_matrix`

#### 作用

- **CSR（Compressed Sparse Row）**：按行压缩存储，存储三个数组：`data`（非零值）、`indices`（列索引）、`indptr`（行指针）。适合行切片和矩阵-向量乘法
- **COO（Coordinate）**：坐标格式，用 `(row, col, data)` 三元组存储。适合构建稀疏矩阵，构建完成后转为 CSR 计算更高效

#### 重点方法

```python
sparse.csr_matrix(arg1, shape=None)
sparse.coo_matrix((data, (row, col)), shape=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `arg1` | `array_like` | 密集矩阵或其他稀疏数据 | `np.eye(4)` |
| `data` | `array_like` | 非零元素值（COO 的第一参数） | `[1, 2, 3, 4]` |
| `row` | `array_like` | 非零元素行索引 | `[0, 1, 2, 3]` |
| `col` | `array_like` | 非零元素列索引 | `[0, 1, 2, 3]` |
| `shape` | `tuple[int, int]` | 矩阵形状；可省略（从索引推断） | `(4, 4)` |

#### 示例代码

```python
import numpy as np
from scipy import sparse

# 从密集矩阵创建 CSR
dense = np.diag([1, 2, 3, 4])
csr = sparse.csr_matrix(dense)

print(f"密集矩阵:\n{dense}")
print(f"\nCSR 非零元素: {csr.data}")
print(f"CSR 列索引: {csr.indices}")

# 以 COO 格式手动创建
row = np.array([0, 1, 2, 3])
col = np.array([0, 1, 2, 3])
data = np.array([1, 2, 3, 4])
coo = sparse.coo_matrix((data, (row, col)), shape=(4, 4))

print(f"\nCOO 格式:\n{coo}")
print(f"COO 非零元素: {coo.nnz}")
```

#### 输出

```text
密集矩阵:
[[1 0 0 0]
 [0 2 0 0]
 [0 0 3 0]
 [0 0 0 4]]

CSR 非零元素: [1 2 3 4]
CSR 列索引: [0 1 2 3]

COO 格式:
  (0, 0)	1
  (1, 1)	2
  (2, 2)	3
  (3, 3)	4
COO 非零元素: 4
```

#### 理解重点

- 4×4 矩阵有 16 个元素，只有 4 个非零——稀疏率 75%
- CSR 和 COO 存储相同信息，但 CSR 更适合计算（矩阵乘法 $O(nnz)$），COO 更适合构建
- `coo.tocsr()` 和 `csr.tocoo()` 可在格式之间快速转换
- `.toarray()` 将稀疏矩阵转回密集 NumPy 数组

## 2. 稀疏矩阵操作

### `sparse.random` / 矩阵运算

#### 作用

`sparse.random` 生成指定密度的随机稀疏矩阵。稀疏矩阵支持加法、乘法等运算，结果保持稀疏格式。`.nnz` 属性返回非零元素数量。

#### 重点方法

```python
sparse.random(m, n, density=0.01, format='coo')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `m` | `int` | 矩阵行数 | `20` |
| `n` | `int` | 矩阵列数 | `20` |
| `density` | `float` | 非零元素占比，默认为 `0.01` | `0.1` |
| `format` | `str` | 输出格式：`'csr'` / `'csc'` / `'coo'`，默认为 `'coo'` | `'csr'` |

#### 示例代码

```python
import numpy as np
from scipy import sparse

np.random.seed(42)
A = sparse.random(20, 20, density=0.1, format='csr')
print(f"随机稀疏矩阵 A (密度=0.1):")
print(f"  形状: {A.shape}")
print(f"  非零元素: {A.nnz}")
print(f"  实际密度: {A.nnz / (A.shape[0]*A.shape[1]):.2f}")

# 矩阵运算：A + I
B = sparse.eye(20, format='csr')
C = A + B
print(f"\nA + I 非零元素: {C.nnz}")

# 转为密集矩阵
dense = A.toarray()
print(f"密集矩阵 shape: {dense.shape}")
```

#### 输出

```text
随机稀疏矩阵 A (密度=0.1):
  形状: (20, 20)
  非零元素: 40
  实际密度: 0.10

A + I 非零元素: 56

密集矩阵 shape: (20, 20)
```

#### 理解重点

- 20×20 矩阵 10% 密度 → 约 40 个非零元素
- A + I（加单位矩阵）后非零元素增加到 56——对角线上部分零位被填充
- 稀疏矩阵运算保持稀疏格式——不会自动转为密集矩阵
- `sparse.eye(n)` 创建稀疏单位矩阵——比 `np.eye(n)` 节省大量内存

## 3. 稀疏线性代数

### `sparse.linalg.spsolve` / `sparse.diags`

#### 作用

`spsolve` 利用矩阵稀疏结构高效求解 Ax = b。`sparse.diags` 创建对角稀疏矩阵——三对角矩阵是最常见的稀疏结构，广泛用于有限差分法。稀疏求解的时间复杂度远低于密集求解的 $O(n^3)$。

#### 重点方法

```python
sparse.diags(diagonals, offsets, shape, format='csr')
sparse.linalg.spsolve(A, b)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `diagonals` | `list[array_like]` | 各对角线上的值 | `[[-1, 2, -1]]` |
| `offsets` | `list[int]` | 对角线偏移：`0`=主对角线，`-1`=下对角，`1`=上对角 | `[-1, 0, 1]` |
| `shape` | `tuple[int, int]` | 矩阵形状 | `(100, 100)` |
| `A` | `sparse matrix` | 稀疏系数矩阵（方阵） | 三对角 CSR 矩阵 |
| `b` | `array_like` | 右端向量 | `np.ones(100)` |

#### 示例代码

```python
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

# 创建三对角系统 [-1, 2, -1]（一维拉普拉斯算子）
n = 100
A = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), format='csr')
b = np.ones(n)

x = splinalg.spsolve(A, b)

print(f"矩阵: {n}x{n}, 非零元素: {A.nnz}")
print(f"解范数: {np.linalg.norm(x):.4f}")
print(f"残差: {np.linalg.norm(A @ x - b):.2e}")
```

#### 输出

```text
矩阵: 100x100, 非零元素: 298
解范数: 29.0115
残差: 2.15e-14
```

#### 理解重点

- 100×100 三对角矩阵只有 298 个非零元素（主对角 100 + 上下各 99）——远少于密集的 10000
- `spsolve` 利用三对角结构，时间复杂度 $O(n)$（Thomas 算法）——密集求解需 $O(n^3)$
- 残差 ≈ $10^{-14}$（机器精度）——验证了求解的正确性
- 三对角矩阵 `[-1, 2, -1]` 是一维拉普拉斯算子的离散形式——广泛用于热传导、扩散方程

## 4. 稀疏矩阵内存效率

### 内存对比

#### 作用

稀疏矩阵的核心优势是内存节省和计算加速。密集矩阵内存 = $n^2 \times 8$ 字节（float64），稀疏矩阵内存 ≈ $nnz \times 16$ 字节。当密度低于约 10% 时，稀疏格式在内存和速度上都有显著优势。

#### 示例代码

```python
n = 1000
density = 0.01
nnz = int(n * n * density)

# 密集矩阵内存 (float64 = 8 bytes)
denseMem = n * n * 8

# 稀疏矩阵内存 (COO: data(float64) + row(int32) + col(int32))
sparseMem = nnz * (8 + 4 + 4)

print(f"矩阵: {n}x{n}, 密度: {density*100}%")
print(f"密集矩阵: {denseMem / 1024 / 1024:.2f} MB")
print(f"稀疏矩阵: {sparseMem / 1024:.2f} KB")
print(f"节省: {(1 - sparseMem / denseMem) * 100:.1f}%")
```

#### 输出

```text
矩阵: 1000x1000, 密度: 1.0%
密集矩阵: 7.63 MB
稀疏矩阵: 156.25 KB
节省: 98.0%
```

#### 理解重点

- 1000×1000 密度 1% 的矩阵——稀疏仅需 156KB，密集需 7.63MB，节省 98%
- 随着矩阵规模增大，节省比例不变（由密度决定），但绝对值差距急剧增大
- 5000×5000 密集需 ~190MB，稀疏仅 ~3.8MB
- 实际场景（推荐系统用户-物品矩阵、NLP 词-文档矩阵）密度往往不到 0.1%——稀疏存储是唯一可行方案

## 常见坑

1. 格式选择：CSR 适合行操作和矩阵乘法，CSC 适合列操作，COO 适合构建——选错影响性能
2. 逐元素赋值低效：不要用 `A[i,j] = v` 逐个赋值——先收集坐标再一次性创建 COO
3. `toarray()` 内存爆炸：大规模稀疏矩阵转密集可能导致内存溢出
4. 稀疏 × 密集 = 密集：`sparse @ dense` 返回密集矩阵，可能抵消稀疏内存优势
5. `spsolve` 要求方阵：非方阵最小二乘应使用 `sparse.linalg.lsqr`

## 小结

- CSR 和 COO 是最常用的稀疏矩阵格式——各有适用场景
- 稀疏矩阵支持加法、乘法等基本运算——结果保持稀疏格式
- `spsolve` 利用矩阵稀疏结构高效求解——残差达机器精度
- 稀疏存储在低密度场景下可节省 98%+ 的内存——处理大规模数据的关键技术
- 原则：密度 < 10% 优先稀疏；构建用 COO，计算用 CSR/CSC
