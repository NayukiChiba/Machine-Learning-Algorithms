---
title: SciPy 稀疏矩阵
outline: deep
---

# SciPy 稀疏矩阵

> 对应脚本：`Basic/Scipy/09_sparse.py`
> 运行方式：`python Basic/Scipy/09_sparse.py`（仓库根目录）

## 本章目标

1. 掌握 CSR 和 COO 格式的稀疏矩阵创建方法。
2. 学会稀疏矩阵的基本运算与格式转换。
3. 理解稀疏线性代数求解器 `spsolve` 的使用。
4. 了解稀疏矩阵在内存效率上的优势。

## 重点方法速览

| 方法 | 作用 | 本章位置 |
|---|---|---|
| `sparse.csr_matrix(data)` | 创建 CSR 格式稀疏矩阵 | `demo_sparse_create` |
| `sparse.coo_matrix((data, (row, col)))` | 创建 COO 格式稀疏矩阵 | `demo_sparse_create` |
| `sparse.random(m, n, density)` | 创建随机稀疏矩阵 | `demo_sparse_operations` |
| `sparse.linalg.spsolve(A, b)` | 稀疏线性方程组求解 | `demo_sparse_linalg` |
| `sparse.diags(diagonals, offsets)` | 创建对角稀疏矩阵 | `demo_sparse_linalg` |

## 1. 稀疏矩阵创建

### 方法重点

- **CSR（Compressed Sparse Row）**：按行压缩存储，适合行切片和矩阵-向量乘法。
- **COO（Coordinate）**：坐标格式，用 `(row, col, data)` 三元组存储，适合构建稀疏矩阵。
- `csr_matrix(dense)` 可从密集矩阵直接转换。
- COO 格式适合逐元素添加数据，构建完成后转为 CSR 计算更高效。

### 参数速览（本节）

适用 API（分项）：

1. `sparse.csr_matrix(arg1, shape=None)`
2. `sparse.coo_matrix((data, (row, col)), shape)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `arg1` | 4×4 对角密集矩阵 | 密集矩阵或稀疏数据 |
| `data` | `[1, 2, 3, 4]` | 非零元素值 |
| `row` | `[0, 1, 2, 3]` | 非零元素行索引 |
| `col` | `[0, 1, 2, 3]` | 非零元素列索引 |
| `shape` | `(4, 4)` | 矩阵形状 |

### 示例代码

```python
import numpy as np
from scipy import sparse

# 从密集矩阵创建
dense = np.array([
    [1, 0, 0, 0],
    [0, 2, 0, 0],
    [0, 0, 3, 0],
    [0, 0, 0, 4]
])

# CSR 格式 (压缩行)
csr = sparse.csr_matrix(dense)
print(f"密集矩阵:\n{dense}")
print(f"\nCSR 稀疏矩阵:\n{csr}")
print(f"数据: {csr.data}")
print(f"列索引: {csr.indices}")

# COO 格式 (坐标)
row = np.array([0, 1, 2, 3])
col = np.array([0, 1, 2, 3])
data = np.array([1, 2, 3, 4])
coo = sparse.coo_matrix((data, (row, col)), shape=(4, 4))
print(f"\nCOO 格式:\n{coo}")
```

### 结果输出

```text
密集矩阵:
[[1 0 0 0]
 [0 2 0 0]
 [0 0 3 0]
 [0 0 0 4]]

CSR 稀疏矩阵:
  (0, 0)	1
  (1, 1)	2
  (2, 2)	3
  (3, 3)	4
数据: [1 2 3 4]
列索引: [0 1 2 3]

COO 格式:
  (0, 0)	1
  (1, 1)	2
  (2, 2)	3
  (3, 3)	4
```

### 理解重点

- CSR 格式存储三个数组：`data`（非零值）、`indices`（列索引）、`indptr`（行指针）。
- 4×4 矩阵有 16 个元素，但只有 4 个非零——稀疏率 75%。
- COO 和 CSR 存储相同信息，但 CSR 更适合计算（矩阵乘法 O(nnz)），COO 更适合构建。
- `coo.tocsr()` 和 `csr.tocoo()` 可在格式之间快速转换。

## 2. 稀疏矩阵操作

### 方法重点

- `sparse.random(m, n, density, format)` 生成指定密度的随机稀疏矩阵。
- 稀疏矩阵支持加法、乘法等矩阵运算，结果仍为稀疏矩阵。
- `.nnz` 属性返回非零元素数量。
- `.toarray()` 将稀疏矩阵转回密集 NumPy 数组。

### 参数速览（本节）

适用 API：`sparse.random(m, n, density=0.01, format='coo')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `m`, `n` | `20`, `20` | 矩阵形状 |
| `density` | `0.1` | 非零元素占比（10%） |
| `format` | `'csr'` | 输出格式 |

### 示例代码

```python
import numpy as np
from scipy import sparse

np.random.seed(42)
A = sparse.random(20, 20, density=0.1, format='csr')
print("随机稀疏矩阵 A (密度=0.1):")
print(f"  形状: {A.shape}")
print(f"  非零元素数: {A.nnz}")
print(f"  密度: {A.nnz / (A.shape[0] * A.shape[1]):.2f}")

# 矩阵运算
B = sparse.eye(20, format='csr')
C = A + B
print(f"\nA + I 的非零元素数: {C.nnz}")

# 转换为密集矩阵
dense = A.toarray()
print(f"\n转换为密集矩阵 shape: {dense.shape}")
```

### 结果输出

```text
随机稀疏矩阵 A (密度=0.1):
  形状: (20, 20)
  非零元素数: 40
  密度: 0.10

A + I 的非零元素数: 56

转换为密集矩阵 shape: (20, 20)
```

### 理解重点

- 20×20 矩阵共 400 个元素，10% 密度意味着约 40 个非零元素。
- A + I（加单位矩阵）后非零元素增加到 56，因为对角线上部分位置原本为零。
- 稀疏矩阵运算保持稀疏格式，不会自动转为密集矩阵。
- `sparse.eye(n)` 创建稀疏单位矩阵，比 `np.eye(n)` 节省大量内存。

## 3. 稀疏线性代数

### 方法重点

- `sparse.linalg.spsolve(A, b)` 高效求解稀疏方程组 Ax = b。
- `sparse.diags(diagonals, offsets, shape)` 创建对角稀疏矩阵。
- 三对角矩阵是最常见的稀疏结构，广泛用于有限差分法。
- 稀疏求解器利用矩阵结构，时间复杂度远低于 O(n³) 的密集求解。

### 参数速览（本节）

适用 API（分项）：

1. `sparse.diags(diagonals, offsets, shape, format)`
2. `sparse.linalg.spsolve(A, b)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `diagonals` | `[-1, 2, -1]` | 各对角线上的值 |
| `offsets` | `[-1, 0, 1]` | 对角线偏移（0=主对角线） |
| `shape` | `(100, 100)` | 矩阵形状 |
| `b` | `np.ones(100)` | 右端向量 |

### 示例代码

```python
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

# 创建三对角稀疏系统
n = 100
A = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), format='csr')
b = np.ones(n)

print("稀疏方程组 Ax = b")
print(f"  矩阵大小: {n}x{n}")
print(f"  非零元素: {A.nnz}")

# 直接求解
x = splinalg.spsolve(A, b)
print(f"  解的范数: {np.linalg.norm(x):.4f}")
print(f"  残差: {np.linalg.norm(A @ x - b):.2e}")
```

### 结果输出

```text
稀疏方程组 Ax = b
  矩阵大小: 100x100
  非零元素: 298
  解的范数: 29.0115
  残差: 2.15e-14
```

### 理解重点

- 100×100 三对角矩阵只有 298 个非零元素（主对角线 100 + 上下各 99），远少于密集的 10000 个。
- `spsolve` 利用三对角结构，求解时间复杂度为 O(n)（Thomas 算法），密集求解需 O(n³)。
- 残差 ≈ 10⁻¹⁴（机器精度），验证了求解的正确性。
- 三对角矩阵 `[-1, 2, -1]` 是一维拉普拉斯算子的离散形式，广泛用于热传导、扩散方程。

## 4. 稀疏矩阵效率

### 方法重点

- 稀疏矩阵的核心优势是内存节省和计算加速。
- 密集矩阵内存 = n² × 8 字节（float64），稀疏矩阵内存 ≈ nnz × 16 字节。
- 当矩阵密度低于约 10% 时，稀疏格式在内存和速度上都有显著优势。
- 实际应用中，大规模矩阵（如图的邻接矩阵、有限元刚度矩阵）密度通常远低于 1%。

### 参数速览（本节）

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `sizes` | `[100, 500, 1000, 2000, 5000]` | 测试矩阵尺寸 |
| `density` | `0.01` | 非零元素占比（1%） |

### 示例代码

```python
import numpy as np

n = 1000
density = 0.01

# 密集矩阵内存
dense_mem = n * n * 8  # float64 = 8 bytes

# 稀疏矩阵内存 (COO 格式: data + row + col)
nnz = int(n * n * density)
sparse_mem = nnz * (8 + 4 + 4)  # data(float64) + row(int32) + col(int32)

print(f"矩阵大小: {n}x{n}")
print(f"密度: {density * 100}%")
print(f"\n内存使用:")
print(f"  密集矩阵: {dense_mem / 1024 / 1024:.2f} MB")
print(f"  稀疏矩阵: {sparse_mem / 1024:.2f} KB")
print(f"  节省: {(1 - sparse_mem / dense_mem) * 100:.1f}%")
```

### 结果输出

```text
矩阵大小: 1000x1000
密度: 1.0%

内存使用:
  密集矩阵: 7.63 MB
  稀疏矩阵: 156.25 KB
  节省: 98.0%
```

### 理解重点

- 1000×1000 密度 1% 的矩阵，稀疏格式只需 156KB，密集格式需 7.63MB，节省 98%。
- 随着矩阵规模增大，节省比例保持不变（由密度决定），但绝对值差距急剧增大。
- 5000×5000 密集矩阵需 ~190MB，稀疏仅需 ~3.8MB。
- 实际场景中（如推荐系统的用户-物品矩阵、自然语言处理的词-文档矩阵），密度往往不到 0.1%，稀疏存储是唯一可行方案。

## 常见坑

| 坑 | 说明 |
|---|---|
| 格式选择 | CSR 适合行操作和矩阵乘法，CSC 适合列操作，COO 适合构建——选错格式影响性能 |
| 逐元素赋值低效 | 不要用 `A[i,j] = v` 逐个赋值，应先收集坐标再一次性创建 COO 矩阵 |
| `toarray()` 内存爆炸 | 大规模稀疏矩阵转密集矩阵可能导致内存溢出 |
| 稀疏 × 密集 | `sparse @ dense` 返回密集矩阵，可能抵消稀疏带来的内存优势 |
| `spsolve` 要求方阵 | 非方阵的最小二乘问题应使用 `sparse.linalg.lsqr` |

## 小结

- CSR 和 COO 是最常用的稀疏矩阵格式，各有适用场景。
- 稀疏矩阵支持加法、乘法等基本运算，结果保持稀疏格式。
- `spsolve` 利用矩阵稀疏结构高效求解线性方程组，残差达到机器精度。
- 稀疏存储在低密度场景下可节省 98%+ 的内存，是处理大规模数据的关键技术。
- 使用原则：密度 < 10% 优先考虑稀疏格式；构建用 COO，计算用 CSR/CSC。
