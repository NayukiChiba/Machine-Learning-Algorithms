---
title: SciPy 线性代数
outline: deep
---

# SciPy 线性代数

## 本章目标

1. 掌握 LU 分解及其在矩阵运算中的作用
2. 学会使用 QR 分解处理非方阵
3. 理解 SVD 奇异值分解的原理与矩阵重构
4. 掌握特征值分解及其几何意义
5. 学会使用 `linalg.solve` 求解线性方程组

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `linalg.lu(A)` | 函数 | LU 分解（A = PLU） |
| `linalg.qr(A)` | 函数 | QR 分解（A = QR） |
| `linalg.svd(A)` | 函数 | 奇异值分解（A = UΣVᴴ） |
| `linalg.eig(A)` | 函数 | 特征值与特征向量 |
| `linalg.solve(A, b)` | 函数 | 线性方程组求解（Ax = b） |

`scipy.linalg` 比 `numpy.linalg` 功能更全面，部分函数效率更高。

## 1. LU 分解

### `linalg.lu`

#### 作用

将矩阵分解为 A = PLU（置换矩阵 × 下三角 × 上三角）。P 用于行交换以保证数值稳定性（选主元策略）。LU 分解是高斯消元法的矩阵形式，分解一次后可高效求解多个右端向量的方程组。

#### 重点方法

```python
linalg.lu(A)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `A` | `array_like` | 待分解的方阵 | `[[1,2,3],[4,5,6],[7,8,10]]` |

#### 示例代码

```python
import numpy as np
from scipy import linalg

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
P, L, U = linalg.lu(A)

print(f"原矩阵 A:\n{A}")
print(f"\nL (下三角):\n{np.round(L, 4)}")
print(f"\nU (上三角):\n{np.round(U, 4)}")
print(f"\n验证 P@L@U:\n{np.round(P @ L @ U, 4)}")
```

#### 输出

```text
原矩阵 A:
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8 10]]

L (下三角):
[[1.     0.     0.    ]
 [0.5714 1.     0.    ]
 [0.1429 0.5    1.    ]]

U (上三角):
[[7.     8.     10.    ]
 [0.     0.4286 0.2857]
 [0.     0.     0.5   ]]

验证 P@L@U:
[[ 1.  2.  3.]
 [ 4.  5.  6.]
 [ 7.  8. 10.]]
```

#### 理解重点

- P 将第 3 行移到第 1 行（选主元），保证 L 的元素绝对值 ≤ 1
- L 的对角线全为 1（单位下三角），U 的对角线是主元
- P@L@U 精确重构原矩阵 A——验证分解正确性
- LU 分解一次后可高效求解多个右端向量的方程组

## 2. QR 分解

### `linalg.qr`

#### 作用

将矩阵分解为 A = QR（正交矩阵 × 上三角矩阵）。Q 的列向量两两正交（$Q^T Q = I$），R 是上三角矩阵。适用于非方阵，常用于最小二乘问题和特征值算法。

#### 重点方法

```python
linalg.qr(A)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `A` | `array_like` | 待分解的矩阵（可为非方阵） | `[[1,2],[3,4],[5,6]]` |

#### 示例代码

```python
import numpy as np
from scipy import linalg

A = np.array([[1, 2], [3, 4], [5, 6]])  # 3×2

Q, R = linalg.qr(A)

print(f"原矩阵 A (3x2):\n{A}")
print(f"\nR (上三角):\n{np.round(R, 4)}")
print(f"\n验证 Q@R:\n{np.round(Q @ R, 4)}")
print(f"Q^T @ Q 是否≈I: {np.allclose(Q.T @ Q, np.eye(3))}")
```

#### 输出

```text
原矩阵 A (3x2):
[[1 2]
 [3 4]
 [5 6]]

R (上三角):
[[-5.9161 -7.4370]
 [ 0.0000  0.8281]
 [ 0.0000  0.0000]]

验证 Q@R:
[[1. 2.]
 [3. 4.]
 [5. 6.]]
Q^T @ Q 是否≈I: True
```

#### 理解重点

- A 是 3×2 矩阵，Q 是 3×3 正交矩阵，R 是 3×2 上三角矩阵
- R 的前 2 行是上三角，第 3 行全零（因为 A 只有 2 列）
- Q 的列向量两两正交且模为 1
- QR 分解是 Gram-Schmidt 正交化过程的矩阵形式

## 3. SVD 分解

### `linalg.svd`

#### 作用

奇异值分解将矩阵分解为 $A = U\Sigma V^H$。U 和 V 是正交矩阵，$\Sigma$ 的对角元素为奇异值（非负且递减）。SVD 适用于任意形状的矩阵——是最通用的矩阵分解方法。奇异值反映矩阵的"信息量"，用于降维、压缩和伪逆计算。

#### 重点方法

```python
linalg.svd(A)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `A` | `array_like` | 待分解的矩阵（任意形状） | `[[1,2,3],[4,5,6]]` |

#### 示例代码

```python
import numpy as np
from scipy import linalg

A = np.array([[1, 2, 3], [4, 5, 6]])  # 2×3

U, s, Vh = linalg.svd(A)

print(f"原矩阵 A (2x3):\n{A}")
print(f"\n奇异值 s: {np.round(s, 4)}")

# 重构
S = np.zeros_like(A, dtype=float)
S[:len(s), :len(s)] = np.diag(s)
reconstructed = U @ S @ Vh
print(f"\n重构 U@S@Vh:\n{np.round(reconstructed, 4)}")
```

#### 输出

```text
原矩阵 A (2x3):
[[1 2 3]
 [4 5 6]]

奇异值 s: [9.5080 0.7729]

重构 U@S@Vh:
[[1. 2. 3.]
 [4. 5. 6.]]
```

#### 理解重点

- `linalg.svd` 返回的 `s` 是一维数组（奇异值），需手动构造对角矩阵 $\Sigma$
- 奇异值 9.51 >> 0.77——矩阵的主要信息集中在第一个奇异值方向
- 保留前 k 个最大奇异值可实现矩阵的低秩近似（数据压缩）
- U@S@Vh 精确重构原矩阵——验证分解正确性

## 4. 特征值与特征向量

### `linalg.eig`

#### 作用

计算方阵的特征值和特征向量。特征方程 $Av = \lambda v$：矩阵乘以特征向量等于特征值缩放特征向量。特征值可能是复数（即使矩阵元素全为实数）。特征向量按列排列，第 i 列对应第 i 个特征值。

#### 重点方法

```python
linalg.eig(A)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `A` | `array_like` | 待分解的方阵 | `[[4,2],[1,3]]` |

#### 示例代码

```python
import numpy as np
from scipy import linalg

A = np.array([[4, 2], [1, 3]])

eigvals, eigvecs = linalg.eig(A)

print(f"矩阵 A:\n{A}")
print(f"\n特征值: {eigvals}")

print("\n验证 Av = λv:")
for i in range(len(eigvals)):
    v = eigvecs[:, i]
    lam = eigvals[i]
    lhs = A @ v
    rhs = lam * v
    print(f"  λ={lam:.1f}: A@v={np.round(lhs, 4)}, λ*v={np.round(rhs, 4)}")
```

#### 输出

```text
矩阵 A:
[[4 2]
 [1 3]]

特征值: [5.+0.j 2.+0.j]

验证 Av = λv:
  λ=5.0+0.0j: A@v=[4.4721 2.2361], λ*v=[4.4721 2.2361]
  λ=2.0+0.0j: A@v=[-1.4142  1.4142], λ*v=[-1.4142  1.4142]
```

#### 理解重点

- 矩阵 A 的特征值为 5 和 2（特征方程 $\det(A-\lambda I)=0 \to \lambda^2-7\lambda+10=0$）
- Av = λv 验证成功：矩阵作用在特征向量上只改变长度不改变方向
- 特征值以复数返回（即使是实数也带 `+0.j`），使用 `.real` 提取实部
- 几何意义：特征向量是矩阵变换下方向不变的"轴"，特征值是沿该轴的缩放因子

## 5. 线性方程组求解

### `linalg.solve`

#### 作用

直接求解 Ax = b。比 `np.linalg.inv(A) @ b` 更高效和数值稳定（内部使用 LU 分解）。要求 A 是方阵且非奇异。对于超定方程组（m > n），应使用 `linalg.lstsq` 求最小二乘解。

#### 重点方法

```python
linalg.solve(A, b)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `A` | `array_like` | 系数矩阵（方阵） | `[[3, 1], [1, 2]]` |
| `b` | `array_like` | 右端向量 | `[9, 8]` |

#### 示例代码

```python
import numpy as np
from scipy import linalg

# 3x + y = 9, x + 2y = 8
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])

x = linalg.solve(A, b)

print(f"方程组: 3x+y=9, x+2y=8")
print(f"解: x={x[0]:.1f}, y={x[1]:.1f}")
print(f"验证 A@x = {A @ x}")
```

#### 输出

```text
方程组: 3x+y=9, x+2y=8
解: x=2.0, y=3.0
验证 A@x = [9. 8.]
```

#### 理解重点

- 解为 x=2, y=3，代入验证：3×2+3=9, 2+2×3=8
- `linalg.solve` 比显式求逆再相乘更快且数值更稳定
- 验证 A@x = b 是检验解正确性的标准方法
- 对于大规模稀疏方程组，使用 `scipy.sparse.linalg.spsolve`

## 常见坑

1. `eig` 返回复数——即使矩阵是实数，特征值也以复数形式返回，需 `.real` 提取
2. `svd` 的 s 是一维数组——不是对角矩阵，重构时需手动构造 $\Sigma$ 矩阵
3. `solve` 要求方阵——A 必须是方阵且非奇异，非方阵用 `lstsq`
4. `scipy.linalg` vs `numpy.linalg`——SciPy 版本功能更多（如 LU 分解），且有时更快
5. 数值精度——矩阵条件数大时，分解和求解的精度会下降

## 小结

- LU 分解（A=PLU）是高斯消元的矩阵形式——用于高效求解线性方程组
- QR 分解（A=QR）产生正交矩阵和上三角矩阵——用于最小二乘和特征值算法
- SVD（$A=U\Sigma V^H$）是最通用的矩阵分解——奇异值反映信息结构，用于降维和压缩
- 特征值分解揭示矩阵的内在几何特性——特征向量是变换的不变方向
- `linalg.solve` 是求解线性方程组的首选——比求逆矩阵更高效稳定
