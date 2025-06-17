# 稀疏矩阵

> 对应代码: [09_sparse.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Scipy/09_sparse.py)

## 稀疏矩阵格式

| 格式 | 全称   | 适用场景         |
| ---- | ------ | ---------------- |
| CSR  | 压缩行 | 行切片、矩阵乘法 |
| CSC  | 压缩列 | 列切片           |
| COO  | 坐标   | 构建、转换       |
| LIL  | 链表   | 增量构建         |

## 创建稀疏矩阵

```python
from scipy import sparse

# 从密集矩阵
csr = sparse.csr_matrix(dense)

# COO 格式
coo = sparse.coo_matrix((data, (row, col)), shape=(n, n))

# 对角矩阵
diag = sparse.diags([1, 2, 3], [0, 1, -1])

# 随机稀疏矩阵
A = sparse.random(1000, 1000, density=0.01)
```

## 稀疏线性代数

```python
from scipy.sparse import linalg as splinalg

x = splinalg.spsolve(A, b)
```

## 练习

```bash
python Basic/Scipy/09_sparse.py
```
