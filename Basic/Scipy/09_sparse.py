"""
稀疏矩阵
对应文档: ../../docs/scipy/09-sparse.md
"""

import numpy as np
from scipy import sparse


def demo_sparse_create():
    """演示稀疏矩阵创建"""
    print("=" * 50)
    print("1. 稀疏矩阵创建")
    print("=" * 50)
    
    # 从密集矩阵创建
    dense = np.array([[1, 0, 0, 0],
                      [0, 2, 0, 0],
                      [0, 0, 3, 0],
                      [0, 0, 0, 4]])
    
    # CSR 格式 (压缩行)
    csr = sparse.csr_matrix(dense)
    print(f"密集矩阵:\n{dense}")
    print(f"\nCSR 稀疏矩阵:\n{csr}")
    print(f"数据: {csr.data}")
    print(f"列索引: {csr.indices}")
    print()
    
    # COO 格式 (坐标)
    row = np.array([0, 1, 2, 3])
    col = np.array([0, 1, 2, 3])
    data = np.array([1, 2, 3, 4])
    coo = sparse.coo_matrix((data, (row, col)), shape=(4, 4))
    print(f"COO 格式:\n{coo}")


def demo_sparse_operations():
    """演示稀疏矩阵操作"""
    print("=" * 50)
    print("2. 稀疏矩阵操作")
    print("=" * 50)
    
    A = sparse.random(5, 5, density=0.3, format='csr')
    print(f"随机稀疏矩阵 A (密度=0.3):")
    print(f"  形状: {A.shape}")
    print(f"  非零元素数: {A.nnz}")
    print(f"  密度: {A.nnz / (A.shape[0]*A.shape[1]):.2f}")
    print()
    
    # 矩阵运算
    B = sparse.eye(5, format='csr')
    C = A + B
    print(f"A + I 的非零元素数: {C.nnz}")
    
    # 转换为密集矩阵
    dense = A.toarray()
    print(f"\n转换为密集矩阵 shape: {dense.shape}")


def demo_sparse_linalg():
    """演示稀疏线性代数"""
    print("=" * 50)
    print("3. 稀疏线性代数")
    print("=" * 50)
    
    from scipy.sparse import linalg as splinalg
    
    # 创建稀疏系统
    n = 100
    A = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), format='csr')
    b = np.ones(n)
    
    print(f"稀疏方程组 Ax = b")
    print(f"  矩阵大小: {n}x{n}")
    print(f"  非零元素: {A.nnz}")
    
    # 直接求解
    x = splinalg.spsolve(A, b)
    print(f"  解的范数: {np.linalg.norm(x):.4f}")
    print(f"  残差: {np.linalg.norm(A @ x - b):.2e}")


def demo_sparse_efficiency():
    """演示稀疏矩阵效率"""
    print("=" * 50)
    print("4. 稀疏矩阵效率")
    print("=" * 50)
    
    n = 1000
    
    # 密集矩阵内存
    dense_memory = n * n * 8  # float64 = 8 bytes
    
    # 稀疏矩阵 (1% 密度)
    density = 0.01
    nnz = int(n * n * density)
    sparse_memory = nnz * (8 + 4 + 4)  # data + row + col
    
    print(f"矩阵大小: {n}x{n}")
    print(f"密度: {density*100}%")
    print(f"\n内存使用:")
    print(f"  密集矩阵: {dense_memory/1024/1024:.2f} MB")
    print(f"  稀疏矩阵: {sparse_memory/1024:.2f} KB")
    print(f"  节省: {(1-sparse_memory/dense_memory)*100:.1f}%")


def demo_all():
    """运行所有演示"""
    demo_sparse_create()
    print()
    demo_sparse_operations()
    print()
    demo_sparse_linalg()
    print()
    demo_sparse_efficiency()


if __name__ == "__main__":
    demo_all()
