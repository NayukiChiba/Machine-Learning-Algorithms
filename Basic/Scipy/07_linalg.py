"""
线性代数扩展
对应文档: ../../docs/scipy/07-linalg.md
"""

import numpy as np
from scipy import linalg


def demo_lu():
    """演示 LU 分解"""
    print("=" * 50)
    print("1. LU 分解")
    print("=" * 50)
    
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    
    P, L, U = linalg.lu(A)
    
    print(f"原矩阵 A:\n{A}")
    print(f"\nP (置换矩阵):\n{P}")
    print(f"\nL (下三角):\n{np.round(L, 4)}")
    print(f"\nU (上三角):\n{np.round(U, 4)}")
    print(f"\n验证 P @ L @ U:\n{np.round(P @ L @ U, 4)}")


def demo_qr():
    """演示 QR 分解"""
    print("=" * 50)
    print("2. QR 分解")
    print("=" * 50)
    
    A = np.array([[1, 2], [3, 4], [5, 6]])
    
    Q, R = linalg.qr(A)
    
    print(f"原矩阵 A (3x2):\n{A}")
    print(f"\nQ (正交矩阵):\n{np.round(Q, 4)}")
    print(f"\nR (上三角):\n{np.round(R, 4)}")
    print(f"\n验证 Q @ R:\n{np.round(Q @ R, 4)}")


def demo_svd():
    """演示奇异值分解"""
    print("=" * 50)
    print("3. SVD 分解")
    print("=" * 50)
    
    A = np.array([[1, 2, 3], [4, 5, 6]])
    
    U, s, Vh = linalg.svd(A)
    
    print(f"原矩阵 A (2x3):\n{A}")
    print(f"\nU:\n{np.round(U, 4)}")
    print(f"\n奇异值 s: {np.round(s, 4)}")
    print(f"\nVh:\n{np.round(Vh, 4)}")
    
    # 重构
    S = np.zeros_like(A, dtype=float)
    S[:len(s), :len(s)] = np.diag(s)
    reconstructed = U @ S @ Vh
    print(f"\n重构 U @ S @ Vh:\n{np.round(reconstructed, 4)}")


def demo_eig():
    """演示特征值分解"""
    print("=" * 50)
    print("4. 特征值与特征向量")
    print("=" * 50)
    
    A = np.array([[4, 2], [1, 3]])
    
    eigenvalues, eigenvectors = linalg.eig(A)
    
    print(f"矩阵 A:\n{A}")
    print(f"\n特征值: {eigenvalues}")
    print(f"\n特征向量:\n{eigenvectors}")
    
    # 验证 A @ v = λ @ v
    print("\n验证 A @ v = λ * v:")
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lam = eigenvalues[i]
        lhs = A @ v
        rhs = lam * v
        print(f"  λ={lam:.4f}: A@v = {np.round(lhs, 4)}, λ*v = {np.round(rhs, 4)}")


def demo_solve():
    """演示线性方程组求解"""
    print("=" * 50)
    print("5. 线性方程组求解")
    print("=" * 50)
    
    # Ax = b
    A = np.array([[3, 1], [1, 2]])
    b = np.array([9, 8])
    
    x = linalg.solve(A, b)
    
    print(f"方程组:")
    print(f"  3x + y = 9")
    print(f"  x + 2y = 8")
    print(f"\n解: x = {x[0]:.4f}, y = {x[1]:.4f}")
    print(f"验证 A @ x = {A @ x}")


def demo_all():
    """运行所有演示"""
    demo_lu()
    print()
    demo_qr()
    print()
    demo_svd()
    print()
    demo_eig()
    print()
    demo_solve()


if __name__ == "__main__":
    demo_all()
