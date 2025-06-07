"""
NumPy 线性代数运算
对应文档: ../../docs/numpy/06-linalg.md

使用方式：
    python 06_linalg.py
"""

import numpy as np


def demo_matrix_multiplication():
    """矩阵乘法"""
    print("=" * 50)
    print("1. 矩阵乘法 (dot, @)")
    print("=" * 50)
    
    # 向量点积
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    print(f"向量 a = {a}")
    print(f"向量 b = {b}")
    print(f"点积 np.dot(a, b) = {np.dot(a, b)}")
    print(f"验证: 1*4 + 2*5 + 3*6 = {1*4 + 2*5 + 3*6}")
    print()
    
    # 矩阵乘法
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    print(f"矩阵 A:\n{A}")
    print(f"矩阵 B:\n{B}")
    print(f"A @ B:\n{A @ B}")
    print(f"np.dot(A, B):\n{np.dot(A, B)}")


def demo_transpose():
    """矩阵转置"""
    print("=" * 50)
    print("2. 矩阵转置")
    print("=" * 50)
    
    A = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"矩阵 A (2x3):\n{A}")
    print(f"A.T (3x2):\n{A.T}")
    print(f"np.transpose(A):\n{np.transpose(A)}")


def demo_determinant_inverse():
    """行列式和逆矩阵"""
    print("=" * 50)
    print("3. 行列式和逆矩阵")
    print("=" * 50)
    
    A = np.array([[4, 7], [2, 6]])
    print(f"矩阵 A:\n{A}")
    print()
    
    # 行列式
    det = np.linalg.det(A)
    print(f"行列式 det(A) = {det:.4f}")
    print()
    
    # 逆矩阵
    A_inv = np.linalg.inv(A)
    print(f"逆矩阵 A^(-1):\n{A_inv}")
    print()
    
    # 验证 A @ A^(-1) = I
    result = A @ A_inv
    print(f"验证 A @ A^(-1):\n{result.round(10)}")
    print(f"是否为单位矩阵: {np.allclose(result, np.eye(2))}")


def demo_eigenvalues():
    """特征值和特征向量"""
    print("=" * 50)
    print("4. 特征值和特征向量")
    print("=" * 50)
    
    A = np.array([[4, 2], [1, 3]])
    print(f"矩阵 A:\n{A}")
    print()
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"特征值: {eigenvalues}")
    print(f"特征向量:\n{eigenvectors}")
    print()
    
    # 验证 A @ v = λ @ v
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lam = eigenvalues[i]
        left = A @ v
        right = lam * v
        print(f"验证特征值 {lam:.4f}:")
        print(f"  A @ v = {left}")
        print(f"  λ * v = {right}")
        print(f"  相等: {np.allclose(left, right)}")


def demo_solve_linear():
    """解线性方程组"""
    print("=" * 50)
    print("5. 解线性方程组 Ax = b")
    print("=" * 50)
    
    # 方程组:
    # 2x + y = 5
    # x + 3y = 7
    A = np.array([[2, 1], [1, 3]])
    b = np.array([5, 7])
    
    print("方程组:")
    print("  2x + y = 5")
    print("  x + 3y = 7")
    print()
    
    # 解方程
    x = np.linalg.solve(A, b)
    print(f"解: x = {x}")
    print(f"  x = {x[0]:.4f}")
    print(f"  y = {x[1]:.4f}")
    print()
    
    # 验证
    result = A @ x
    print(f"验证 A @ x = {result}")
    print(f"是否等于 b: {np.allclose(result, b)}")


def demo_norm():
    """向量和矩阵范数"""
    print("=" * 50)
    print("6. 向量和矩阵范数")
    print("=" * 50)
    
    v = np.array([3, 4])
    print(f"向量 v = {v}")
    print()
    
    # 向量范数
    print(f"L1 范数 (曼哈顿距离): {np.linalg.norm(v, ord=1)}")
    print(f"L2 范数 (欧几里得距离): {np.linalg.norm(v, ord=2)}")
    print(f"  验证: sqrt(3² + 4²) = {np.sqrt(3**2 + 4**2)}")
    print(f"无穷范数: {np.linalg.norm(v, ord=np.inf)}")
    print()
    
    # 矩阵范数
    A = np.array([[1, 2], [3, 4]])
    print(f"矩阵 A:\n{A}")
    print(f"Frobenius 范数: {np.linalg.norm(A):.4f}")


def demo_all():
    """运行所有演示"""
    demo_matrix_multiplication()
    print()
    demo_transpose()
    print()
    demo_determinant_inverse()
    print()
    demo_eigenvalues()
    print()
    demo_solve_linear()
    print()
    demo_norm()


if __name__ == "__main__":
    demo_all()
