"""
线性代数扩展
对应文档: ../../docs/scipy/07-linalg.md
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


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
    
    # === 可视化 ===
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    matrices = [A, P, L, U]
    titles = ['原矩阵 A', '置换矩阵 P', '下三角矩阵 L', '上三角矩阵 U']
    cmaps = ['coolwarm', 'Blues', 'Greens', 'Oranges']
    
    for ax, mat, title, cmap in zip(axes, matrices, titles, cmaps):
        im = ax.imshow(mat, cmap=cmap, aspect='auto')
        ax.set_title(title)
        
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center', 
                       fontsize=10, color='black' if abs(mat[i, j]) < 5 else 'white')
        
        ax.set_xticks(range(mat.shape[1]))
        ax.set_yticks(range(mat.shape[0]))
        plt.colorbar(im, ax=ax, shrink=0.6)
    
    plt.suptitle('LU 分解: A = P @ L @ U', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/scipy/07_lu.png', dpi=150, bbox_inches='tight')


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
    
    # === 可视化 ===
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    ax1 = axes[0]
    im1 = ax1.imshow(A, cmap='coolwarm', aspect='auto')
    ax1.set_title('原矩阵 A')
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax1.text(j, i, f'{A[i, j]}', ha='center', va='center', fontsize=12)
    plt.colorbar(im1, ax=ax1, shrink=0.6)
    
    ax2 = axes[1]
    im2 = ax2.imshow(U, cmap='Blues', aspect='auto')
    ax2.set_title('U (左奇异向量)')
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            ax2.text(j, i, f'{U[i, j]:.2f}', ha='center', va='center', fontsize=10)
    plt.colorbar(im2, ax=ax2, shrink=0.6)
    
    ax3 = axes[2]
    ax3.bar(range(len(s)), s, color='green', alpha=0.7, edgecolor='black')
    ax3.set_title('奇异值 σ')
    ax3.set_xlabel('索引')
    ax3.set_ylabel('奇异值')
    ax3.set_xticks(range(len(s)))
    ax3.grid(True, alpha=0.3, axis='y')
    
    ax4 = axes[3]
    im4 = ax4.imshow(Vh, cmap='Oranges', aspect='auto')
    ax4.set_title('V^H (右奇异向量)')
    for i in range(Vh.shape[0]):
        for j in range(Vh.shape[1]):
            ax4.text(j, i, f'{Vh[i, j]:.2f}', ha='center', va='center', fontsize=10)
    plt.colorbar(im4, ax=ax4, shrink=0.6)
    
    plt.suptitle('奇异值分解 (SVD): A = U Σ V^H', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/scipy/07_svd.png', dpi=150, bbox_inches='tight')


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
    
    # === 可视化 ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 矩阵热力图
    ax1 = axes[0]
    im = ax1.imshow(A, cmap='coolwarm', aspect='auto')
    ax1.set_title('矩阵 A')
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax1.text(j, i, f'{A[i, j]}', ha='center', va='center', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax1)
    
    # 特征向量可视化
    ax2 = axes[1]
    colors = ['blue', 'red']
    origin = np.zeros(2)
    
    for i, (lam, color) in enumerate(zip(eigenvalues, colors)):
        v = eigenvectors[:, i].real
        # 归一化用于显示
        v_norm = v / np.linalg.norm(v)
        ax2.quiver(0, 0, v_norm[0], v_norm[1], angles='xy', scale_units='xy', scale=1, 
                   color=color, width=0.02, label=f'v{i+1} (λ={lam.real:.2f})')
        
        # 变换后的向量
        Av = A @ v_norm
        ax2.quiver(0, 0, Av[0], Av[1], angles='xy', scale_units='xy', scale=1,
                   color=color, width=0.01, alpha=0.5, linestyle='--')
    
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_aspect('equal')
    ax2.axhline(0, color='black', lw=0.5)
    ax2.axvline(0, color='black', lw=0.5)
    ax2.set_title('特征向量 (实线) 与变换结果 (虚线)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/scipy/07_eig.png', dpi=150, bbox_inches='tight')


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
    import os
    os.makedirs('outputs/scipy', exist_ok=True)
    
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
