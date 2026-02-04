"""
稀疏矩阵
对应文档: ../../docs/scipy/09-sparse.md
"""

import numpy as np
from scipy import sparse
import matplotlib

matplotlib.use("Agg")  # 使用非交互式后端，避免字体警告
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def demo_sparse_create():
    """演示稀疏矩阵创建"""
    print("=" * 50)
    print("1. 稀疏矩阵创建")
    print("=" * 50)

    # 从密集矩阵创建
    dense = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]])

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

    # === 可视化 ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 密集矩阵
    ax1 = axes[0]
    im1 = ax1.imshow(dense, cmap="Blues")
    ax1.set_title("密集表示")
    for i in range(dense.shape[0]):
        for j in range(dense.shape[1]):
            ax1.text(
                j,
                i,
                f"{dense[i, j]}",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )
    plt.colorbar(im1, ax=ax1)

    # 稀疏结构
    ax2 = axes[1]
    ax2.spy(csr, markersize=20, color="blue")
    ax2.set_title("稀疏结构 (spy 图)")
    ax2.set_xlabel("列")
    ax2.set_ylabel("行")

    plt.tight_layout()
    plt.savefig("outputs/scipy/09_create.png", dpi=150, bbox_inches="tight")


def demo_sparse_operations():
    """演示稀疏矩阵操作"""
    print("=" * 50)
    print("2. 稀疏矩阵操作")
    print("=" * 50)

    np.random.seed(42)
    A = sparse.random(20, 20, density=0.1, format="csr")
    print(f"随机稀疏矩阵 A (密度=0.1):")
    print(f"  形状: {A.shape}")
    print(f"  非零元素数: {A.nnz}")
    print(f"  密度: {A.nnz / (A.shape[0] * A.shape[1]):.2f}")
    print()

    # 矩阵运算
    B = sparse.eye(20, format="csr")
    C = A + B
    print(f"A + I 的非零元素数: {C.nnz}")

    # 转换为密集矩阵
    dense = A.toarray()
    print(f"\n转换为密集矩阵 shape: {dense.shape}")

    # === 可视化 ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax1 = axes[0]
    ax1.spy(A, markersize=3, color="blue")
    ax1.set_title(f"稀疏矩阵 A\n(nnz={A.nnz}, 密度={A.nnz / 400:.1%})")
    ax1.set_xlabel("列")
    ax1.set_ylabel("行")

    ax2 = axes[1]
    ax2.spy(B, markersize=3, color="green")
    ax2.set_title(f"单位矩阵 I\n(nnz={B.nnz})")
    ax2.set_xlabel("列")
    ax2.set_ylabel("行")

    ax3 = axes[2]
    ax3.spy(C, markersize=3, color="red")
    ax3.set_title(f"A + I\n(nnz={C.nnz})")
    ax3.set_xlabel("列")
    ax3.set_ylabel("行")

    plt.tight_layout()
    plt.savefig("outputs/scipy/09_ops.png", dpi=150, bbox_inches="tight")


def demo_sparse_linalg():
    """演示稀疏线性代数"""
    print("=" * 50)
    print("3. 稀疏线性代数")
    print("=" * 50)

    from scipy.sparse import linalg as splinalg

    # 创建稀疏系统
    n = 100
    A = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), format="csr")
    b = np.ones(n)

    print(f"稀疏方程组 Ax = b")
    print(f"  矩阵大小: {n}x{n}")
    print(f"  非零元素: {A.nnz}")

    # 直接求解
    x = splinalg.spsolve(A, b)
    print(f"  解的范数: {np.linalg.norm(x):.4f}")
    print(f"  残差: {np.linalg.norm(A @ x - b):.2e}")

    # === 可视化 ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 矩阵结构
    ax1 = axes[0]
    ax1.spy(A, markersize=1, color="blue")
    ax1.set_title(f"三对角矩阵 A\n({n}×{n}, nnz={A.nnz})")
    ax1.set_xlabel("列")
    ax1.set_ylabel("行")

    # 解向量
    ax2 = axes[1]
    ax2.plot(range(n), x, "b-", lw=2)
    ax2.set_title("解向量 x")
    ax2.set_xlabel("索引")
    ax2.set_ylabel("x 值")
    ax2.grid(True, alpha=0.3)

    # 残差
    ax3 = axes[2]
    residual = np.abs(A @ x - b)
    ax3.semilogy(range(n), residual, "r-", lw=1)
    ax3.set_title(f"残差 |Ax - b|\n(最大残差: {np.max(residual):.2e})")
    ax3.set_xlabel("索引")
    ax3.set_ylabel("残差 (对数尺度)")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/scipy/09_linalg.png", dpi=150, bbox_inches="tight")


def demo_sparse_efficiency():
    """演示稀疏矩阵效率"""
    print("=" * 50)
    print("4. 稀疏矩阵效率")
    print("=" * 50)

    sizes = [100, 500, 1000, 2000, 5000]
    density = 0.01

    dense_memory = []
    sparse_memory = []

    for n in sizes:
        # 密集矩阵内存
        dm = n * n * 8 / 1024 / 1024  # MB
        dense_memory.append(dm)

        # 稀疏矩阵 (1% 密度)
        nnz = int(n * n * density)
        sm = nnz * (8 + 4 + 4) / 1024 / 1024  # MB
        sparse_memory.append(sm)

    n = 1000
    dense_mem = n * n * 8  # float64 = 8 bytes
    nnz = int(n * n * density)
    sparse_mem = nnz * (8 + 4 + 4)  # data + row + col

    print(f"矩阵大小: {n}x{n}")
    print(f"密度: {density * 100}%")
    print(f"\n内存使用:")
    print(f"  密集矩阵: {dense_mem / 1024 / 1024:.2f} MB")
    print(f"  稀疏矩阵: {sparse_mem / 1024:.2f} KB")
    print(f"  节省: {(1 - sparse_mem / dense_mem) * 100:.1f}%")

    # === 可视化 ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 内存对比
    ax1 = axes[0]
    x = np.arange(len(sizes))
    width = 0.35

    bars1 = ax1.bar(
        x - width / 2, dense_memory, width, label="密集矩阵", color="coral", alpha=0.7
    )
    bars2 = ax1.bar(
        x + width / 2,
        sparse_memory,
        width,
        label="稀疏矩阵 (1%密度)",
        color="steelblue",
        alpha=0.7,
    )

    ax1.set_xlabel("矩阵大小")
    ax1.set_ylabel("内存 (MB)")
    ax1.set_title("内存使用对比")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{n}×{n}" for n in sizes])
    ax1.legend()
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3, axis="y")

    # 节省百分比
    ax2 = axes[1]
    savings = [(1 - s / d) * 100 for d, s in zip(dense_memory, sparse_memory)]
    ax2.bar(range(len(sizes)), savings, color="green", alpha=0.7, edgecolor="black")
    ax2.axhline(99, color="red", linestyle="--", label="99% 节省")
    ax2.set_xlabel("矩阵大小")
    ax2.set_ylabel("内存节省 (%)")
    ax2.set_title(f"稀疏矩阵内存节省 (密度={density * 100}%)")
    ax2.set_xticks(range(len(sizes)))
    ax2.set_xticklabels([f"{n}×{n}" for n in sizes])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(95, 100)

    plt.tight_layout()
    plt.savefig("outputs/scipy/09_efficiency.png", dpi=150, bbox_inches="tight")


def demo_all():
    """运行所有演示"""
    import os

    os.makedirs("outputs/scipy", exist_ok=True)

    demo_sparse_create()
    print()
    demo_sparse_operations()
    print()
    demo_sparse_linalg()
    print()
    demo_sparse_efficiency()


if __name__ == "__main__":
    demo_all()
