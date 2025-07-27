"""
空间数据与距离计算
对应文档: ../../docs/scipy/10-spatial.md
"""

import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def demo_distance():
    """演示距离计算"""
    print("=" * 50)
    print("1. 距离计算")
    print("=" * 50)

    from scipy.spatial import distance

    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    print(f"向量 a: {a}")
    print(f"向量 b: {b}")
    print()

    print("距离度量:")
    print(f"  欧氏距离: {distance.euclidean(a, b):.4f}")
    print(f"  曼哈顿距离: {distance.cityblock(a, b):.4f}")
    print(f"  切比雪夫距离: {distance.chebyshev(a, b):.4f}")
    print(f"  余弦距离: {distance.cosine(a, b):.4f}")
    print()

    # 距离矩阵
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    dist_matrix = distance.cdist(points, points, "euclidean")
    print(f"4个点的距离矩阵:\n{np.round(dist_matrix, 4)}")

    # === 可视化 ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 距离矩阵热力图
    ax1 = axes[0]
    im = ax1.imshow(dist_matrix, cmap="YlOrRd")
    ax1.set_title("点间距离矩阵")
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    ax1.set_xticklabels(["(0,0)", "(1,0)", "(0,1)", "(1,1)"])
    ax1.set_yticklabels(["(0,0)", "(1,0)", "(0,1)", "(1,1)"])

    for i in range(4):
        for j in range(4):
            ax1.text(
                j, i, f"{dist_matrix[i, j]:.2f}", ha="center", va="center", fontsize=10
            )

    plt.colorbar(im, ax=ax1, label="欧氏距离")

    # 点的可视化
    ax2 = axes[1]
    ax2.scatter(points[:, 0], points[:, 1], c="blue", s=200, zorder=5)
    for i, p in enumerate(points):
        ax2.annotate(
            f"P{i}({p[0]},{p[1]})",
            p,
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=10,
        )

    # 画连线和距离
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            ax2.plot(
                [points[i, 0], points[j, 0]],
                [points[i, 1], points[j, 1]],
                "gray",
                linestyle="--",
                alpha=0.5,
            )

    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_aspect("equal")
    ax2.set_title("点的位置")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/scipy/10_distance.png", dpi=150, bbox_inches="tight")


def demo_kdtree():
    """演示 KD 树"""
    print("=" * 50)
    print("2. KD 树")
    print("=" * 50)

    # 创建点集
    np.random.seed(42)
    points = np.random.rand(100, 2) * 10

    # 构建 KD 树
    tree = spatial.KDTree(points)

    print(f"点集大小: {len(points)}")
    print()

    # 最近邻查询
    query_point = [5, 5]
    dist, idx = tree.query(query_point)
    print(f"查询点: {query_point}")
    print(f"最近邻: {points[idx]} (距离: {dist:.4f})")
    print()

    # K 最近邻
    dists, idxs = tree.query(query_point, k=5)
    print("5个最近邻:")
    for d, i in zip(dists, idxs):
        print(f"  {points[i]} (距离: {d:.4f})")

    # === 可视化 ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 所有点和查询点
    ax1 = axes[0]
    ax1.scatter(points[:, 0], points[:, 1], c="blue", s=30, alpha=0.6, label="数据点")
    ax1.scatter(
        query_point[0],
        query_point[1],
        c="red",
        s=200,
        marker="*",
        zorder=5,
        label="查询点",
    )
    ax1.scatter(
        points[idx, 0], points[idx, 1], c="green", s=100, zorder=5, label="最近邻"
    )

    # 画到最近邻的线
    ax1.plot(
        [query_point[0], points[idx, 0]], [query_point[1], points[idx, 1]], "g--", lw=2
    )

    ax1.set_title("KD 树最近邻搜索")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # K 最近邻
    ax2 = axes[1]
    ax2.scatter(points[:, 0], points[:, 1], c="blue", s=30, alpha=0.6, label="数据点")
    ax2.scatter(
        query_point[0],
        query_point[1],
        c="red",
        s=200,
        marker="*",
        zorder=5,
        label="查询点",
    )

    # K 个最近邻
    colors = plt.cm.Greens(np.linspace(0.4, 1, len(idxs)))
    for i, (d, j) in enumerate(zip(dists, idxs)):
        ax2.scatter(points[j, 0], points[j, 1], c=[colors[i]], s=100, zorder=5)
        ax2.plot(
            [query_point[0], points[j, 0]],
            [query_point[1], points[j, 1]],
            color=colors[i],
            linestyle="--",
            lw=1.5,
        )
        ax2.annotate(
            f"#{i + 1} (d={d:.2f})",
            points[j],
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )

    # 画搜索半径
    circle = plt.Circle(
        query_point,
        dists[-1],
        fill=False,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"搜索半径 r={dists[-1]:.2f}",
    )
    ax2.add_patch(circle)

    ax2.set_title("K 最近邻 (K=5)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("outputs/scipy/10_kdtree.png", dpi=150, bbox_inches="tight")


def demo_convex_hull():
    """演示凸包"""
    print("=" * 50)
    print("3. 凸包")
    print("=" * 50)

    np.random.seed(42)
    points = np.random.rand(30, 2)

    # 计算凸包
    hull = spatial.ConvexHull(points)

    print(f"点数: {len(points)}")
    print(f"凸包顶点数: {len(hull.vertices)}")
    print(f"凸包顶点索引: {hull.vertices}")
    print(f"凸包面积: {hull.volume:.4f}")

    # === 可视化 ===
    fig, ax = plt.subplots(figsize=(8, 8))

    # 所有点
    ax.scatter(points[:, 0], points[:, 1], c="blue", s=50, label="数据点")

    # 凸包顶点
    ax.scatter(
        points[hull.vertices, 0],
        points[hull.vertices, 1],
        c="red",
        s=100,
        zorder=5,
        label="凸包顶点",
    )

    # 凸包边界
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], "r-", lw=2)

    # 闭合凸包
    hull_points = points[hull.vertices]
    hull_points = np.vstack([hull_points, hull_points[0]])
    ax.fill(hull_points[:, 0], hull_points[:, 1], alpha=0.2, color="red")

    ax.set_title(
        f"凸包 (Convex Hull)\n顶点数: {len(hull.vertices)}, 面积: {hull.volume:.4f}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("outputs/scipy/10_hull.png", dpi=150, bbox_inches="tight")


def demo_voronoi():
    """演示 Voronoi 图"""
    print("=" * 50)
    print("4. Voronoi 图")
    print("=" * 50)

    np.random.seed(42)
    points = np.random.rand(10, 2)

    # 计算 Voronoi 图
    vor = spatial.Voronoi(points)

    print(f"点数: {len(points)}")
    print(f"Voronoi 顶点数: {len(vor.vertices)}")
    print(f"Voronoi 区域数: {len(vor.regions)}")
    print(f"\n点对应的区域:")
    for i, region_idx in enumerate(vor.point_region):
        print(f"  点 {i} -> 区域 {region_idx}")

    # === 可视化 ===
    fig, ax = plt.subplots(figsize=(10, 8))

    # 使用 voronoi_plot_2d
    spatial.voronoi_plot_2d(
        vor,
        ax=ax,
        show_vertices=True,
        line_colors="blue",
        line_width=2,
        line_alpha=0.6,
        point_size=10,
    )

    # 标记种子点
    ax.scatter(points[:, 0], points[:, 1], c="red", s=100, zorder=5, label="种子点")
    for i, p in enumerate(points):
        ax.annotate(f"P{i}", p, textcoords="offset points", xytext=(5, 5), fontsize=10)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_title("Voronoi 图")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/scipy/10_voronoi.png", dpi=150, bbox_inches="tight")


def demo_delaunay():
    """演示 Delaunay 三角剖分"""
    print("=" * 50)
    print("5. Delaunay 三角剖分")
    print("=" * 50)

    np.random.seed(42)
    points = np.random.rand(15, 2)

    # 计算 Delaunay 三角剖分
    tri = spatial.Delaunay(points)

    print(f"点数: {len(points)}")
    print(f"三角形数: {len(tri.simplices)}")
    print(f"\n前3个三角形顶点索引:")
    for i, simplex in enumerate(tri.simplices[:3]):
        print(f"  三角形 {i}: {simplex}")

    # === 可视化 ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Delaunay 三角剖分
    ax1 = axes[0]
    ax1.triplot(points[:, 0], points[:, 1], tri.simplices, "b-", lw=1.5)
    ax1.scatter(points[:, 0], points[:, 1], c="red", s=80, zorder=5)

    for i, p in enumerate(points):
        ax1.annotate(f"{i}", p, textcoords="offset points", xytext=(3, 3), fontsize=9)

    ax1.set_title(f"Delaunay 三角剖分\n({len(tri.simplices)} 个三角形)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")

    # 与 Voronoi 图重叠
    ax2 = axes[1]
    vor = spatial.Voronoi(points)

    # Delaunay
    ax2.triplot(
        points[:, 0],
        points[:, 1],
        tri.simplices,
        "b-",
        lw=1.5,
        alpha=0.5,
        label="Delaunay",
    )

    # Voronoi
    spatial.voronoi_plot_2d(
        vor,
        ax=ax2,
        show_vertices=False,
        line_colors="red",
        line_width=1.5,
        line_alpha=0.7,
        point_size=0,
    )

    ax2.scatter(points[:, 0], points[:, 1], c="green", s=80, zorder=5, label="点集")

    ax2.set_xlim(-0.2, 1.2)
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_title("Delaunay (蓝) vs Voronoi (红)\n(互为对偶)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("outputs/scipy/10_delaunay.png", dpi=150, bbox_inches="tight")


def demo_all():
    """运行所有演示"""
    import os

    os.makedirs("outputs/scipy", exist_ok=True)

    demo_distance()
    print()
    demo_kdtree()
    print()
    demo_convex_hull()
    print()
    demo_voronoi()
    print()
    demo_delaunay()


if __name__ == "__main__":
    demo_all()
