"""
空间数据与距离计算
对应文档: ../../docs/scipy/10-spatial.md
"""

import numpy as np
from scipy import spatial


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
    dist_matrix = distance.cdist(points, points, 'euclidean')
    print(f"4个点的距离矩阵:\n{np.round(dist_matrix, 4)}")


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
    dists, idxs = tree.query(query_point, k=3)
    print("3个最近邻:")
    for d, i in zip(dists, idxs):
        print(f"  {points[i]} (距离: {d:.4f})")


def demo_convex_hull():
    """演示凸包"""
    print("=" * 50)
    print("3. 凸包")
    print("=" * 50)
    
    np.random.seed(42)
    points = np.random.rand(20, 2)
    
    # 计算凸包
    hull = spatial.ConvexHull(points)
    
    print(f"点数: {len(points)}")
    print(f"凸包顶点数: {len(hull.vertices)}")
    print(f"凸包顶点索引: {hull.vertices}")
    print(f"凸包面积: {hull.volume:.4f}")


def demo_voronoi():
    """演示 Voronoi 图"""
    print("=" * 50)
    print("4. Voronoi 图")
    print("=" * 50)
    
    np.random.seed(42)
    points = np.random.rand(5, 2)
    
    # 计算 Voronoi 图
    vor = spatial.Voronoi(points)
    
    print(f"点数: {len(points)}")
    print(f"Voronoi 顶点数: {len(vor.vertices)}")
    print(f"Voronoi 区域数: {len(vor.regions)}")
    print(f"\n点对应的区域:")
    for i, region_idx in enumerate(vor.point_region):
        print(f"  点 {i} -> 区域 {region_idx}")


def demo_delaunay():
    """演示 Delaunay 三角剖分"""
    print("=" * 50)
    print("5. Delaunay 三角剖分")
    print("=" * 50)
    
    np.random.seed(42)
    points = np.random.rand(10, 2)
    
    # 计算 Delaunay 三角剖分
    tri = spatial.Delaunay(points)
    
    print(f"点数: {len(points)}")
    print(f"三角形数: {len(tri.simplices)}")
    print(f"\n前3个三角形顶点索引:")
    for i, simplex in enumerate(tri.simplices[:3]):
        print(f"  三角形 {i}: {simplex}")


def demo_all():
    """运行所有演示"""
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
