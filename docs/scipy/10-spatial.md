# 空间数据与距离计算

> 对应代码: [10_spatial.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Scipy/10_spatial.py)

## 距离计算

```python
from scipy.spatial import distance

distance.euclidean(a, b)   # 欧氏距离
distance.cityblock(a, b)   # 曼哈顿距离
distance.cosine(a, b)      # 余弦距离

# 距离矩阵
dist_matrix = distance.cdist(points, points, 'euclidean')
```

## KD 树

```python
from scipy import spatial

tree = spatial.KDTree(points)
dist, idx = tree.query(query_point)       # 最近邻
dists, idxs = tree.query(query_point, k=3)  # K 最近邻
```

## 计算几何

```python
# 凸包
hull = spatial.ConvexHull(points)

# Voronoi 图
vor = spatial.Voronoi(points)

# Delaunay 三角剖分
tri = spatial.Delaunay(points)
```

## 练习

```bash
python Basic/Scipy/10_spatial.py
```
