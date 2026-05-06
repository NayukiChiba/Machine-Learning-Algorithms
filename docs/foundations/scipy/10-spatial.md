---
title: SciPy 空间数据与距离计算
outline: deep
---

# SciPy 空间数据与距离计算

## 本章目标

1. 掌握常见距离度量（欧氏、曼哈顿、切比雪夫、余弦）及距离矩阵计算
2. 学会使用 KD 树进行高效最近邻搜索
3. 理解凸包（Convex Hull）的计算与属性
4. 掌握 Voronoi 图的构建与区域分析
5. 了解 Delaunay 三角剖分及其与 Voronoi 图的对偶关系

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `distance.euclidean(u, v)` | 函数 | 欧氏距离（L2 范数） |
| `distance.cdist(XA, XB, metric)` | 函数 | 成对距离矩阵 |
| `spatial.KDTree(data)` | 构造器 | 构建 KD 树空间索引 |
| `tree.query(x, k)` | 方法 | K 最近邻查询 |
| `spatial.ConvexHull(points)` | 构造器 | 凸包计算 |
| `spatial.Voronoi(points)` | 构造器 | Voronoi 图 |
| `spatial.Delaunay(points)` | 构造器 | Delaunay 三角剖分 |

## 1. 距离计算

### `distance.euclidean` / `distance.cdist`

#### 作用

`distance.euclidean` 计算欧氏距离（L2 范数）。`distance.cdist` 计算两组点之间的成对距离矩阵。另有 `cityblock`（曼哈顿 L1）、`chebyshev`（L∞）、`cosine`（1−余弦相似度）等度量。

#### 重点方法

```python
distance.euclidean(u, v)
distance.cdist(XA, XB, metric='euclidean')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `u` / `v` | `array_like` | 两个向量 | `[1, 2, 3]`, `[4, 5, 6]` |
| `XA` | `array_like` | 第一组点，形状 `(m, d)` | `[[0,0],[1,0],[0,1],[1,1]]` |
| `XB` | `array_like` | 第二组点，形状 `(n, d)` | 同上 |
| `metric` | `str` | 距离度量：`'euclidean'` / `'cityblock'` / `'cosine'` 等，默认为 `'euclidean'` | `'cosine'` |

#### 示例代码

```python
import numpy as np
from scipy.spatial import distance

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(f"向量 a: {a}, 向量 b: {b}")
print(f"欧氏距离: {distance.euclidean(a, b):.4f}")
print(f"曼哈顿距离: {distance.cityblock(a, b):.4f}")
print(f"切比雪夫距离: {distance.chebyshev(a, b):.4f}")
print(f"余弦距离: {distance.cosine(a, b):.4f}")

# 距离矩阵
pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
dMat = distance.cdist(pts, pts, 'euclidean')
print(f"\n距离矩阵:\n{np.round(dMat, 4)}")
```

#### 输出

```text
向量 a: [1 2 3], 向量 b: [4 5 6]
欧氏距离: 5.1962
曼哈顿距离: 9.0000
切比雪夫距离: 3.0000
余弦距离: 0.0254

距离矩阵:
[[0.     1.     1.     1.4142]
 [1.     0.     1.4142 1.    ]
 [1.     1.4142 0.     1.    ]
 [1.4142 1.     1.     0.    ]]
```

#### 理解重点

- 欧氏距离 = $\sqrt{3^2+3^2+3^2} = \sqrt{27} \approx 5.196$——直线距离
- 曼哈顿距离 = |3|+|3|+|3| = 9——沿坐标轴走（如城市街区）
- 切比雪夫距离 = max(3,3,3) = 3——各维度最大差值
- 余弦距离 ≈ 0.025——两向量方向几乎一致（余弦相似度 ≈ 0.975）
- 距离矩阵对称，对角线为 0；(0,0) 到 (1,1) 的距离为 $\sqrt{2} \approx 1.414$

## 2. KD 树

### `spatial.KDTree`

#### 作用

KD 树将数据空间递归二分，实现高效空间查询。查询时间复杂度 $O(\log n)$，远优于暴力搜索的 $O(n)$。适用于低维空间（通常 d < 20），高维时性能退化。

#### 重点方法

```python
tree = spatial.KDTree(data, leafsize=10)
tree.query(x, k=1)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `data` | `array_like` | 点集数据，形状 `(n, d)` | 100 个随机二维点 |
| `leafsize` | `int` | 叶节点最大点数，默认为 `10` | `10` |
| `x` | `array_like` | 查询点 | `[5, 5]` |
| `k` | `int` | 最近邻个数，默认为 `1` | `5` |

#### 示例代码

```python
import numpy as np
from scipy import spatial

np.random.seed(42)
points = np.random.rand(100, 2) * 10

tree = spatial.KDTree(points)
print(f"点集大小: {len(points)}")

# 最近邻
queryPt = [5, 5]
dist, idx = tree.query(queryPt)
print(f"查询点: {queryPt}")
print(f"最近邻: {points[idx]} (距离: {dist:.4f})")

# K=5 最近邻
dists, idxs = tree.query(queryPt, k=5)
print("\n5 个最近邻:")
for d, i in zip(dists, idxs):
    print(f"  {points[i]} (距离: {d:.4f})")
```

#### 输出

```text
点集大小: 100
查询点: [5, 5]
最近邻: [4.9785 5.0266] (距离: 0.0351)

5 个最近邻:
  [4.9785 5.0266] (距离: 0.0351)
  [5.2716 5.1034] (距离: 0.2867)
  [4.6477 5.1877] (距离: 0.3980)
  [5.4408 5.0297] (距离: 0.4414)
  [4.6399 4.7096] (距离: 0.4646)
```

#### 理解重点

- KD 树将 100 个点组织成树结构——查询最近邻只需访问少数节点
- 最近邻距离 ≈ 0.035——在 [0,10]×[0,10] 区域内 100 个点分布比较密集
- K=5 查询返回按距离排序的 5 个最近邻
- 构建时间 $O(n\log n)$，查询 $O(\log n)$——适合反复查询场景
- `query_ball_point(x, r)` 可查询半径 r 内的所有点

## 3. 凸包

### `spatial.ConvexHull`

#### 作用

计算点集的凸包（包围所有点的最小凸多边形）。`hull.vertices` 返回凸包顶点的索引。**注意**：二维中 `hull.volume` 返回面积，`hull.area` 返回周长——SciPy 使用通用 N 维术语。

#### 重点方法

```python
spatial.ConvexHull(points)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `points` | `array_like` | 点集数据，形状 `(n, d)` | 30 个随机二维点 |

#### 示例代码

```python
import numpy as np
from scipy import spatial

np.random.seed(42)
points = np.random.rand(30, 2)

hull = spatial.ConvexHull(points)

print(f"点数: {len(points)}")
print(f"凸包顶点数: {len(hull.vertices)}")
print(f"凸包顶点索引: {hull.vertices}")
print(f"凸包面积 (volume): {hull.volume:.4f}")
```

#### 输出

```text
点数: 30
凸包顶点数: 8
凸包顶点索引: [16  1  3 22 14 23 15 27]
凸包面积 (volume): 0.8014
```

#### 理解重点

- 30 个随机点中约 8 个位于凸包边界上——其余点在凸包内部
- 凸包面积接近 1——点在 [0,1]×[0,1] 均匀分布，凸包几乎覆盖整个正方形
- 二维中 `hull.volume` = 面积，`hull.area` = 周长
- `ConvexHull` 基于 Qhull 库——时间复杂度 $O(n\log n)$

## 4. Voronoi 图

### `spatial.Voronoi`

#### 作用

计算 Voronoi 图——将空间划分为每个种子点的最近邻区域。每个 Voronoi 区域内的所有位置，到对应种子点的距离比到其他任何种子点都近。

#### 重点方法

```python
spatial.Voronoi(points)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `points` | `array_like` | 种子点集，形状 `(n, 2)` | 10 个随机二维点 |

#### 示例代码

```python
import numpy as np
from scipy import spatial

np.random.seed(42)
points = np.random.rand(10, 2)

vor = spatial.Voronoi(points)

print(f"种子点数: {len(points)}")
print(f"Voronoi 顶点数: {len(vor.vertices)}")
print(f"Voronoi 区域数: {len(vor.regions)}")

print("\n点对应的区域:")
for i, regionIdx in enumerate(vor.point_region):
    print(f"  点 {i} -> 区域 {regionIdx}")
```

#### 输出

```text
种子点数: 10
Voronoi 顶点数: 13
Voronoi 区域数: 11

点对应的区域:
  点 0 -> 区域 1
  点 1 -> 区域 3
  点 2 -> 区域 2
  点 3 -> 区域 8
  点 4 -> 区域 5
  点 5 -> 区域 10
  点 6 -> 区域 7
  点 7 -> 区域 4
  点 8 -> 区域 6
  点 9 -> 区域 9
```

#### 理解重点

- 10 个种子点产生 11 个区域（含空区域），13 个 Voronoi 顶点
- `vor.regions` 中包含 −1 的区域延伸到无穷远（边界点）
- 每个 Voronoi 区域内所有位置到对应种子点的距离最近
- 广泛用于：最近邻区域划分、选址问题、GIS、晶体结构分析

## 5. Delaunay 三角剖分

### `spatial.Delaunay`

#### 作用

计算 Delaunay 三角剖分——将点集连接成不重叠的三角形，最大化最小角（避免狭长三角形）。与 Voronoi 图互为对偶：Voronoi 的每条边垂直平分对应的 Delaunay 边。

#### 重点方法

```python
spatial.Delaunay(points)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `points` | `array_like` | 点集数据，形状 `(n, 2)` | 15 个随机二维点 |

#### 示例代码

```python
import numpy as np
from scipy import spatial

np.random.seed(42)
points = np.random.rand(15, 2)

tri = spatial.Delaunay(points)

print(f"点数: {len(points)}")
print(f"三角形数: {len(tri.simplices)}")

print("\n前 3 个三角形顶点索引:")
for i, simplex in enumerate(tri.simplices[:3]):
    print(f"  三角形 {i}: {simplex}")
```

#### 输出

```text
点数: 15
三角形数: 20

前 3 个三角形顶点索引:
  三角形 0: [13  3  7]
  三角形 1: [ 2  7 10]
  三角形 2: [ 7  3  2]
```

#### 理解重点

- 15 个点生成 20 个三角形——符合 Euler 公式：三角形数 ≈ 2n − h − 2
- Delaunay 满足"空圆性质"：每个三角形的外接圆内不包含其他点
- 与 Voronoi 图的对偶关系：Delaunay 两点相连 ⇔ 它们的 Voronoi 区域共享边
- 应用场景：有限元网格生成、地形建模、三维重建、路径规划
- `tri.find_simplex(point)` 可查找某个点位于哪个三角形内

## 常见坑

1. `cosine` 返回距离不是相似度：`distance.cosine` 返回 $1 - \cos\theta$，范围 [0, 2]——不是余弦相似度
2. `hull.volume` 在二维中是面积：二维中 `volume`=面积、`area`=周长——容易混淆
3. KD 树不适合高维：维度超过 ~20 时 KD 树退化为暴力搜索——应使用 Ball Tree
4. Voronoi 无穷区域：边界点的 Voronoi 区域延伸到无穷远——`regions` 中包含 −1
5. `cdist` 内存：n 个点的距离矩阵大小为 $n^2$——大规模点集可能内存不足

## 小结

- `distance` 模块提供丰富的距离度量——`cdist` 高效计算成对距离矩阵
- KD 树将最近邻搜索从 $O(n)$ 加速到 $O(\log n)$——空间索引的核心数据结构
- 凸包是包围点集的最小凸多边形——用于形状分析和碰撞检测
- Voronoi 图将空间划分为最近邻区域——广泛用于选址和区域分析
- Delaunay 三角剖分与 Voronoi 图互为对偶——有限元网格生成的基础算法
- 空间数据核心流程：选择距离度量 → 构建空间索引 → 执行空间查询/分析
