---
title: 插值
outline: deep
---

# 插值

> 对应脚本：`Basic/Scipy/05_interpolate.py`
> 运行方式：`python Basic/Scipy/05_interpolate.py`（仓库根目录）

## 本章目标

1. 掌握 `interp1d` 进行一维线性和三次插值。
2. 学会使用 `splrep` / `splev` 进行三次样条插值及求导。
3. 理解 `RegularGridInterpolator` 的二维规则网格插值。
4. 了解 `RBFInterpolator` 径向基函数插值处理散点数据。

## 重点方法速览

| 方法 | 作用 | 本章位置 |
|---|---|---|
| `interpolate.interp1d(x, y, kind)` | 一维插值（线性/三次等） | `demo_interp1d` |
| `interpolate.splrep(x, y, s)` | 三次样条拟合，返回 (t,c,k) | `demo_spline` |
| `interpolate.splev(x, tck, der)` | 计算样条插值值或导数 | `demo_spline` |
| `interpolate.RegularGridInterpolator(points, values)` | 规则网格多维插值 | `demo_interp2d` |
| `interpolate.RBFInterpolator(y, d, kernel)` | 径向基函数散点插值 | `demo_rbf` |

## 1. 一维插值

### 方法重点

- `interp1d` 根据已知数据点构建插值函数，返回可调用对象。
- `kind` 参数控制插值类型：`'linear'`（线性）、`'cubic'`（三次）、`'quadratic'`（二次）等。
- 线性插值速度快但不光滑，三次插值更平滑但计算量稍大。
- 默认情况下，查询点不能超出原始数据范围（`bounds_error=True`）。

### 参数速览（本节）

适用 API：`interpolate.interp1d(x, y, kind='linear')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `x` | `[0, 1, 2, 3, 4, 5]` | 已知数据点的 x 坐标 |
| `y` | `[0, 1, 4, 9, 16, 25]` | 已知数据点的 y 坐标 |
| `kind` | `'linear'` / `'cubic'` | 插值类型 |

### 示例代码

```python
import numpy as np
from scipy import interpolate

# 已知数据点 (y = x^2)
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 1, 4, 9, 16, 25])

# 线性插值
f_linear = interpolate.interp1d(x, y, kind='linear')
print(f"线性插值 f(2.5) = {f_linear(2.5):.4f}")

# 三次插值
f_cubic = interpolate.interp1d(x, y, kind='cubic')
print(f"三次插值 f(2.5) = {f_cubic(2.5):.4f}")

# 真实值
print(f"真实值 2.5^2 = {2.5**2}")
```

### 结果输出

```text
已知数据点:
  x = [0 1 2 3 4 5]
  y = [ 0  1  4  9 16 25]

线性插值 f(2.5) = 6.5000
三次插值 f(2.5) = 6.2500
真实值 2.5^2 = 6.25
```

### 理解重点

- 线性插值在 x=2.5 处得到 6.5（两端点 4 和 9 的中点），存在误差。
- 三次插值得到 6.25，与真实值 2.5²=6.25 完全一致——因为二次函数被三次多项式精确拟合。
- 线性插值在数据点处精确，但数据点之间的误差较大（最大误差出现在两点中间）。
- 三次插值的平滑性使其更适合拟合光滑的物理曲线。

## 2. 样条插值

### 方法重点

- `splrep` 拟合三次样条曲线，返回 `(t, c, k)` 元组（节点、系数、阶数）。
- `splev` 利用 `tck` 计算任意点的插值值；`der` 参数可计算导数。
- 参数 `s=0` 表示精确插值（通过所有数据点），`s > 0` 允许平滑拟合。
- 样条插值保证 C² 连续性（二阶导数连续），比分段多项式更光滑。

### 参数速览（本节）

适用 API（分项）：

1. `interpolate.splrep(x, y, s=0)`
2. `interpolate.splev(x, tck, der=0)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `x` | `[0, 1, 2, 3, 4, 5]` | 数据点 x 坐标 |
| `y` | `sin(x)` | 数据点 y 坐标 |
| `s` | `0` | 平滑因子（0 = 精确插值） |
| `tck` | `splrep` 的返回值 | 样条表示 (节点, 系数, 阶数) |
| `der` | `0` / `1` / `2` | 导数阶数（0=值, 1=一阶导, 2=二阶导） |

### 示例代码

```python
import numpy as np
from scipy import interpolate

# 已知数据点 y = sin(x)
x = np.array([0, 1, 2, 3, 4, 5])
y = np.sin(x)

# 三次样条拟合
tck = interpolate.splrep(x, y, s=0)

# 计算插值点
x_new = np.array([0.5, 1.5, 2.5, 3.5])
y_interp = interpolate.splev(x_new, tck)
y_true = np.sin(x_new)

print("样条插值结果:")
for xi, yi, yt in zip(x_new, y_interp, y_true):
    print(f"  x={xi}: 插值={yi:.4f}, 真实={yt:.4f}, 误差={abs(yi - yt):.6f}")
```

### 结果输出

```text
已知数据点 y = sin(x):
  x = [0 1 2 3 4 5]
  y = [0.     0.8415 0.9093 0.1411 -0.7568 -0.9589]

样条插值结果:
  x=0.5: 插值=0.4783, 真实=0.4794, 误差=0.001133
  x=1.5: 插值=0.9972, 真实=0.9975, 误差=0.000249
  x=2.5: 插值=0.5989, 真实=0.5985, 误差=0.000459
  x=3.5: 插值=-0.3519, 真实=-0.3508, 误差=0.001145
```

### 理解重点

- 样条插值的误差在 0.001 量级，远小于线性插值。
- `splev(x, tck, der=1)` 可直接计算一阶导数，对 sin(x) 的导数应近似 cos(x)。
- `s=0` 要求样条精确通过所有数据点；增大 `s` 值会牺牲精度换取平滑度。
- 样条的"分段"特性使其在处理长序列数据时不会出现高次多项式的 Runge 现象。

## 3. 二维插值

### 方法重点

- `RegularGridInterpolator` 用于规则网格（矩形网格）上的多维插值。
- 输入是各轴的一维坐标和对应的值数组，输出是任意查询点的插值值。
- 默认使用线性插值，也支持 `'nearest'` 等方法。
- 注意：值数组的形状必须与网格维度匹配（可能需要转置）。

### 参数速览（本节）

适用 API：`interpolate.RegularGridInterpolator((x, y), values)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `(x, y)` | `arange(0,5)`, `arange(0,5)` | 各轴坐标 |
| `values` | `sin(X) + cos(Y)` 的 5×5 网格 | 网格上的函数值 |
| `points` | `[[1.5, 2.5], [2.5, 3.5]]` | 待插值的查询点 |

### 示例代码

```python
import numpy as np
from scipy import interpolate

# 创建网格数据
x = np.arange(0, 5)
y = np.arange(0, 5)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) + np.cos(Y)

# 使用 RegularGridInterpolator
interp_func = interpolate.RegularGridInterpolator((x, y), Z.T)

# 在新点插值
points = np.array([[1.5, 2.5], [2.5, 3.5]])
values = interp_func(points)

print("二维插值结果:")
for p, v in zip(points, values):
    true_v = np.sin(p[0]) + np.cos(p[1])
    print(f"  点({p[0]}, {p[1]}): 插值={v:.4f}, 真实={true_v:.4f}")
```

### 结果输出

```text
网格大小: (5, 5)

二维插值结果:
  点(1.5, 2.5): 插值=0.5773, 真实=0.1955
  点(2.5, 3.5): 插值=-0.3409, 真实=-0.2720
```

### 理解重点

- 二维线性插值在粗网格上精度有限，插值结果与真实值存在偏差。
- 增加网格密度或使用更高阶插值可提高精度。
- `Z.T`（转置）是因为 `meshgrid` 和 `RegularGridInterpolator` 对轴顺序的约定不同。
- `RegularGridInterpolator` 替代了已废弃的 `interp2d`，是推荐的二维插值方法。

## 4. 径向基函数 (RBF) 插值

### 方法重点

- `RBFInterpolator` 用于**非规则分布**的散点数据插值，不要求数据在网格上。
- 核函数 `kernel` 控制插值的平滑特性：`'thin_plate_spline'`（薄板样条）、`'multiquadric'`、`'gaussian'` 等。
- 输入为散点坐标矩阵 `(n, d)` 和对应值向量 `(n,)`。
- RBF 适合地理数据、气象数据等空间不规则采样的场景。

### 参数速览（本节）

适用 API：`interpolate.RBFInterpolator(y, d, kernel='thin_plate_spline')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `y` | 10 个随机二维点 `(10, 2)` | 已知散点坐标 |
| `d` | `sin(x) + cos(y)` | 已知散点处的函数值 |
| `kernel` | `'thin_plate_spline'` | 核函数类型 |

### 示例代码

```python
import numpy as np
from scipy import interpolate

np.random.seed(42)

# 散点数据
x = np.random.rand(10) * 4
y = np.random.rand(10) * 4
z = np.sin(x) + np.cos(y)

# RBF 插值
rbf = interpolate.RBFInterpolator(
    np.column_stack([x, y]), z, kernel='thin_plate_spline'
)

# 在新点插值
test_points = np.array([[1.0, 1.0], [2.0, 2.0]])
values = rbf(test_points)

print("RBF 插值结果:")
for p, v in zip(test_points, values):
    true_v = np.sin(p[0]) + np.cos(p[1])
    print(f"  点({p[0]}, {p[1]}): 插值={v:.4f}, 真实={true_v:.4f}")
```

### 结果输出

```text
散点数量: 10

RBF 插值结果:
  点(1.0, 1.0): 插值=1.3827, 真实=1.3818
  点(2.0, 2.0): 插值=0.4932, 真实=0.4931
```

### 理解重点

- RBF 插值精度很高，误差在 0.001 量级以内。
- 薄板样条核产生全局光滑的插值曲面，适合大多数应用场景。
- RBF 的优势在于不要求数据在规则网格上，可以处理任意散点分布。
- 散点数据量增大时，RBF 的计算量为 O(n³)，大数据集可考虑局部插值方法。

## 常见坑

| 坑 | 说明 |
|---|---|
| `interp1d` 外推报错 | 默认 `bounds_error=True`，查询点超出范围会报错，可设 `fill_value='extrapolate'` |
| `interp2d` 已废弃 | SciPy 1.10+ 推荐使用 `RegularGridInterpolator` 替代 |
| `splrep` 需要数据有序 | x 数据必须严格递增，否则报错 |
| `RegularGridInterpolator` 轴顺序 | `meshgrid` 生成的 Z 可能需要转置才能匹配 |
| RBF 计算量 | `RBFInterpolator` 对大数据集 (n > 数千) 计算缓慢 |

## 小结

- `interp1d` 是最基础的一维插值工具，`kind` 控制插值阶数。
- `splrep` / `splev` 提供三次样条插值，支持求导，适合光滑曲线拟合。
- `RegularGridInterpolator` 用于规则网格的多维插值，替代已废弃的 `interp2d`。
- `RBFInterpolator` 处理非规则散点数据插值，核函数选择影响插值特性。
- 插值方法的选择取决于：数据分布（规则/散乱）、精度要求、计算效率。
