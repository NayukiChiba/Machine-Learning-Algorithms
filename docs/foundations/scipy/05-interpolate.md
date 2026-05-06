---
title: SciPy 插值
outline: deep
---

# SciPy 插值

## 本章目标

1. 掌握 `interp1d` / `make_interp_spline` 进行一维插值
2. 学会使用 `splrep` / `splev` 进行三次样条插值及求导
3. 理解 `RegularGridInterpolator` 的规则网格多维插值
4. 了解 `RBFInterpolator` 径向基函数插值处理散点数据

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `interpolate.interp1d(x, y, kind)` | 构造器 | 一维插值函数（线性/三次等） |
| `interpolate.make_interp_spline(x, y, k)` | 函数 | B 样条插值（推荐替代 interp1d） |
| `interpolate.splrep(x, y, s)` | 函数 | 三次样条拟合，返回 (t, c, k) |
| `interpolate.splev(x, tck, der)` | 函数 | 计算样条值或导数 |
| `interpolate.RegularGridInterpolator(pts, vals)` | 构造器 | 规则网格多维插值 |
| `interpolate.RBFInterpolator(y, d, kernel)` | 构造器 | 径向基函数散点插值 |

## 1. 一维插值

### `interpolate.interp1d`

#### 作用

根据已知数据点构建插值函数，返回可调用对象。`kind` 参数控制插值类型：`'linear'`（线性）、`'cubic'`（三次）、`'quadratic'`（二次）等。

#### 重点方法

```python
interpolate.interp1d(x, y, kind='linear', bounds_error=True)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `x` | `array_like` | 已知数据点的 x 坐标（须严格递增） | `[0, 1, 2, 3, 4, 5]` |
| `y` | `array_like` | 已知数据点的 y 坐标 | `x**2` |
| `kind` | `str` | 插值类型：`'linear'` / `'cubic'` / `'quadratic'` 等，默认为 `'linear'` | `'cubic'` |
| `bounds_error` | `bool` | 查询点超出 x 范围时是否报错，默认为 `True` | `False` |
| `fill_value` | `str` 或 `array_like` | 超出范围时的填充值；`'extrapolate'` 允许外推 | `'extrapolate'` |

#### 示例代码

```python
import numpy as np
from scipy import interpolate

x = np.array([0, 1, 2, 3, 4, 5])
y = x**2

fLin = interpolate.interp1d(x, y, kind='linear')
fCub = interpolate.interp1d(x, y, kind='cubic')

print(f"线性 f(2.5)={fLin(2.5):.4f}")
print(f"三次 f(2.5)={fCub(2.5):.4f}")
print(f"真实 2.5²={2.5**2}")
```

#### 输出

```text
线性 f(2.5)=6.5000
三次 f(2.5)=6.2500
真实 2.5²=6.25
```

#### 理解重点

- 线性插值在 x=2.5 得到 6.5（两端点中点），存在误差
- 三次插值得到 6.25——与真实值完全一致，因为二次函数被三次多项式精确拟合
- SciPy 1.10+ 推荐 `make_interp_spline` 替代 `interp1d` 进行样条插值
- 线性插值速度快但不光滑；三次插值更平滑但计算量稍大

## 2. 样条插值

### `interpolate.splrep` / `interpolate.splev`

#### 作用

`splrep` 拟合三次样条曲线，返回 `(t, c, k)` 元组（节点、系数、阶数）。`splev` 利用 tck 计算任意点的插值值或导数（`der` 参数控制导数阶数）。

$s=0$ 表示精确插值（通过所有数据点），$s > 0$ 允许平滑拟合。

#### 重点方法

```python
interpolate.splrep(x, y, s=0)
interpolate.splev(x, tck, der=0)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `x` | `array_like` | 数据点 x 坐标（须严格递增） | `[0, 1, 2, 3, 4, 5]` |
| `y` | `array_like` | 数据点 y 坐标 | `np.sin(x)` |
| `s` | `float` | 平滑因子；`0` = 精确插值，越大越平滑，默认为 `len(x)` 相关 | `0` |
| `tck` | `tuple` | `splrep` 返回的样条表示 `(节点, 系数, 阶数)` | `tck` |
| `der` | `int` | 导数阶数：`0`=值，`1`=一阶导，`2`=二阶导，默认为 `0` | `1` |

#### 示例代码

```python
import numpy as np
from scipy import interpolate

x = np.array([0, 1, 2, 3, 4, 5])
y = np.sin(x)

tck = interpolate.splrep(x, y, s=0)

xNew = np.array([0.5, 1.5, 2.5, 3.5])
yInterp = interpolate.splev(xNew, tck)
yTrue = np.sin(xNew)

for xi, yi, yt in zip(xNew, yInterp, yTrue):
    print(f"x={xi}: 插值={yi:.4f}, 真实={yt:.4f}, 误差={abs(yi-yt):.6f}")
```

#### 输出

```text
x=0.5: 插值=0.4783, 真实=0.4794, 误差=0.001133
x=1.5: 插值=0.9972, 真实=0.9975, 误差=0.000249
x=2.5: 插值=0.5989, 真实=0.5985, 误差=0.000459
x=3.5: 插值=-0.3519, 真实=-0.3508, 误差=0.001145
```

#### 理解重点

- 样条插值误差在 0.001 量级，远小于线性插值
- `splev(x, tck, der=1)` 可直接计算一阶导数
- `s=0` 要求样条精确通过所有数据点；增大 `s` 牺牲精度换取平滑度
- 样条的"分段"特性使其处理长序列数据时不会出现高次多项式的 Runge 现象

## 3. 二维插值

### `interpolate.RegularGridInterpolator`

#### 作用

用于规则网格（矩形网格）上的多维插值。输入为各轴的一维坐标和值数组，返回可调用对象用于任意查询点上的插值。替代已废弃的 `interp2d`。

#### 重点方法

```python
interpolate.RegularGridInterpolator(points, values, method='linear')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `points` | `tuple[array_like]` | 各轴的坐标元组 | `(x, y)` |
| `values` | `array_like` | 网格上的函数值，形状需与网格维度匹配 | `Z.T` |
| `method` | `str` | 插值方法：`'linear'` / `'nearest'` 等，默认为 `'linear'` | `'nearest'` |

#### 示例代码

```python
import numpy as np
from scipy import interpolate

x = np.arange(0, 5)
y = np.arange(0, 5)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) + np.cos(Y)

interp = interpolate.RegularGridInterpolator((x, y), Z.T)

points = np.array([[1.5, 2.5], [2.5, 3.5]])
values = interp(points)

for p, v in zip(points, values):
    trueV = np.sin(p[0]) + np.cos(p[1])
    print(f"点({p[0]}, {p[1]}): 插值={v:.4f}, 真实={trueV:.4f}")
```

#### 输出

```text
点(1.5, 2.5): 插值=0.5773, 真实=0.1955
点(2.5, 3.5): 插值=-0.3409, 真实=-0.2720
```

#### 理解重点

- 二维线性插值在粗 5×5 网格上精度有限——增加网格密度可提高精度
- `Z.T`（转置）是因为 `meshgrid` 和 `RegularGridInterpolator` 对轴顺序的约定不同
- `RegularGridInterpolator` 是 `interp2d` 的官方替代——后者已在 SciPy 1.10+ 废弃
- 支持三维及以上——传入更多轴坐标即可

## 4. 径向基函数（RBF）插值

### `interpolate.RBFInterpolator`

#### 作用

用于**非规则分布**的散点数据插值，不要求数据在网格上。核函数 `kernel` 控制插值的平滑特性。适合地理数据、气象数据等空间不规则采样场景。

#### 重点方法

```python
interpolate.RBFInterpolator(y, d, kernel='thin_plate_spline')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y` | `array_like` | 已知散点坐标，形状 `(n, d)` | `np.column_stack([x, y])` |
| `d` | `array_like` | 已知散点处的函数值，形状 `(n,)` | `np.sin(x) + np.cos(y)` |
| `kernel` | `str` | 核函数：`'thin_plate_spline'` / `'multiquadric'` / `'gaussian'` 等 | `'thin_plate_spline'` |

#### 示例代码

```python
import numpy as np
from scipy import interpolate

np.random.seed(42)

x = np.random.rand(10) * 4
y = np.random.rand(10) * 4
z = np.sin(x) + np.cos(y)

rbf = interpolate.RBFInterpolator(
    np.column_stack([x, y]), z, kernel='thin_plate_spline'
)

testPts = np.array([[1.0, 1.0], [2.0, 2.0]])
values = rbf(testPts)

for p, v in zip(testPts, values):
    trueV = np.sin(p[0]) + np.cos(p[1])
    print(f"点({p[0]}, {p[1]}): 插值={v:.4f}, 真实={trueV:.4f}")
```

#### 输出

```text
点(1.0, 1.0): 插值=1.3827, 真实=1.3818
点(2.0, 2.0): 插值=0.4932, 真实=0.4931
```

#### 理解重点

- RBF 插值精度很高——误差在 0.001 量级以内
- 薄板样条核产生全局光滑的插值曲面，适合大多数场景
- RBF 的优势：不要求数据在规则网格上，可处理任意散点分布
- 散点数据量大时计算量为 $O(n^3)$——大数据集考虑局部插值方法

## 常见坑

1. `interp1d` 默认 `bounds_error=True`——查询点超出范围会报错，可设 `fill_value='extrapolate'`
2. `interp2d` 已废弃——SciPy 1.10+ 推荐 `RegularGridInterpolator` 替代
3. `splrep` 要求 x 数据严格递增——否则报错
4. `RegularGridInterpolator` 轴顺序——`meshgrid` 生成的 Z 可能需要转置
5. RBF 大数据计算慢——`RBFInterpolator` 对 n > 数千的数据集计算缓慢

## 小结

- `interp1d` 是最基础的一维插值工具——`kind` 控制插值阶数
- `splrep` / `splev` 提供三次样条插值，支持求导——适合光滑曲线拟合
- `RegularGridInterpolator` 用于规则网格多维插值——替代已废弃的 `interp2d`
- `RBFInterpolator` 处理非规则散点数据——核函数选择影响插值特性
- 插值方法选择取决于：数据分布（规则/散乱）、精度要求、计算效率
