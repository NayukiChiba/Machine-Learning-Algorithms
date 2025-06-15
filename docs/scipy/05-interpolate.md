# 插值方法

> 对应代码: [05_interpolate.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Scipy/05_interpolate.py)

## 一维插值

```python
from scipy import interpolate

f = interpolate.interp1d(x, y, kind='linear')
f = interpolate.interp1d(x, y, kind='cubic')

y_new = f(x_new)
```

kind 参数:

- `linear`: 线性插值
- `cubic`: 三次插值
- `nearest`: 最近邻

## 样条插值

```python
# B-样条
tck = interpolate.splrep(x, y, s=0)
y_new = interpolate.splev(x_new, tck)
```

## 二维插值

```python
# 规则网格
interp_func = interpolate.RegularGridInterpolator((x, y), Z)
values = interp_func(points)
```

## 径向基函数 (RBF)

```python
rbf = interpolate.RBFInterpolator(points, values, kernel='thin_plate_spline')
```

## 练习

```bash
python Basic/Scipy/05_interpolate.py
```
