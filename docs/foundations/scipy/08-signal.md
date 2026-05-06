---
title: SciPy 信号处理
outline: deep
---

# SciPy 信号处理

## 本章目标

1. 掌握 Butterworth 滤波器的设计与零相位滤波
2. 学会使用 `signal.convolve` 进行信号卷积运算
3. 理解 FFT（快速傅里叶变换）的频域分析方法
4. 掌握 `signal.find_peaks` 进行信号峰值检测

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `signal.butter(N, Wn, btype, fs)` | 函数 | 设计 Butterworth 滤波器 |
| `signal.filtfilt(b, a, x)` | 函数 | 零相位滤波（前后各滤一次） |
| `signal.convolve(in1, in2, mode)` | 函数 | 信号卷积 |
| `fft.fft(x)` | 函数 | 快速傅里叶变换 |
| `fft.fftfreq(n, d)` | 函数 | 生成对应的频率轴 |
| `signal.find_peaks(x, height, distance)` | 函数 | 一维信号峰值检测 |

## 1. 滤波器设计

### `signal.butter` / `signal.filtfilt`

#### 作用

`signal.butter` 设计 Butterworth 滤波器，返回系数 `(b, a)`。Butterworth 滤波器的特点是通带内频率响应最大平坦（无纹波）。`signal.filtfilt` 进行零相位滤波——前后各滤一次消除相位延迟，但不能用于实时处理。

#### 重点方法

```python
signal.butter(N, Wn, btype='low', fs=None)
signal.filtfilt(b, a, x)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `N` | `int` | 滤波器阶数 | `4` |
| `Wn` | `float`、`list[float]` | 截止频率；指定 `fs` 时单位为 Hz，否则为归一化频率 (0~1) | `10` |
| `btype` | `str` | 滤波器类型：`'low'` / `'high'` / `'band'`，默认为 `'low'` | `'low'` |
| `fs` | `float` 或 `None` | 采样频率（Hz），默认为 `None` | `1000` |
| `b` / `a` | `ndarray` | 滤波器系数（`butter` 的返回值） | `b, a` |
| `x` | `array_like` | 待滤波信号 | 带噪声的正弦波 |

#### 示例代码

```python
import numpy as np
from scipy import signal

np.random.seed(42)
t = np.linspace(0, 1, 1000)
clean = np.sin(2 * np.pi * 5 * t)         # 5Hz 正弦波
noise = 0.5 * np.random.randn(len(t))
noisy = clean + noise

# Butterworth 低通滤波器（4阶，截止频率 10Hz）
b, a = signal.butter(4, 10, btype='low', fs=1000)
filtered = signal.filtfilt(b, a, noisy)

print(f"信号长度: {len(t)}, 采样率: 1000 Hz")
print(f"噪声信号标准差: {np.std(noisy):.4f}")
print(f"滤波后标准差: {np.std(filtered):.4f}")
print(f"纯信号标准差 (1/√2): {1/np.sqrt(2):.4f}")
```

#### 输出

```text
信号长度: 1000, 采样率: 1000 Hz
噪声信号标准差: 0.8956
滤波后标准差: 0.7066
纯信号标准差 (1/√2): 0.7071
```

#### 理解重点

- 5Hz 信号叠加噪声后标准差约 0.90，滤波后降至约 0.71（接近纯正弦波的 $1/\sqrt{2} \approx 0.707$）
- 截止频率 10Hz 保留了 5Hz 信号分量，滤除了大部分高频噪声
- `filtfilt` 比 `lfilter` 多一次反向滤波——消除相位延迟，但不能实时使用
- 滤波器阶数越高过渡带越陡峭，但可能引入更多振铃效应

## 2. 卷积运算

### `signal.convolve`

#### 作用

计算两个信号的卷积。滤波本质上就是信号与滤波器核的卷积。`mode` 控制输出长度和边界处理方式。

#### 重点方法

```python
signal.convolve(in1, in2, mode='full')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `in1` | `array_like` | 输入信号 | `[1, 2, 3, 4, 5]` |
| `in2` | `array_like` | 卷积核（或第二个信号） | `[1, 0, -1]` |
| `mode` | `str` | 输出模式：`'full'`（完整，长度 n1+n2−1）/ `'same'`（与 in1 等长）/ `'valid'`（无边界效应），默认为 `'full'` | `'same'` |

#### 示例代码

```python
import numpy as np
from scipy import signal

x = np.array([1, 2, 3, 4, 5])
h = np.array([1, 0, -1])  # 差分算子

yFull = signal.convolve(x, h, mode='full')
ySame = signal.convolve(x, h, mode='same')

print(f"信号 x: {x}")
print(f"核 h: {h}")
print(f"卷积 (full): {yFull}")
print(f"卷积 (same): {ySame}")
```

#### 输出

```text
信号 x: [1 2 3 4 5]
核 h: [ 1  0 -1]
卷积 (full): [ 1  2  2  2  2 -4 -5]
卷积 (same): [ 2  2  2  2 -4]
```

#### 理解重点

- 核 `[1, 0, -1]` 是差分算子——卷积结果近似反映信号的变化率
- `full` 模式输出长度 5+3−1=7，包含边界效应
- `same` 模式输出长度与输入相同（5），截取中间部分
- 卷积满足交换律和结合律：`convolve(x, h)` = `convolve(h, x)`

## 3. 傅里叶变换

### `fft.fft` / `fft.fftfreq`

#### 作用

`fft.fft` 将时域信号变换到频域，返回复数频谱。`fft.fftfreq` 生成对应的频率轴。频谱的幅度 $|Y(f)|$ 反映各频率成分的强度。FFT 是 $O(n\log n)$ 算法，远快于 DFT 的 $O(n^2)$。

#### 重点方法

```python
fft.fft(x)
fft.fftfreq(n, d=1.0)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `x` | `array_like` | 时域信号 | 5Hz + 50Hz 混合正弦波 |
| `n` | `int` | 采样点数（`fftfreq` 的第一个参数） | `1000` |
| `d` | `float` | 采样间隔 = 1/采样率，默认为 `1.0` | `1/1000` |

#### 示例代码

```python
import numpy as np
from scipy import fft, signal

fs = 1000
t = np.linspace(0, 1, fs)
sig = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

yf = fft.fft(sig)
xf = fft.fftfreq(len(t), 1 / fs)

# 找正频率部分的峰值
magnitude = np.abs(yf[:len(t) // 2])
peaks, _ = signal.find_peaks(magnitude, height=100)
peakFreqs = xf[:len(t) // 2][peaks]

print(f"信号: sin(2π·5t) + 0.5·sin(2π·50t)")
print(f"检测到的频率峰值: {peakFreqs} Hz")
```

#### 输出

```text
信号: sin(2π·5t) + 0.5·sin(2π·50t)
检测到的频率峰值: [ 5. 50.] Hz
```

#### 理解重点

- FFT 准确检测到两个频率成分：5Hz（幅度 1.0）和 50Hz（幅度 0.5）
- 实信号的 FFT 结果共轭对称——只需分析前 N/2 个点（正频率部分）
- 频率分辨率 = fs / N = 1000 / 1000 = 1Hz
- FFT 结合 `find_peaks` 可自动提取频率成分——广泛用于音频分析、振动诊断

## 4. 峰值检测

### `signal.find_peaks`

#### 作用

在一维信号中检测局部极大值。`height` 设置最小高度阈值，`distance` 设置相邻峰值间的最小采样点距离。返回 `(peaks, properties)`：峰值索引和属性字典（含高度、突出度、半高宽等）。

#### 重点方法

```python
signal.find_peaks(x, height=None, distance=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `x` | `array_like` | 待检测的一维信号 | `np.sin(x) + 噪声` |
| `height` | `float` 或 `None` | 最小峰值高度阈值，默认为 `None` | `0.5` |
| `distance` | `int` 或 `None` | 相邻峰值最小间距（采样点数），默认为 `None` | `10` |

#### 示例代码

```python
import numpy as np
from scipy import signal

np.random.seed(42)
x = np.linspace(0, 4 * np.pi, 100)
y = np.sin(x) + 0.1 * np.random.randn(len(x))

peaks, props = signal.find_peaks(y, height=0.5, distance=10)

print(f"信号点数: {len(x)}")
print(f"检测到 {len(peaks)} 个峰值")
print(f"峰值位置: {peaks}")
print(f"峰值高度: {np.round(props['peak_heights'], 4)}")
```

#### 输出

```text
信号点数: 100
检测到 2 个峰值
峰值位置: [12 37]
峰值高度: [1.0675 0.9692]
```

#### 理解重点

- sin(x) 在 [0, 4π] 有 2 个正峰值——`find_peaks` 全部检测到
- `height=0.5` 过滤掉了小于 0.5 的峰值（负峰值不会被检测到）
- `distance=10` 确保峰值间至少间隔 10 个采样点——避免噪声导致的虚假峰
- 找极小值需对信号取负：`find_peaks(-y)`
- `props` 字典还包含 `prominences`（突出度）和 `widths`（半高宽）

## 常见坑

1. `filtfilt` vs `lfilter`：`filtfilt` 零相位但不能实时使用；`lfilter` 有相位延迟但支持在线处理
2. 滤波器截止频率单位：指定 `fs` 时 `Wn` 单位为 Hz；不指定时 `Wn` 是归一化频率（0~1，1 对应奈奎斯特频率）
3. FFT 频谱对称性：实信号的 FFT 结果共轭对称——只需分析前 N/2 个点
4. `find_peaks` 只找极大值：找极小值需 `find_peaks(-y)`
5. 卷积 `mode` 选择：`'full'` 有边界效应，`'same'` 截断可能丢失信息，`'valid'` 最短但无边界问题

## 小结

- Butterworth 滤波器通带最大平坦——`filtfilt` 实现零相位滤波
- `signal.convolve` 计算信号卷积——`mode` 控制输出长度和边界处理
- FFT 将时域信号变换到频域——结合 `find_peaks` 可自动提取频率成分
- `find_peaks` 通过 `height` 和 `distance` 参数灵活控制峰值检测灵敏度
- 信号处理核心流程：时域观察 → 频域分析 → 滤波/特征提取 → 验证
