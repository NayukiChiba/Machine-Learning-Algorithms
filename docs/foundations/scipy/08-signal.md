---
title: SciPy 信号处理
outline: deep
---

# SciPy 信号处理

> 对应脚本：`Basic/Scipy/08_signal.py`
> 运行方式：`python Basic/Scipy/08_signal.py`（仓库根目录）

## 本章目标

1. 掌握 Butterworth 滤波器的设计与使用。
2. 学会使用 `signal.convolve` 进行信号卷积运算。
3. 理解 FFT（快速傅里叶变换）的频域分析方法。
4. 掌握 `signal.find_peaks` 进行信号峰值检测。

## 重点方法速览

| 方法 | 作用 | 本章位置 |
|---|---|---|
| `signal.butter(N, Wn, btype, fs)` | 设计 Butterworth 滤波器 | `demo_filter` |
| `signal.filtfilt(b, a, x)` | 零相位滤波 | `demo_filter` |
| `signal.convolve(in1, in2, mode)` | 信号卷积 | `demo_convolution` |
| `fft.fft(x)` / `fft.fftfreq(n, d)` | 快速傅里叶变换 | `demo_fft` |
| `signal.find_peaks(x, height, distance)` | 峰值检测 | `demo_peak_finding` |

## 1. 滤波器设计

### 方法重点

- `signal.butter(N, Wn, btype, fs)` 设计 Butterworth 滤波器，返回滤波器系数 `(b, a)`。
- `signal.filtfilt(b, a, x)` 进行零相位滤波，前后各滤一次消除相位延迟。
- Butterworth 滤波器的特点是通带内频率响应最大平坦（无纹波）。
- `btype` 控制滤波器类型：`'low'`（低通）、`'high'`（高通）、`'band'`（带通）。

### 参数速览（本节）

适用 API（分项）：

1. `signal.butter(N, Wn, btype='low', fs=None)`
2. `signal.filtfilt(b, a, x)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `N` | `4` | 滤波器阶数 |
| `Wn` | `10` | 截止频率 (Hz) |
| `btype` | `'low'` | 低通滤波器 |
| `fs` | `1000` | 采样频率 (Hz) |
| `x` | 带噪声的 5Hz 正弦信号 | 待滤波信号 |

### 示例代码

```python
import numpy as np
from scipy import signal

# 创建带噪声的信号
np.random.seed(42)
t = np.linspace(0, 1, 1000)
clean_signal = np.sin(2 * np.pi * 5 * t)  # 5Hz 正弦波
noise = 0.5 * np.random.randn(len(t))
noisy_signal = clean_signal + noise

# Butterworth 低通滤波器
b, a = signal.butter(4, 10, btype='low', fs=1000)
filtered = signal.filtfilt(b, a, noisy_signal)

print(f"信号长度: {len(t)}")
print("采样率: 1000 Hz")
print("\nButterworth 低通滤波器:")
print("  阶数: 4")
print("  截止频率: 10 Hz")
print(f"  噪声信号标准差: {np.std(noisy_signal):.4f}")
print(f"  滤波后标准差: {np.std(filtered):.4f}")
```

### 结果输出

```text
信号长度: 1000
采样率: 1000 Hz

Butterworth 低通滤波器:
  阶数: 4
  截止频率: 10 Hz
  噪声信号标准差: 0.8956
  滤波后标准差: 0.7066
```

### 理解重点

- 5Hz 信号叠加高斯噪声后标准差约 0.90，滤波后降至约 0.71（接近纯正弦波的 1/√2 ≈ 0.707）。
- 截止频率 10Hz 保留了 5Hz 信号分量，滤除了大部分高频噪声。
- `filtfilt` 比 `lfilter` 多了一次反向滤波，消除了相位延迟，但不能用于实时处理。
- 滤波器阶数越高，过渡带越陡峭，但可能引入更多振铃效应。

## 2. 卷积运算

### 方法重点

- `signal.convolve(in1, in2, mode)` 计算两个信号的卷积。
- `mode='full'` 返回完整卷积结果（长度 = len(in1) + len(in2) - 1）。
- `mode='same'` 返回与第一个输入等长的结果。
- 卷积是信号处理的基本运算，滤波本质上就是信号与滤波器核的卷积。

### 参数速览（本节）

适用 API：`signal.convolve(in1, in2, mode='full')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `in1` | `[1, 2, 3, 4, 5]` | 输入信号 |
| `in2` | `[1, 0, -1]` | 卷积核（差分算子） |
| `mode` | `'full'` / `'same'` | 输出模式 |

### 示例代码

```python
import numpy as np
from scipy import signal

x = np.array([1, 2, 3, 4, 5])
h = np.array([1, 0, -1])

# 卷积
y_full = signal.convolve(x, h, mode='full')
y_same = signal.convolve(x, h, mode='same')

print(f"信号 x: {x}")
print(f"核 h: {h}")
print(f"\n卷积结果 (full): {y_full}")
print(f"卷积结果 (same): {y_same}")
```

### 结果输出

```text
信号 x: [1 2 3 4 5]
核 h: [ 1  0 -1]

卷积结果 (full): [ 1  2  2  2  2 -4 -5]
卷积结果 (same): [ 2  2  2  2 -4]
```

### 理解重点

- 核 `[1, 0, -1]` 是一个差分算子，卷积结果近似反映信号的变化率。
- `full` 模式输出长度为 5 + 3 - 1 = 7，包含边界效应。
- `same` 模式输出长度与输入信号相同（5），截取中间部分。
- 卷积满足交换律和结合律：`convolve(x, h)` = `convolve(h, x)`。

## 3. 傅里叶变换

### 方法重点

- `fft.fft(x)` 将时域信号变换到频域，返回复数频谱。
- `fft.fftfreq(n, d)` 生成对应的频率轴（n 为采样点数，d 为采样间隔）。
- 频谱的幅度 `|Y(f)|` 反映各频率成分的强度。
- FFT 是 O(n log n) 算法，远快于 DFT 的 O(n²)。

### 参数速览（本节）

适用 API（分项）：

1. `fft.fft(x)`
2. `fft.fftfreq(n, d=1.0)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `x` | 5Hz + 50Hz 混合信号 | 时域信号 |
| `n` | `1000` | 采样点数 |
| `d` | `1/1000` | 采样间隔（1/采样率） |
| `fs` | `1000` | 采样率 (Hz) |

### 示例代码

```python
import numpy as np
from scipy import fft, signal

# 创建混合频率信号
fs = 1000  # 采样率
t = np.linspace(0, 1, fs)
freq1, freq2 = 5, 50  # Hz
sig = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)

# FFT
yf = fft.fft(sig)
xf = fft.fftfreq(len(t), 1 / fs)

# 找峰值
magnitude = np.abs(yf[:len(t) // 2])
peaks, properties = signal.find_peaks(magnitude, height=100)
peak_freqs = xf[:len(t) // 2][peaks]

print(f"信号: sin(2pi*{freq1}t) + 0.5*sin(2pi*{freq2}t)")
print(f"检测到的频率峰值: {peak_freqs} Hz")
```

### 结果输出

```text
信号: sin(2pi*5t) + 0.5*sin(2pi*50t)
检测到的频率峰值: [ 5. 50.] Hz
```

### 理解重点

- FFT 准确检测到两个频率成分：5Hz（幅度 1.0）和 50Hz（幅度 0.5）。
- 频谱是对称的（实信号），只需分析前一半（正频率部分）。
- 频率分辨率 = fs / N = 1000 / 1000 = 1Hz，即能区分相差 1Hz 的频率成分。
- FFT 结合 `find_peaks` 可以自动提取信号中的频率成分，广泛用于音频分析、振动诊断等。

## 4. 峰值检测

### 方法重点

- `signal.find_peaks(x, height, distance)` 在一维信号中检测局部极大值。
- `height` 参数设置峰值的最小高度阈值。
- `distance` 参数设置相邻峰值之间的最小距离（采样点数）。
- 返回 `(peaks, properties)`：峰值索引和属性字典（包含高度等信息）。

### 参数速览（本节）

适用 API：`signal.find_peaks(x, height=None, distance=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `x` | `sin(x) + 噪声` | 待检测信号 |
| `height` | `0.5` | 最小峰值高度 |
| `distance` | `10` | 相邻峰值最小间距（采样点） |

### 示例代码

```python
import numpy as np
from scipy import signal

# 创建带峰值的信号
np.random.seed(42)
x = np.linspace(0, 4 * np.pi, 100)
y = np.sin(x) + 0.1 * np.random.randn(len(x))

# 找峰值
peaks, properties = signal.find_peaks(y, height=0.5, distance=10)

print(f"信号点数: {len(x)}")
print(f"检测到 {len(peaks)} 个峰值")
print(f"峰值位置: {peaks}")
print(f"峰值高度: {np.round(properties['peak_heights'], 4)}")
```

### 结果输出

```text
信号点数: 100
检测到 2 个峰值
峰值位置: [12 37]
峰值高度: [1.0675 0.9692]
```

### 理解重点

- sin(x) 在 [0, 4π] 有 2 个正峰值，`find_peaks` 全部检测到。
- `height=0.5` 过滤掉了小于 0.5 的峰值（负峰值不会被检测到）。
- `distance=10` 确保检测到的峰值之间至少间隔 10 个采样点，避免噪声导致的虚假峰。
- `peak_prominences` 和 `peak_widths` 可进一步分析峰值的突出度和半高宽。

## 常见坑

| 坑 | 说明 |
|---|---|
| `filtfilt` vs `lfilter` | `filtfilt` 零相位但不能实时使用；`lfilter` 有相位延迟但支持在线处理 |
| 滤波器截止频率单位 | 指定 `fs` 时 `Wn` 单位是 Hz；不指定时 `Wn` 是归一化频率（0~1，1 对应奈奎斯特频率） |
| FFT 频谱对称性 | 实信号的 FFT 结果是共轭对称的，只需分析前 N/2 个点 |
| `find_peaks` 只找极大值 | 要找极小值需对信号取负：`find_peaks(-y)` |
| 卷积 mode 选择 | `'full'` 有边界效应，`'same'` 截断可能丢失信息，`'valid'` 最短但无边界问题 |

## 小结

- Butterworth 滤波器通带最大平坦，`filtfilt` 实现零相位滤波。
- `signal.convolve` 计算信号卷积，`mode` 参数控制输出长度和边界处理。
- FFT 将时域信号变换到频域，结合 `find_peaks` 可自动提取频率成分。
- `find_peaks` 通过 `height` 和 `distance` 参数灵活控制峰值检测的灵敏度。
- 信号处理的核心流程：时域观察 → 频域分析 → 滤波/特征提取 → 验证。
