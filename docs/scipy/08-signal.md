# 信号处理基础

> 对应代码: [08_signal.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Scipy/08_signal.py)

## 滤波器设计

```python
from scipy import signal

# Butterworth 低通滤波器
b, a = signal.butter(order, cutoff, btype='low', fs=fs)
filtered = signal.filtfilt(b, a, data)
```

## 卷积

```python
y = signal.convolve(x, h, mode='full')
y = signal.convolve(x, h, mode='same')
```

## 傅里叶变换

```python
from scipy import fft

yf = fft.fft(signal)
xf = fft.fftfreq(n, 1/fs)
```

## 峰值检测

```python
peaks, properties = signal.find_peaks(y, height=0.5, distance=10)
```

## 练习

```bash
python Basic/Scipy/08_signal.py
```
