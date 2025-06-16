"""
信号处理基础
对应文档: ../../docs/scipy/08-signal.md
"""

import numpy as np
from scipy import signal


def demo_filter():
    """演示滤波器"""
    print("=" * 50)
    print("1. 滤波器设计")
    print("=" * 50)
    
    # 创建带噪声的信号
    t = np.linspace(0, 1, 1000)
    clean_signal = np.sin(2 * np.pi * 5 * t)  # 5Hz 正弦波
    noise = 0.5 * np.random.randn(len(t))
    noisy_signal = clean_signal + noise
    
    print(f"信号长度: {len(t)}")
    print(f"采样率: 1000 Hz")
    print()
    
    # Butterworth 低通滤波器
    b, a = signal.butter(4, 10, btype='low', fs=1000)
    filtered = signal.filtfilt(b, a, noisy_signal)
    
    print("Butterworth 低通滤波器:")
    print(f"  阶数: 4")
    print(f"  截止频率: 10 Hz")
    print(f"  噪声信号标准差: {np.std(noisy_signal):.4f}")
    print(f"  滤波后标准差: {np.std(filtered):.4f}")


def demo_convolution():
    """演示卷积"""
    print("=" * 50)
    print("2. 卷积运算")
    print("=" * 50)
    
    x = np.array([1, 2, 3, 4, 5])
    h = np.array([1, 0, -1])
    
    # 卷积
    y_full = signal.convolve(x, h, mode='full')
    y_same = signal.convolve(x, h, mode='same')
    
    print(f"信号 x: {x}")
    print(f"核 h: {h}")
    print(f"\n卷积结果 (full): {y_full}")
    print(f"卷积结果 (same): {y_same}")


def demo_fft():
    """演示傅里叶变换"""
    print("=" * 50)
    print("3. 傅里叶变换")
    print("=" * 50)
    
    from scipy import fft
    
    # 创建混合频率信号
    t = np.linspace(0, 1, 1000)
    freq1, freq2 = 5, 50  # Hz
    sig = np.sin(2*np.pi*freq1*t) + 0.5*np.sin(2*np.pi*freq2*t)
    
    # FFT
    yf = fft.fft(sig)
    xf = fft.fftfreq(len(t), 1/1000)
    
    # 找峰值
    magnitude = np.abs(yf[:len(t)//2])
    peaks, _ = signal.find_peaks(magnitude, height=100)
    peak_freqs = xf[:len(t)//2][peaks]
    
    print(f"信号: sin(2π·{freq1}t) + 0.5·sin(2π·{freq2}t)")
    print(f"检测到的频率峰值: {peak_freqs} Hz")


def demo_peak_finding():
    """演示峰值检测"""
    print("=" * 50)
    print("4. 峰值检测")
    print("=" * 50)
    
    # 创建带峰值的信号
    x = np.linspace(0, 4*np.pi, 100)
    y = np.sin(x) + 0.1 * np.random.randn(len(x))
    
    # 找峰值
    peaks, properties = signal.find_peaks(y, height=0.5, distance=10)
    
    print(f"信号点数: {len(x)}")
    print(f"检测到 {len(peaks)} 个峰值")
    print(f"峰值位置: {peaks}")
    print(f"峰值高度: {np.round(properties['peak_heights'], 4)}")


def demo_all():
    """运行所有演示"""
    demo_filter()
    print()
    demo_convolution()
    print()
    demo_fft()
    print()
    demo_peak_finding()


if __name__ == "__main__":
    demo_all()
