"""
信号处理基础
对应文档: ../../docs/scipy/08-signal.md
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def demo_filter():
    """演示滤波器"""
    print("=" * 50)
    print("1. 滤波器设计")
    print("=" * 50)

    # 创建带噪声的信号
    np.random.seed(42)
    t = np.linspace(0, 1, 1000)
    clean_signal = np.sin(2 * np.pi * 5 * t)  # 5Hz 正弦波
    noise = 0.5 * np.random.randn(len(t))
    noisy_signal = clean_signal + noise

    print(f"信号长度: {len(t)}")
    print(f"采样率: 1000 Hz")
    print()

    # Butterworth 低通滤波器
    b, a = signal.butter(4, 10, btype="low", fs=1000)
    filtered = signal.filtfilt(b, a, noisy_signal)

    print("Butterworth 低通滤波器:")
    print(f"  阶数: 4")
    print(f"  截止频率: 10 Hz")
    print(f"  噪声信号标准差: {np.std(noisy_signal):.4f}")
    print(f"  滤波后标准差: {np.std(filtered):.4f}")

    # === 可视化 ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 时域对比
    ax1 = axes[0, 0]
    ax1.plot(t, noisy_signal, "gray", alpha=0.5, label="带噪信号")
    ax1.plot(t, clean_signal, "b-", lw=2, label="原始信号 (5Hz)")
    ax1.plot(t, filtered, "r-", lw=2, label="滤波后信号")
    ax1.set_xlim(0, 0.5)
    ax1.set_title("时域信号对比")
    ax1.set_xlabel("时间 (s)")
    ax1.set_ylabel("幅值")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 频率响应
    ax2 = axes[0, 1]
    w, h = signal.freqz(b, a, worN=2000, fs=1000)
    ax2.plot(w, 20 * np.log10(np.abs(h)), "b-", lw=2)
    ax2.axvline(10, color="r", linestyle="--", label="截止频率 10Hz")
    ax2.axhline(-3, color="gray", linestyle=":", label="-3dB")
    ax2.set_title("滤波器频率响应")
    ax2.set_xlabel("频率 (Hz)")
    ax2.set_ylabel("增益 (dB)")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-60, 5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 频谱对比
    ax3 = axes[1, 0]
    from scipy import fft

    freqs = fft.fftfreq(len(t), 1 / 1000)
    fft_noisy = np.abs(fft.fft(noisy_signal))
    fft_filtered = np.abs(fft.fft(filtered))

    ax3.plot(
        freqs[: len(freqs) // 2],
        fft_noisy[: len(freqs) // 2],
        "gray",
        alpha=0.5,
        label="带噪信号",
    )
    ax3.plot(
        freqs[: len(freqs) // 2],
        fft_filtered[: len(freqs) // 2],
        "r-",
        lw=2,
        label="滤波后",
    )
    ax3.axvline(5, color="blue", linestyle="--", alpha=0.7, label="信号频率 5Hz")
    ax3.axvline(10, color="green", linestyle=":", alpha=0.7, label="截止频率 10Hz")
    ax3.set_xlim(0, 100)
    ax3.set_title("频谱对比")
    ax3.set_xlabel("频率 (Hz)")
    ax3.set_ylabel("幅度")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 误差
    ax4 = axes[1, 1]
    error = np.abs(filtered - clean_signal)
    ax4.plot(t, error, "purple", lw=1)
    ax4.fill_between(t, error, alpha=0.3, color="purple")
    ax4.set_title(f"滤波误差 (均方误差: {np.mean(error**2):.6f})")
    ax4.set_xlabel("时间 (s)")
    ax4.set_ylabel("绝对误差")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/scipy/08_filter.png", dpi=150, bbox_inches="tight")


def demo_convolution():
    """演示卷积"""
    print("=" * 50)
    print("2. 卷积运算")
    print("=" * 50)

    x = np.array([1, 2, 3, 4, 5])
    h = np.array([1, 0, -1])

    # 卷积
    y_full = signal.convolve(x, h, mode="full")
    y_same = signal.convolve(x, h, mode="same")

    print(f"信号 x: {x}")
    print(f"核 h: {h}")
    print(f"\n卷积结果 (full): {y_full}")
    print(f"卷积结果 (same): {y_same}")

    # === 可视化 ===
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax1 = axes[0]
    ax1.stem(range(len(x)), x, basefmt="k-", linefmt="b-", markerfmt="bo")
    ax1.set_title("输入信号 x")
    ax1.set_xlabel("索引")
    ax1.set_ylabel("值")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.stem(range(len(h)), h, basefmt="k-", linefmt="r-", markerfmt="ro")
    ax2.set_title("卷积核 h = [1, 0, -1]")
    ax2.set_xlabel("索引")
    ax2.set_ylabel("值")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.stem(range(len(y_full)), y_full, basefmt="k-", linefmt="g-", markerfmt="go")
    ax3.set_title("卷积结果 x * h")
    ax3.set_xlabel("索引")
    ax3.set_ylabel("值")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/scipy/08_conv.png", dpi=150, bbox_inches="tight")


def demo_fft():
    """演示傅里叶变换"""
    print("=" * 50)
    print("3. 傅里叶变换")
    print("=" * 50)

    from scipy import fft

    # 创建混合频率信号
    fs = 1000  # 采样率
    t = np.linspace(0, 1, fs)
    freq1, freq2 = 5, 50  # Hz
    sig = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)

    # FFT
    yf = fft.fft(sig)
    xf = fft.fftfreq(len(t), 1 / fs)

    # 找峰值
    magnitude = np.abs(yf[: len(t) // 2])
    peaks, properties = signal.find_peaks(magnitude, height=100)
    peak_freqs = xf[: len(t) // 2][peaks]

    print(f"信号: sin(2π·{freq1}t) + 0.5·sin(2π·{freq2}t)")
    print(f"检测到的频率峰值: {peak_freqs} Hz")

    # === 可视化 ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 时域信号
    ax1 = axes[0, 0]
    ax1.plot(t, sig, "b-", lw=1)
    ax1.set_xlim(0, 0.4)
    ax1.set_title(
        f"时域信号: $sin(2\\pi \\cdot {freq1}t) + 0.5 \\cdot sin(2\\pi \\cdot {freq2}t)$"
    )
    ax1.set_xlabel("时间 (s)")
    ax1.set_ylabel("幅值")
    ax1.grid(True, alpha=0.3)

    # 频谱
    ax2 = axes[0, 1]
    ax2.plot(xf[: len(t) // 2], magnitude, "b-", lw=1)
    ax2.scatter(
        peak_freqs, magnitude[peaks], c="red", s=100, zorder=5, label="频率峰值"
    )
    for freq in peak_freqs:
        ax2.annotate(
            f"{freq:.0f} Hz",
            (freq, magnitude[np.abs(xf[: len(t) // 2] - freq).argmin()]),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=10,
        )
    ax2.set_xlim(0, 100)
    ax2.set_title("频谱 (FFT)")
    ax2.set_xlabel("频率 (Hz)")
    ax2.set_ylabel("幅度")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 分离的频率成分
    ax3 = axes[1, 0]
    ax3.plot(t, np.sin(2 * np.pi * freq1 * t), "b-", lw=2, label=f"{freq1} Hz 成分")
    ax3.plot(
        t, 0.5 * np.sin(2 * np.pi * freq2 * t), "r-", lw=2, label=f"{freq2} Hz 成分"
    )
    ax3.set_xlim(0, 0.4)
    ax3.set_title("频率成分分解")
    ax3.set_xlabel("时间 (s)")
    ax3.set_ylabel("幅值")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 相位谱
    ax4 = axes[1, 1]
    phase = np.angle(yf[: len(t) // 2])
    ax4.plot(xf[: len(t) // 2], phase, "g-", lw=1, alpha=0.7)
    ax4.scatter(peak_freqs, phase[peaks], c="red", s=100, zorder=5)
    ax4.set_xlim(0, 100)
    ax4.set_title("相位谱")
    ax4.set_xlabel("频率 (Hz)")
    ax4.set_ylabel("相位 (弧度)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/scipy/08_fft.png", dpi=150, bbox_inches="tight")


def demo_peak_finding():
    """演示峰值检测"""
    print("=" * 50)
    print("4. 峰值检测")
    print("=" * 50)

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

    # === 可视化 ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(x, y, "b-", lw=2, label="信号")
    ax1.scatter(
        x[peaks], y[peaks], c="red", s=100, zorder=5, label=f"峰值 ({len(peaks)}个)"
    )
    ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="高度阈值 = 0.5")

    for i, peak in enumerate(peaks):
        ax1.annotate(
            f"({x[peak]:.1f}, {y[peak]:.2f})",
            (x[peak], y[peak]),
            textcoords="offset points",
            xytext=(5, 10),
            fontsize=9,
        )

    ax1.set_title("峰值检测")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 峰值属性
    ax2 = axes[1]
    prominences = signal.peak_prominences(y, peaks)[0]
    widths = signal.peak_widths(y, peaks, rel_height=0.5)[0]

    bar_width = 0.35
    x_pos = np.arange(len(peaks))

    ax2.bar(
        x_pos - bar_width / 2,
        properties["peak_heights"],
        bar_width,
        label="峰值高度",
        color="steelblue",
        alpha=0.7,
    )
    ax2.bar(
        x_pos + bar_width / 2,
        prominences,
        bar_width,
        label="突出度",
        color="coral",
        alpha=0.7,
    )

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"峰{i + 1}" for i in range(len(peaks))])
    ax2.set_title("峰值属性")
    ax2.set_ylabel("值")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("outputs/scipy/08_peaks.png", dpi=150, bbox_inches="tight")


def demo_all():
    """运行所有演示"""
    import os

    os.makedirs("outputs/scipy", exist_ok=True)

    demo_filter()
    print()
    demo_convolution()
    print()
    demo_fft()
    print()
    demo_peak_finding()


if __name__ == "__main__":
    demo_all()
