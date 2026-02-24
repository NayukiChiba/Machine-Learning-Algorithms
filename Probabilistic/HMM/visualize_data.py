"""
数据可视化模块
"""

import os
import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from matplotlib import pyplot as plt
from pandas import DataFrame
from utils.decorate import print_func_info
from config import OUTPUTS_ROOT
from generate_data import generate_data

HMM_OUTPUTS = os.path.join(OUTPUTS_ROOT, "HMM")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


@print_func_info
def visualize_data(data: DataFrame):
    """
    数据可视化

    args:
        data(DataFrame): 输入数据
    """
    os.makedirs(HMM_OUTPUTS, exist_ok=True)

    # 1. 观测序列
    plt.figure(figsize=(10, 4))
    plt.plot(data["time"], data["obs"], lw=1)
    plt.title("观测序列", fontsize=14, fontweight="bold")
    plt.xlabel("time")
    plt.ylabel("obs")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(HMM_OUTPUTS, "01_observation_sequence.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")

    # 2. 隐状态序列（真实）
    plt.figure(figsize=(10, 4))
    plt.plot(data["time"], data["state_true"], lw=1, color="coral")
    plt.title("隐状态序列（真实）", fontsize=14, fontweight="bold")
    plt.xlabel("time")
    plt.ylabel("state")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(HMM_OUTPUTS, "02_state_sequence_true.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")


if __name__ == "__main__":
    visualize_data(generate_data())
