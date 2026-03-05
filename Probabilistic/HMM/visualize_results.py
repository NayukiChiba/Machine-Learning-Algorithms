"""
结果可视化模块
"""

import os
import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from matplotlib import pyplot as plt
from utils.decorate import print_func_info
from config import OUTPUTS_ROOT
from train_model import train_model
from generate_data import generate_data
from preprocess_data import preprocess_data
from evaluate_model import evaluate_model

HMM_OUTPUTS = os.path.join(OUTPUTS_ROOT, "HMM")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


@print_func_info
def visualize_results(data, y_pred):
    """
    可视化 HMM 结果

    args:
        data: 原始数据 DataFrame
        y_pred: 预测隐状态
    """
    os.makedirs(HMM_OUTPUTS, exist_ok=True)

    # 1. 预测隐状态序列
    plt.figure(figsize=(10, 4))
    plt.plot(data["time"], y_pred, lw=1, color="seagreen")
    plt.title("隐状态序列（预测）", fontsize=14, fontweight="bold")
    plt.xlabel("time")
    plt.ylabel("state")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(HMM_OUTPUTS, "03_state_sequence_pred.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")

    # 2. 真实 vs 预测 对比图
    plt.figure(figsize=(10, 4))
    plt.plot(data["time"], data["state_true"], lw=1, label="真实", color="coral")
    plt.plot(data["time"], y_pred, lw=1, label="预测", color="seagreen")
    plt.title("隐状态对比（真实 vs 预测）", fontsize=14, fontweight="bold")
    plt.xlabel("time")
    plt.ylabel("state")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(HMM_OUTPUTS, "04_state_compare.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")


if __name__ == "__main__":
    data = generate_data()
    X_obs, lengths, y_true, n_symbols = preprocess_data(data)
    model = train_model(X_obs, lengths)
    y_pred = evaluate_model(model, X_obs, lengths, y_true)
    visualize_results(data, y_pred)
