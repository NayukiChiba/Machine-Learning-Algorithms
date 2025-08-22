"""
K 值对比：绘制 K=1..15 的准确率曲线
"""

import os
import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入同级模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from utils.decorate import print_func_info
from config import OUTPUTS_ROOT
from generate_data import generate_data
from preprocess_data import preprocess_data
from train_model import train_model

KNN_OUTPUTS = os.path.join(OUTPUTS_ROOT, "KNN")

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


@print_func_info
def compare_k_values(
    k_min: int = 1,
    k_max: int = 15,
    weights: str = "uniform",
    metric: str = "minkowski",
):
    """
    对比不同 K 的训练/测试准确率

    args:
        k_min: K 起始值
        k_max: K 结束值
        weights: 投票权重
        metric: 距离度量
    """
    os.makedirs(KNN_OUTPUTS, exist_ok=True)

    # 生成数据并预处理
    df = generate_data(n_samples=400, noise=0.25, random_state=42)
    X_train, X_test, y_train, y_test, scaler, X_train_orig, X_test_orig = (
        preprocess_data(df, test_size=0.2, random_state=42)
    )

    ks = list(range(k_min, k_max + 1))
    train_accs = []
    test_accs = []

    for k in ks:
        model = train_model(
            X_train,
            y_train,
            n_neighbors=k,
            weights=weights,
            metric=metric,
        )
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_accs.append(accuracy_score(y_train, y_train_pred))
        test_accs.append(accuracy_score(y_test, y_test_pred))

    # 绘制曲线
    plt.figure(figsize=(8, 5))
    plt.plot(ks, train_accs, marker="o", label="训练集准确率")
    plt.plot(ks, test_accs, marker="s", label="测试集准确率")
    plt.title("K 值对比曲线", fontsize=14, fontweight="bold")
    plt.xlabel("K 值")
    plt.ylabel("Accuracy")
    plt.xticks(ks)
    plt.grid(True, alpha=0.3)
    plt.legend()

    filepath = os.path.join(KNN_OUTPUTS, "06_k_comparison.png")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")


if __name__ == "__main__":
    compare_k_values(k_min=1, k_max=15)
