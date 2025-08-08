"""
SVR 完整流程：
数据 -> 探索 -> 可视化 -> 预处理 -> 训练 -> 评估 -> 结果可视化
"""

from pathlib import Path
import sys

# 加入项目根目录，便于导入公共模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from generate_data import generate_data
from explore_data import explore_data
from visualize_data import visualize_data
from preprocess_data import preprocess_data
from train_model import train_model
from evaluate_model import evaluate_model
from visualize_results import visualize_results


def main():
    print("\n" + "=" * 70)
    print("支持向量回归（SVR）完整流程")
    print("=" * 70)

    # 1) 数据生成
    print("\n[1/7] 生成数据")
    df = generate_data(n_samples=400, noise=1.0, random_state=42)

    # 2) 数据探索
    print("\n[2/7] 数据探索")
    explore_data(df)

    # 3) 数据可视化
    print("\n[3/7] 数据可视化")
    visualize_data(df)

    # 4) 预处理（划分 + 标准化）
    print("\n[4/7] 数据预处理")
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test = (
        preprocess_data(df, test_size=0.2, random_state=42)
    )

    # 5) 训练 + 6) 评估
    print("\n[5/7] 模型训练 & [6/7] 模型评估")
    model_configs = {
        "SVR-Linear": {"kernel": "linear", "C": 10.0, "epsilon": 0.1},
        "SVR-RBF": {"kernel": "rbf", "C": 10.0, "epsilon": 0.1, "gamma": "scale"},
        "SVR-Poly": {
            "kernel": "poly",
            "C": 10.0,
            "epsilon": 0.1,
            "gamma": "scale",
            "degree": 3,
            "coef0": 1.0,
        },
    }

    results = []
    results_dict = {}

    for name, cfg in model_configs.items():
        print("\n" + "-" * 60)
        print(f"模型：{name}")
        model = train_model(X_train_scaled, y_train, **cfg)

        metrics, y_train_pred, y_test_pred = evaluate_model(
            model_name=name,
            model=model,
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
        )

        results.append(
            {
                "模型": name,
                "训练R2": metrics["train_r2"],
                "测试R2": metrics["test_r2"],
                "训练RMSE": metrics["train_rmse"],
                "测试RMSE": metrics["test_rmse"],
                "训练MAE": metrics["train_mae"],
                "测试MAE": metrics["test_mae"],
            }
        )

        results_dict[name] = {
            "model": model,
            "metrics": metrics,
            "y_train_pred": y_train_pred,
            "y_test_pred": y_test_pred,
            "support_vectors": model.support_.shape[0],
        }

    # 汇总对比
    summary = pd.DataFrame(results).sort_values("测试R2", ascending=False)
    print("\n" + "=" * 70)
    print("模型对比汇总（按测试R2排序）")
    print(summary.round(4).to_string(index=False))

    # 7) 结果可视化
    print("\n[7/7] 结果可视化")
    visualize_results(results_dict, y_test)

    print("\n流程结束")


if __name__ == "__main__":
    main()
