from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from generate_data import generate_data

# from explore_data import explore_data
from visualize_data import visualize_data
from preprocess_data import preprocess_data
from train_model import train_model
from evaluate_model import evaluate_model
from visualize_results import visualize_results

"""
主函数：执行完整的正则化回归流程
"""


def main():
    print("\n" + "=" * 60)
    print("正则化回归项目：Lasso / Ridge / ElasticNet")
    print("=" * 60)

    # 1. 生成数据
    print("\n正在生成数据...")
    df = generate_data(add_noise_features=8, add_corr_features=True, random_state=42)

    # 2. 数据探索
    # correlation = explore_data(df)

    # 3. 数据可视化
    print("\n" + "=" * 60)
    print("数据可视化")
    print("=" * 60)
    visualize_data(df)

    # 4. 数据预处理
    X_train, X_test, y_train, y_test, scaler, X_train_orig, X_test_orig = (
        preprocess_data(df, test_size=0.2, random_state=42)
    )

    # 5. 模型训练
    models = train_model(
        X_train,
        y_train,
        feature_names=X_train_orig.columns.tolist(),
        alphas={"lasso": 0.15, "ridge": 2.0, "elasticnet": 0.2},
        l1_ratio=0.5,
        random_state=42,
    )

    # 6. 模型评估
    y_train_pred, y_test_pred, metrics = evaluate_model(
        models, X_train, X_test, y_train, y_test
    )

    # 7. 结果可视化
    visualize_results(
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        feature_names=X_train_orig.columns.tolist(),
        models=models,
    )


if __name__ == "__main__":
    main()
