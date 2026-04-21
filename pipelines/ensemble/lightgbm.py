"""
pipelines/ensemble/lightgbm.py
LightGBM 分类端到端流水线

运行方式: python -m pipelines.ensemble.lightgbm
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_generation import lightgbm_data
from model_training.ensemble.lightgbm import train_model
from result_visualization.confusion_matrix import plot_confusion_matrix
from result_visualization.roc_curve import plot_roc_curve
from result_visualization.feature_importance import plot_feature_importance

MODEL = "lightgbm"


def run():
    """LightGBM 分类完整流水线"""
    print("=" * 60)
    print("LightGBM 分类流水线")
    print("=" * 60)

    data = lightgbm_data.copy()
    X = data.drop(columns=["label"])
    y = data["label"]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = train_model(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    plot_confusion_matrix(
        y_test,
        y_pred,
        title="LightGBM 混淆矩阵",
        model_name=MODEL,
    )

    y_scores = model.predict_proba(X_test_s)
    plot_roc_curve(
        y_test,
        y_scores,
        title="LightGBM ROC 曲线",
        model_name=MODEL,
    )

    plot_feature_importance(
        model,
        feature_names=feature_names,
        title="LightGBM 特征重要性",
        model_name=MODEL,
    )

    print(f"\n{'=' * 60}")
    print("LightGBM 流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
