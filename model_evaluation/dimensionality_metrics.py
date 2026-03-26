"""
model_evaluation/dimensionality_metrics.py
降维模型评估指标

包含: 解释方差比、累计解释方差、重建误差

使用方式:
    from model_evaluation.dimensionality_metrics import evaluate_dimensionality
"""

import numpy as np


def evaluate_dimensionality(
    model,
    X_original=None,
    X_transformed=None,
    print_report: bool = True,
) -> dict:
    """
    计算降维模型的全套评估指标

    Args:
        model: 训练好的降维模型（需有 explained_variance_ratio_ 属性）
        X_original: 原始数据（用于计算重建误差，可选）
        X_transformed: 降维后数据（用于计算重建误差，可选）
        print_report: 是否打印报告

    Returns:
        dict: 包含所有指标的字典
    """
    metrics = {}

    # 解释方差比
    if hasattr(model, "explained_variance_ratio_"):
        evr = model.explained_variance_ratio_
        metrics["explained_variance_ratio"] = evr
        metrics["cumulative_variance_ratio"] = np.cumsum(evr)
        metrics["total_explained_variance"] = np.sum(evr)
        metrics["n_components"] = len(evr)

    # 重建误差
    if (
        X_original is not None
        and X_transformed is not None
        and hasattr(model, "inverse_transform")
    ):
        X_reconstructed = model.inverse_transform(X_transformed)
        reconstruction_error = np.mean((X_original - X_reconstructed) ** 2)
        metrics["reconstruction_error"] = reconstruction_error

    if print_report:
        print("=" * 60)
        print("降维评估报告")
        print("=" * 60)
        if "n_components" in metrics:
            print(f"  主成分数:           {metrics['n_components']}")
            print(f"  总解释方差比:       {metrics['total_explained_variance']:.4f}")
            print("  各成分解释方差比:")
            for i, v in enumerate(metrics["explained_variance_ratio"]):
                cum = metrics["cumulative_variance_ratio"][i]
                print(f"    PC{i + 1}: {v:.4f}  (累计: {cum:.4f})")
        if "reconstruction_error" in metrics:
            print(f"  重建误差 (MSE):     {metrics['reconstruction_error']:.6f}")

    return metrics


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X, _ = make_classification(n_samples=300, n_features=10, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    evaluate_dimensionality(pca, X_original=X_scaled, X_transformed=X_pca)
