"""
Scikit-learn 模型选择与调参
对应文档: ../docs/05_model_selection.md

使用方式：
    from code.05_model_selection import *
    demo_cross_validation()
    demo_grid_search()
"""

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def demo_cross_val_score():
    """cross_val_score 基础交叉验证"""
    print("=" * 50)
    print("1. cross_val_score")
    print("=" * 50)

    from sklearn.model_selection import cross_val_score

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    model = make_pipeline(StandardScaler(), SVC())

    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    print(f"各折得分: {scores.round(4)}")
    print(f"平均: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    # === 可视化: 交叉验证分数 ===
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(1, len(scores) + 1)
    bars = ax.bar(x, scores, color="#4ECDC4", edgecolor="black", alpha=0.8)
    ax.axhline(
        scores.mean(),
        color="#FF6B6B",
        linestyle="--",
        lw=2,
        label=f"平均: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})",
    )
    ax.fill_between(
        [0.5, len(scores) + 0.5],
        scores.mean() - scores.std() * 2,
        scores.mean() + scores.std() * 2,
        alpha=0.2,
        color="#FF6B6B",
    )

    for bar, score in zip(bars, scores):
        ax.annotate(
            f"{score:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("折数 (Fold)")
    ax.set_ylabel("准确率")
    ax.set_title("5折交叉验证结果")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in x])
    ax.set_ylim(0.9, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("outputs/sklearn/05_cross_val.png", dpi=150, bbox_inches="tight")
    plt.close()


def demo_cross_validate():
    """cross_validate 详细交叉验证"""
    print("=" * 50)
    print("2. cross_validate (多指标)")
    print("=" * 50)

    from sklearn.model_selection import cross_validate

    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    model = make_pipeline(StandardScaler(), SVC())

    cv_results = cross_validate(
        model, X, y, cv=5, scoring=["accuracy", "f1_macro"], return_train_score=True
    )

    print(f"返回的键: {cv_results.keys()}")
    print(f"测试准确率: {cv_results['test_accuracy'].mean():.4f}")
    print(f"训练准确率: {cv_results['train_accuracy'].mean():.4f}")
    print(f"测试F1: {cv_results['test_f1_macro'].mean():.4f}")


def demo_cv_splitters():
    """各种划分策略"""
    print("=" * 50)
    print("3. 划分策略对比")
    print("=" * 50)

    from sklearn.model_selection import (
        KFold,
        StratifiedKFold,
        TimeSeriesSplit,
        LeaveOneOut,
    )

    X = np.arange(10).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    # KFold
    print("KFold(3, shuffle=True):")
    for i, (tr, te) in enumerate(KFold(3, shuffle=True, random_state=42).split(X)):
        print(f"  Fold {i + 1}: train={tr.tolist()}, test={te.tolist()}")

    # StratifiedKFold - 保持类别比例
    print("\nStratifiedKFold(3):")
    for i, (tr, te) in enumerate(
        StratifiedKFold(3, shuffle=True, random_state=42).split(X, y)
    ):
        print(
            f"  Fold {i + 1}: train类别分布={np.bincount(y[tr])}, test类别分布={np.bincount(y[te])}"
        )

    # TimeSeriesSplit
    print("\nTimeSeriesSplit(3):")
    for i, (tr, te) in enumerate(TimeSeriesSplit(3).split(X)):
        print(f"  Fold {i + 1}: train={tr.tolist()}, test={te.tolist()}")


def demo_grid_search():
    """GridSearchCV 网格搜索"""
    print("=" * 50)
    print("4. GridSearchCV")
    print("=" * 50)

    from sklearn.model_selection import GridSearchCV

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    model = make_pipeline(StandardScaler(), SVC())

    param_grid = {"svc__C": [0.1, 1, 10], "svc__kernel": ["linear", "rbf"]}

    grid = GridSearchCV(
        model, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
    )
    grid.fit(X, y)

    print(f"\n最佳参数: {grid.best_params_}")
    print(f"最佳得分: {grid.best_score_:.4f}")
    print(f"最佳模型: {grid.best_estimator_}")


def demo_random_search():
    """RandomizedSearchCV 随机搜索"""
    print("=" * 50)
    print("5. RandomizedSearchCV")
    print("=" * 50)

    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import loguniform

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    model = make_pipeline(StandardScaler(), SVC())

    param_dist = {
        "svc__C": loguniform(0.01, 100),
        "svc__gamma": loguniform(0.001, 10),
        "svc__kernel": ["rbf", "linear"],
    }

    random_search = RandomizedSearchCV(
        model,
        param_dist,
        n_iter=20,
        cv=5,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
    )
    random_search.fit(X, y)

    print(f"最佳参数: {random_search.best_params_}")
    print(f"最佳得分: {random_search.best_score_:.4f}")


def demo_learning_curve():
    """learning_curve 学习曲线"""
    print("=" * 50)
    print("6. learning_curve 学习曲线")
    print("=" * 50)

    from sklearn.model_selection import learning_curve, StratifiedKFold

    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    model = make_pipeline(StandardScaler(), SVC())

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X,
        y,
        cv=cv,
        train_sizes=np.linspace(0.3, 1.0, 5),
        scoring="accuracy",
        shuffle=True,
        random_state=42,
    )

    print(f"训练集大小: {train_sizes}")
    print(f"训练得分: {train_scores.mean(axis=1).round(3)}")
    print(f"测试得分: {test_scores.mean(axis=1).round(3)}")

    # === 可视化: 学习曲线 ===
    fig, ax = plt.subplots(figsize=(10, 6))

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    ax.plot(train_sizes, train_mean, "o-", color="#4ECDC4", label="训练集", lw=2)
    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
        color="#4ECDC4",
    )
    ax.plot(train_sizes, test_mean, "s-", color="#FF6B6B", label="验证集", lw=2)
    ax.fill_between(
        train_sizes,
        test_mean - test_std,
        test_mean + test_std,
        alpha=0.2,
        color="#FF6B6B",
    )

    ax.set_xlabel("训练样本数")
    ax.set_ylabel("准确率")
    ax.set_title("学习曲线 (Learning Curve)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/sklearn/05_learning_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def demo_validation_curve():
    """validation_curve 验证曲线"""
    print("=" * 50)
    print("7. validation_curve 验证曲线")
    print("=" * 50)

    from sklearn.model_selection import validation_curve

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    param_range = np.logspace(-3, 2, 5)

    train_scores, test_scores = validation_curve(
        make_pipeline(StandardScaler(), SVC()),
        X,
        y,
        param_name="svc__C",
        param_range=param_range,
        cv=5,
        scoring="accuracy",
    )

    print(f"C 值: {param_range}")
    print(f"测试得分: {test_scores.mean(axis=1).round(3)}")

    # === 可视化: 验证曲线 ===
    fig, ax = plt.subplots(figsize=(10, 6))

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    ax.semilogx(param_range, train_mean, "o-", color="#4ECDC4", label="训练集", lw=2)
    ax.fill_between(
        param_range,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
        color="#4ECDC4",
    )
    ax.semilogx(param_range, test_mean, "s-", color="#FF6B6B", label="验证集", lw=2)
    ax.fill_between(
        param_range,
        test_mean - test_std,
        test_mean + test_std,
        alpha=0.2,
        color="#FF6B6B",
    )

    ax.set_xlabel("参数 C")
    ax.set_ylabel("准确率")
    ax.set_title("验证曲线 (Validation Curve) - SVC 参数 C")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/sklearn/05_validation_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def demo_all():
    """运行所有演示"""
    import os

    os.makedirs("outputs/sklearn", exist_ok=True)

    demo_cross_val_score()
    print()
    demo_cross_validate()
    print()
    demo_cv_splitters()
    print()
    demo_grid_search()
    print()
    demo_random_search()
    print()
    demo_learning_curve()
    print()
    demo_validation_curve()


if __name__ == "__main__":
    demo_all()
