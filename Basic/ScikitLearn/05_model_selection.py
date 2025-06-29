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


def demo_cross_val_score():
    """cross_val_score 基础交叉验证"""
    print("=" * 50)
    print("1. cross_val_score")
    print("=" * 50)
    
    from sklearn.model_selection import cross_val_score
    
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    
    model = make_pipeline(StandardScaler(), SVC())
    
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    print(f"各折得分: {scores.round(4)}")
    print(f"平均: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")


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
        model, X, y, cv=5,
        scoring=['accuracy', 'f1_macro'],
        return_train_score=True
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
    
    from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, LeaveOneOut
    
    X = np.arange(10).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    # KFold
    print("KFold(3, shuffle=True):")
    for i, (tr, te) in enumerate(KFold(3, shuffle=True, random_state=42).split(X)):
        print(f"  Fold {i+1}: train={tr.tolist()}, test={te.tolist()}")
    
    # StratifiedKFold - 保持类别比例
    print("\nStratifiedKFold(3):")
    for i, (tr, te) in enumerate(StratifiedKFold(3, shuffle=True, random_state=42).split(X, y)):
        print(f"  Fold {i+1}: train类别分布={np.bincount(y[tr])}, test类别分布={np.bincount(y[te])}")
    
    # TimeSeriesSplit
    print("\nTimeSeriesSplit(3):")
    for i, (tr, te) in enumerate(TimeSeriesSplit(3).split(X)):
        print(f"  Fold {i+1}: train={tr.tolist()}, test={te.tolist()}")


def demo_grid_search():
    """GridSearchCV 网格搜索"""
    print("=" * 50)
    print("4. GridSearchCV")
    print("=" * 50)
    
    from sklearn.model_selection import GridSearchCV
    
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    
    model = make_pipeline(StandardScaler(), SVC())
    
    param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__kernel': ['linear', 'rbf']
    }
    
    grid = GridSearchCV(
        model, param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
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
        'svc__C': loguniform(0.01, 100),
        'svc__gamma': loguniform(0.001, 10),
        'svc__kernel': ['rbf', 'linear']
    }
    
    random_search = RandomizedSearchCV(
        model, param_dist,
        n_iter=20,
        cv=5,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
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

    # 使用 StratifiedKFold 确保每个 fold 中类别分布均衡
    # 使用 3 折以确保每个训练集有足够样本
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # 训练集比例从 0.3 开始，确保每个 fold 有足够样本覆盖所有类别
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y,
        cv=cv,
        train_sizes=np.linspace(0.3, 1.0, 5),
        scoring='accuracy',
        shuffle=True,
        random_state=42
    )
    
    print(f"训练集大小: {train_sizes}")
    print(f"训练得分: {train_scores.mean(axis=1).round(3)}")
    print(f"测试得分: {test_scores.mean(axis=1).round(3)}")


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
        X, y,
        param_name='svc__C',
        param_range=param_range,
        cv=5,
        scoring='accuracy'
    )
    
    print(f"C 值: {param_range}")
    print(f"测试得分: {test_scores.mean(axis=1).round(3)}")


def demo_all():
    """运行所有演示"""
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
