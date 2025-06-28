"""
Scikit-learn 评估指标与可视化
对应文档: ../docs/06_metrics.md

使用方式：
    from code.06_metrics import *
    demo_classification_metrics()
    demo_confusion_matrix()
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def demo_classification_metrics():
    """分类指标基础"""
    print("=" * 50)
    print("1. 分类指标基础")
    print("=" * 50)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    cancer = datasets.load_breast_cancer()
    X, y = cancer.data, cancer.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print(f"准确率 (accuracy): {accuracy_score(y_test, y_pred):.4f}")
    print(f"精确率 (precision): {precision_score(y_test, y_pred):.4f}")
    print(f"召回率 (recall): {recall_score(y_test, y_pred):.4f}")
    print(f"F1分数: {f1_score(y_test, y_pred):.4f}")


def demo_confusion_matrix():
    """混淆矩阵"""
    print("=" * 50)
    print("2. 混淆矩阵")
    print("=" * 50)
    
    from sklearn.metrics import confusion_matrix, classification_report
    
    cancer = datasets.load_breast_cancer()
    X, y = cancer.data, cancer.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"混淆矩阵:\n{cm}")
    print(f"TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"FN={cm[1,0]}, TP={cm[1,1]}")
    
    print(f"\n分类报告:\n{classification_report(y_test, y_pred, target_names=['恶性', '良性'])}")


def demo_roc_auc():
    """ROC AUC"""
    print("=" * 50)
    print("3. ROC AUC")
    print("=" * 50)
    
    from sklearn.metrics import roc_auc_score, roc_curve
    
    cancer = datasets.load_breast_cancer()
    X, y = cancer.data, cancer.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
    print(f"ROC AUC: {auc:.4f}")
    print(f"FPR 范围: [{fpr.min():.2f}, {fpr.max():.2f}]")
    print(f"TPR 范围: [{tpr.min():.2f}, {tpr.max():.2f}]")


def demo_multiclass_metrics():
    """多分类指标"""
    print("=" * 50)
    print("4. 多分类指标 (average 参数)")
    print("=" * 50)
    
    from sklearn.metrics import f1_score, roc_auc_score
    
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    
    print("F1 不同 average:")
    for avg in ['micro', 'macro', 'weighted']:
        f1 = f1_score(y_test, y_pred, average=avg)
        print(f"  {avg}: {f1:.4f}")
    
    print("\n多分类 ROC AUC:")
    for strategy in ['ovr', 'ovo']:
        auc = roc_auc_score(y_test, y_proba, multi_class=strategy)
        print(f"  {strategy}: {auc:.4f}")


def demo_regression_metrics():
    """回归指标"""
    print("=" * 50)
    print("5. 回归指标")
    print("=" * 50)
    
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.linear_model import LinearRegression
    
    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    
    print(f"R²: {r2_score(y_test, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")


def demo_custom_scorer():
    """自定义评分函数"""
    print("=" * 50)
    print("6. 自定义评分函数 (make_scorer)")
    print("=" * 50)
    
    from sklearn.metrics import make_scorer, precision_score, recall_score
    from sklearn.model_selection import cross_val_score
    
    cancer = datasets.load_breast_cancer()
    X, y = cancer.data, cancer.target
    clf = LogisticRegression(max_iter=10000)
    
    # 自定义评分: 0.7*精确率 + 0.3*召回率
    def custom_score(y_true, y_pred):
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        return 0.7 * p + 0.3 * r
    
    custom_scorer = make_scorer(custom_score)
    scores = cross_val_score(clf, X, y, cv=5, scoring=custom_scorer)
    
    print(f"自定义评分各折: {scores.round(4)}")
    print(f"平均: {scores.mean():.4f}")


def demo_display_tools():
    """sklearn 可视化工具"""
    print("=" * 50)
    print("7. sklearn 可视化工具")
    print("=" * 50)
    
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
    
    cancer = datasets.load_breast_cancer()
    X, y = cancer.data, cancer.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train, y_train)
    
    print("可用的 Display 类:")
    print("  - ConfusionMatrixDisplay.from_estimator(model, X, y)")
    print("  - ConfusionMatrixDisplay.from_predictions(y_true, y_pred)")
    print("  - RocCurveDisplay.from_estimator(model, X, y)")
    print("  - RocCurveDisplay.from_predictions(y_true, y_proba)")
    print("  - PrecisionRecallDisplay.from_estimator(model, X, y)")
    print("  - DecisionBoundaryDisplay.from_estimator(model, X)")


def demo_all():
    """运行所有演示"""
    demo_classification_metrics()
    print()
    demo_confusion_matrix()
    print()
    demo_roc_auc()
    print()
    demo_multiclass_metrics()
    print()
    demo_regression_metrics()
    print()
    demo_custom_scorer()
    print()
    demo_display_tools()


if __name__ == "__main__":
    demo_all()
