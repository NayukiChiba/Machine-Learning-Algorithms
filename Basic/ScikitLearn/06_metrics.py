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
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


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
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"准确率 (accuracy): {acc:.4f}")
    print(f"精确率 (precision): {prec:.4f}")
    print(f"召回率 (recall): {rec:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # === 可视化: 指标对比 ===
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['准确率\n(Accuracy)', '精确率\n(Precision)', '召回率\n(Recall)', 'F1分数']
    values = [acc, prec, rec, f1]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    bars = ax.bar(metrics, values, color=colors, edgecolor='black', alpha=0.8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('分数')
    ax.set_title('分类模型评估指标对比 (乳腺癌数据集)')
    ax.axhline(0.9, color='gray', linestyle='--', alpha=0.5, label='0.9 基准线')
    
    for bar, val in zip(bars, values):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('outputs/sklearn/06_classification_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()


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
    
    # === 可视化: 混淆矩阵 ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图: 混淆矩阵热力图
    ax1 = axes[0]
    im = ax1.imshow(cm, cmap='Blues')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['恶性 (0)', '良性 (1)'])
    ax1.set_yticklabels(['恶性 (0)', '良性 (1)'])
    ax1.set_xlabel('预测标签')
    ax1.set_ylabel('真实标签')
    ax1.set_title('混淆矩阵 (Confusion Matrix)')
    
    # 添加数值标注
    labels = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            text = f'{labels[i][j]}\n{cm[i,j]}'
            color = 'white' if cm[i,j] > cm.max()/2 else 'black'
            ax1.text(j, i, text, ha='center', va='center', color=color, fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax1)
    
    # 右图: 指标图解
    ax2 = axes[1]
    ax2.axis('off')
    
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    
    formula_text = f"""
混淆矩阵解读:

  TN (真负例) = {tn}    FP (假正例) = {fp}
  FN (假负例) = {fn}    TP (真正例) = {tp}

指标计算公式:

  Accuracy  = (TP+TN)/(TP+TN+FP+FN) = {accuracy:.3f}
  Precision = TP/(TP+FP) = {precision:.3f}
  Recall    = TP/(TP+FN) = {recall:.3f}
  F1        = 2*P*R/(P+R) = {f1:.3f}
"""
    ax2.text(0.1, 0.5, formula_text, fontsize=12, family='Microsoft YaHei',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('outputs/sklearn/06_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()


def demo_roc_auc():
    """ROC AUC"""
    print("=" * 50)
    print("3. ROC AUC")
    print("=" * 50)
    
    from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
    
    cancer = datasets.load_breast_cancer()
    X, y = cancer.data, cancer.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"FPR 范围: [{fpr.min():.2f}, {fpr.max():.2f}]")
    print(f"TPR 范围: [{tpr.min():.2f}, {tpr.max():.2f}]")
    
    # === 可视化: ROC 和 PR 曲线 ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图: ROC 曲线
    ax1 = axes[0]
    ax1.plot(fpr, tpr, color='#FF6B6B', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='随机猜测')
    ax1.fill_between(fpr, tpr, alpha=0.3, color='#FF6B6B')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.05])
    ax1.set_xlabel('假正例率 (FPR)')
    ax1.set_ylabel('真正例率 (TPR)')
    ax1.set_title('ROC 曲线 (Receiver Operating Characteristic)')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 右图: Precision-Recall 曲线
    ax2 = axes[1]
    ax2.plot(recall, precision, color='#4ECDC4', lw=2, label=f'PR (AUC = {pr_auc:.3f})')
    ax2.fill_between(recall, precision, alpha=0.3, color='#4ECDC4')
    ax2.axhline(y_test.sum()/len(y_test), color='gray', linestyle='--', label='基线 (正类比例)')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.05])
    ax2.set_xlabel('召回率 (Recall)')
    ax2.set_ylabel('精确率 (Precision)')
    ax2.set_title('Precision-Recall 曲线')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/sklearn/06_roc_pr.png', dpi=150, bbox_inches='tight')
    plt.close()


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
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"R^2: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # === 可视化: 回归指标 ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 左图: 预测 vs 真实
    ax1 = axes[0]
    ax1.scatter(y_test, y_pred, c='#45B7D1', alpha=0.6, s=40, edgecolors='white')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='完美预测')
    ax1.set_xlabel('真实值')
    ax1.set_ylabel('预测值')
    ax1.set_title(f'预测 vs 真实 (R^2 = {r2:.3f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 中图: 残差分布
    ax2 = axes[1]
    residuals = y_test - y_pred
    ax2.hist(residuals, bins=20, color='#4ECDC4', edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', lw=2, label='零残差')
    ax2.set_xlabel('残差')
    ax2.set_ylabel('频数')
    ax2.set_title('残差分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 右图: 指标柱状图
    ax3 = axes[2]
    metrics = ['R^2', 'RMSE', 'MAE']
    values = [r2, rmse/100, mae/100]  # 归一化以便比较
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax3.bar(metrics, values, color=colors, edgecolor='black')
    ax3.set_ylabel('分数')
    ax3.set_title('回归指标 (RMSE/MAE 已归一化)')
    for bar, val, orig in zip(bars, values, [r2, rmse, mae]):
        ax3.annotate(f'{orig:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('outputs/sklearn/06_regression_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()


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
    import os
    os.makedirs('outputs/sklearn', exist_ok=True)
    
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

