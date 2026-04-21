"""
pipelines/classification/decision_tree.py
决策树分类端到端流水线

运行方式: python -m pipelines.classification.decision_tree
"""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from config import get_model_output_dir
from data_generation import decision_tree_classification_data
from data_visualization import (
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_labeled_2d_scatter,
)
from model_training.classification.decision_tree import train_model
from result_visualization.confusion_matrix import plot_confusion_matrix
from result_visualization.roc_curve import plot_roc_curve
from result_visualization.decision_boundary import plot_decision_boundary
from result_visualization.learning_curve import plot_learning_curve
from result_visualization.feature_importance import plot_feature_importance
from result_visualization.tree_structure import get_tree_rules, plot_tree_structure

MODEL = "decision_tree"


def build_learning_curve_model() -> DecisionTreeClassifier:
    """
    构造与主模型超参数一致的学习曲线模型

    这里单独封装一个函数，而不是在 `plot_learning_curve(...)` 里临时写一行，
    原因是：
    1. 让“主模型”和“学习曲线评估模型”参数保持一致；
    2. 避免后面改训练超参数时漏改评估模型；
    3. 让代码语义更清楚。
    """
    return DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        criterion="gini",
        random_state=42,
    )


def show_data_preview(data, feature_names: list[str]) -> None:
    """
    展示决策树训练前的数据图

    这里不再通过 `pipeline_preview` 这种中间层适配，
    而是直接复用 `data_visualization` 中的单数据集公共函数。
    这样后续其它 pipeline 也可以直接按同样方式接入。
    """
    save_dir = get_model_output_dir(MODEL)

    print("\n开始生成数据展示图...")
    plot_class_distribution(
        data,
        target_col="label",
        save_dir=save_dir,
        title="决策树数据展示：类别分布",
        filename="data_class_distribution.png",
    )
    plot_labeled_2d_scatter(
        data,
        x_col=feature_names[0],
        y_col=feature_names[1],
        label_col="label",
        save_dir=save_dir,
        title="决策树数据展示：原始散点图",
        filename="data_scatter.png",
    )
    plot_correlation_heatmap(
        data,
        columns=feature_names + ["label"],
        save_dir=save_dir,
        title="决策树数据展示：相关性热力图",
        filename="data_correlation.png",
    )
    print("数据展示图生成完成。")


def show_model_evaluation(
    model,
    y_test,
    y_pred,
    y_scores,
    feature_names: list[str],
) -> None:
    """
    在终端展示决策树的最终评估结果

    这里展示的是“评估信息”，不是额外生成报告文件。
    主要目的是让用户在命令行里直接看到这棵树最终学成了什么样子。

    Args:
        model: 已训练的决策树模型
        y_test: 测试集真实标签
        y_pred: 测试集预测标签
        y_scores: 测试集预测概率
        feature_names: 特征名称列表
    """
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, digits=4, zero_division=0)
    auc_score = roc_auc_score(y_test, y_scores, multi_class="ovr")
    importances = sorted(
        zip(feature_names, model.feature_importances_, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    rules_text = get_tree_rules(model, feature_names)

    print()
    print("=" * 60)
    print("决策树模型评估展示")
    print("=" * 60)
    print(f"测试集准确率: {accuracy:.4f}")
    print(f"多分类 AUC(OVR): {auc_score:.4f}")
    print(f"最大深度: {model.get_depth()}")
    print(f"叶子节点数: {model.get_n_leaves()}")
    print(f"划分标准: {model.criterion}")
    print()
    print("混淆矩阵原始计数:")
    print(cm)
    print()
    print("特征重要性:")
    for feature_name, importance in importances:
        print(f"  {feature_name}: {importance:.6f}")
    print()
    print("分类报告:")
    print(report_text)
    print("树规则:")
    print(rules_text)


def run():
    """
    决策树分类完整流水线

    当前流程被刻意拆成“数据展示”和“模型结果展示”两大段：
    1. 先看数据长什么样；
    2. 再看模型学到了什么。
    这样更符合教学和排查问题时的阅读顺序。
    """
    print("=" * 60)
    print("决策树分类流水线")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 第 1 步：读取当前算法对应的原始数据
    # ------------------------------------------------------------------
    # 这里用的是专门为决策树分类准备的 blob 多分类数据。
    # 这份数据本身就是二维的，因此非常适合：
    # 1. 训练树模型；
    # 2. 直接观察原始散点分布；
    # 3. 观察决策树的轴对齐切分边界。
    data = decision_tree_classification_data.copy()
    X = data.drop(columns=["label"])
    y = data["label"]
    feature_names = list(X.columns)

    # ------------------------------------------------------------------
    # 第 2 步：先做“数据展示”，这一步不参与模型训练
    # ------------------------------------------------------------------
    # 这是你刚才特别指出的问题：pipeline 里不能只有结果展示，
    # 还要先把当前算法的数据展示出来。
    # 因此这里额外生成三类图：
    # 1. 类别分布图：看各类别是否均衡；
    # 2. 原始散点图：看二维空间中的类分布；
    # 3. 相关性热力图：做一个数值层面的补充观察。
    show_data_preview(data, feature_names)

    # ------------------------------------------------------------------
    # 第 3 步：划分训练集 / 测试集
    # ------------------------------------------------------------------
    # stratify=y 的目的是让训练集和测试集都尽量保持原始类别比例，
    # 避免因为切分随机性导致某一类样本在测试集中过少。
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ------------------------------------------------------------------
    # 第 4 步：训练主模型
    # ------------------------------------------------------------------
    # 这里训练出来的 model 是整条流水线真正的“最终决策树模型”。
    # 后面的特征重要性、结构图、规则展示、决策边界，都应该尽量围绕它展开。
    model = train_model(X_train.values, y_train.values)

    # ------------------------------------------------------------------
    # 第 5 步：在测试集上做预测
    # ------------------------------------------------------------------
    # y_pred 用于离散类别评估；
    # y_scores 用于 ROC / AUC 这类基于概率的评估。
    y_pred = model.predict(X_test.values)
    y_scores = model.predict_proba(X_test.values)

    # ------------------------------------------------------------------
    # 第 6 步：生成模型评估图
    # ------------------------------------------------------------------
    # 这些图回答的是不同层面的问题：
    # 1. 混淆矩阵：各类别是否被分错；
    # 2. ROC 曲线：类别区分能力；
    # 3. 特征重要性：模型最依赖哪些特征；
    # 4. 树结构图：这棵树具体如何分裂；
    # 5. 决策边界：平面空间里具体怎么切分；
    # 6. 学习曲线：数据量变化时模型表现是否稳定。
    plot_confusion_matrix(y_test, y_pred, title="决策树 混淆矩阵", model_name=MODEL)

    plot_roc_curve(
        y_test,
        y_scores,
        title="决策树 ROC 曲线",
        model_name=MODEL,
    )

    plot_feature_importance(
        model,
        feature_names=feature_names,
        title="决策树 特征重要性",
        model_name=MODEL,
    )

    plot_tree_structure(
        model,
        feature_names=feature_names,
        class_names=[str(label) for label in sorted(y.unique())],
        title="决策树 结构图",
        model_name=MODEL,
    )

    # 当前决策树数据本身已经是二维特征，因此不需要再做 PCA。
    # 直接用最终主模型画决策边界，才能确保“你看到的边界”
    # 就是“最终模型实际学出来的边界”。
    plot_decision_boundary(
        model,
        X.values,
        y.values,
        feature_names=feature_names,
        title="决策树 决策边界",
        model_name=MODEL,
    )

    plot_learning_curve(
        build_learning_curve_model(),
        X_train.values,
        y_train.values,
        title="决策树 学习曲线",
        model_name=MODEL,
    )

    # ------------------------------------------------------------------
    # 第 7 步：在终端直接展示关键评估结果
    # ------------------------------------------------------------------
    # 这一步不是写额外“报告文件”，而是让命令行执行结束时，
    # 用户可以立即看到：
    # 1. 准确率和 AUC；
    # 2. 分类报告；
    # 3. 混淆矩阵计数；
    # 4. 特征重要性；
    # 5. 决策树规则文本。
    show_model_evaluation(model, y_test, y_pred, y_scores, feature_names)

    print(f"\n{'=' * 60}")
    print("决策树分类流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
