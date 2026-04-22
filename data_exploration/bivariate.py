"""
data_exploration/bivariate.py
双变量分析模块

对每个数据集中的变量进行两两关系分析，包括:
  - 连续-连续: 皮尔逊相关系数、斯皮尔曼相关系数
  - 连续-目标(分类): 各类别下的均值差异
  - 连续-目标(回归): 特征与目标的相关性排序
  - 特征间相关性矩阵

使用方式:
    from data_exploration.bivariate import bivariate_analysis
    bivariate_analysis()

或直接运行:
    python -m data_exploration.bivariate
"""

from pandas import DataFrame
import numpy as np


# --- 通用工具 ---


def _corr_strength(corr_val: float) -> str:
    """
    根据相关系数绝对值返回强度描述

    |r| >= 0.8 → 强相关
    |r| >= 0.5 → 中等相关
    |r| >= 0.3 → 弱相关
    |r| < 0.3  → 极弱/无相关
    """
    abs_val = abs(corr_val)
    if abs_val >= 0.8:
        return "强相关"
    elif abs_val >= 0.5:
        return "中等相关"
    elif abs_val >= 0.3:
        return "弱相关"
    else:
        return "极弱/无相关"


def _print_correlation_matrix(data: DataFrame, columns: list[str]) -> None:
    """
    打印特征间的皮尔逊相关系数矩阵 (只打印上三角，避免重复)

    只列出 |r| >= 0.3 的特征对，按绝对值从大到小排序

    args:
        data(DataFrame): 数据
        columns(list[str]): 需要分析的列名列表
    """
    if len(columns) < 2:
        print("特征数量不足，无法计算相关性矩阵")
        return

    # 计算皮尔逊相关系数矩阵
    corr_matrix = data[columns].corr(method="pearson")

    # 提取上三角的所有特征对 (排除对角线)
    pairs = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col_a = columns[i]
            col_b = columns[j]
            r = corr_matrix.loc[col_a, col_b]
            pairs.append((col_a, col_b, r))

    # 按相关系数绝对值从大到小排序
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    # 打印所有特征对
    print(f"共 {len(pairs)} 个特征对")

    # 先打印显著相关的 (|r| >= 0.3)
    significant = [p for p in pairs if abs(p[2]) >= 0.3]
    weak = [p for p in pairs if abs(p[2]) < 0.3]

    if significant:
        print(f"显著相关 (|r| >= 0.3): {len(significant)} 对")
        for col_a, col_b, r in significant:
            # 判断正/负相关
            direction = "正相关" if r > 0 else "负相关"
            print(
                f"  {col_a} <-> {col_b}: r = {r:.3f} ({direction}, {_corr_strength(r)})"
            )
    else:
        print("无显著相关的特征对 (所有 |r| < 0.3)")

    if weak:
        print(f"弱相关/无相关 (|r| < 0.3): {len(weak)} 对 (省略详情)")


def _print_feature_target_corr(
    data: DataFrame, feature_cols: list[str], target_col: str
) -> None:
    """
    打印每个特征与目标变量的相关系数 (回归任务)

    同时计算皮尔逊和斯皮尔曼相关系数:
      - 皮尔逊: 衡量线性关系
      - 斯皮尔曼: 衡量单调关系 (对非线性更鲁棒)

    args:
        data(DataFrame): 数据
        feature_cols(list[str]): 特征列名列表
        target_col(str): 目标变量列名
    """
    results = []
    for col in feature_cols:
        # 皮尔逊相关系数 (线性关系)
        pearson_r = data[col].corr(data[target_col], method="pearson")
        # 斯皮尔曼相关系数 (单调关系)
        spearman_r = data[col].corr(data[target_col], method="spearman")
        results.append((col, pearson_r, spearman_r))

    # 按皮尔逊相关系数绝对值从大到小排序
    results.sort(key=lambda x: abs(x[1]), reverse=True)

    for col, pearson_r, spearman_r in results:
        print(f"[{col}]")
        print(f"  皮尔逊 r:  {pearson_r:.3f} ({_corr_strength(pearson_r)})")
        print(f"  斯皮尔曼 ρ: {spearman_r:.3f} ({_corr_strength(spearman_r)})")

        # 如果皮尔逊和斯皮尔曼差异较大，说明可能存在非线性关系
        diff = abs(abs(spearman_r) - abs(pearson_r))
        if diff > 0.15:
            print(f"  注意: 皮尔逊与斯皮尔曼差异 {diff:.3f}，可能存在非线性关系")


def _print_class_feature_diff(
    data: DataFrame, feature_cols: list[str], target_col: str
) -> None:
    """
    打印分类任务中各类别下每个特征的均值差异

    通过比较不同类别下同一特征的均值，可以判断该特征的区分能力

    args:
        data(DataFrame): 数据
        feature_cols(list[str]): 特征列名列表
        target_col(str): 目标变量列名 (离散类别)
    """
    # 获取所有类别
    classes = sorted(data[target_col].unique())

    for col in feature_cols:
        print(f"[{col}]")

        # 各类别下的均值和标准差
        for cls in classes:
            subset = data[data[target_col] == cls][col]
            print(
                f"  类别 {cls}: "
                f"均值 = {subset.mean():.3f}, "
                f"标准差 = {subset.std():.3f}, "
                f"样本数 = {len(subset)}"
            )

        # 计算类别间均值的最大差异 (用于判断区分度)
        class_means = [data[data[target_col] == cls][col].mean() for cls in classes]
        max_diff = max(class_means) - min(class_means)
        overall_std = data[col].std()

        # 用均值差异 / 总标准差作为区分度指标
        if overall_std > 0:
            separability = max_diff / overall_std
            if separability > 2.0:
                desc = "强区分"
            elif separability > 1.0:
                desc = "中等区分"
            elif separability > 0.5:
                desc = "弱区分"
            else:
                desc = "几乎无区分"
            print(
                f"  类间最大均值差: {max_diff:.3f} (区分度: {separability:.2f}, {desc})"
            )
        else:
            print(f"  类间最大均值差: {max_diff:.3f}")


def _print_cluster_feature_diff(
    data: DataFrame, feature_cols: list[str], label_col: str
) -> None:
    """
    打印聚类数据集中各真实簇下每个特征的均值差异

    与分类的类别差异分析类似，但这里的标签仅用于评估

    args:
        data(DataFrame): 数据
        feature_cols(list[str]): 特征列名列表
        label_col(str): 真实标签列名
    """
    clusters = sorted(data[label_col].unique())

    for col in feature_cols:
        print(f"[{col}]")
        for cluster in clusters:
            subset = data[data[label_col] == cluster][col]
            print(
                f"  簇 {cluster}: "
                f"均值 = {subset.mean():.3f}, "
                f"标准差 = {subset.std():.3f}, "
                f"样本数 = {len(subset)}"
            )


def _print_sequence_transition(data: DataFrame) -> None:
    """
    打印 HMM 序列数据的转移统计

    分析相邻时间步之间的状态转移和观测转移频率

    args:
        data(DataFrame): HMM 序列数据
    """
    # 隐状态转移频率
    states = data["state_true"].values

    # 统计相邻状态转移次数
    print(f"隐状态转移统计 (共 {len(states) - 1} 次转移)")
    transition_counts = {}
    for i in range(len(states) - 1):
        pair = (states[i], states[i + 1])
        transition_counts[pair] = transition_counts.get(pair, 0) + 1

    # 按起始状态分组打印
    for s_from in sorted(np.unique(states)):
        print(f"  从状态 {s_from}:")
        total_from = sum(v for (f, t), v in transition_counts.items() if f == s_from)
        for s_to in sorted(np.unique(states)):
            cnt = transition_counts.get((s_from, s_to), 0)
            if total_from > 0:
                ratio = cnt / total_from
                print(f"    -> 状态 {s_to}: {cnt} 次 ({ratio * 100:.1f}%)")

    # 观测-隐状态的联合分布
    print("观测符号与隐状态的联合频率")
    obs = data["obs"].values
    for s in sorted(np.unique(states)):
        mask = states == s
        obs_in_state = obs[mask]
        total_in_state = len(obs_in_state)
        print(f"  状态 {s} (共 {total_in_state} 步):")
        for o in sorted(np.unique(obs)):
            cnt = (obs_in_state == o).sum()
            ratio = cnt / total_in_state if total_in_state > 0 else 0
            print(f"    观测 {o}: {cnt} 次 ({ratio * 100:.1f}%)")


def explore_classification_bivariate(
    data: DataFrame,
    dataset_name: str,
    target_col: str = "label",
) -> None:
    """
    对单个分类数据集执行双变量分析

    Args:
        data: 分类数据集
        dataset_name: 数据集名称
        target_col: 标签列名
    """
    feature_cols = [column for column in data.columns if column != target_col]

    print("=" * 60)
    print(f"{dataset_name}：双变量数据探索")
    print("=" * 60)
    print("--- 特征间相关性 ---")
    _print_correlation_matrix(data, feature_cols)
    print("--- 各类别下的特征均值差异 ---")
    _print_class_feature_diff(data, feature_cols, target_col)


def explore_clustering_bivariate(
    data: DataFrame,
    dataset_name: str,
    label_col: str = "true_label",
) -> None:
    """
    对单个聚类数据集执行双变量分析

    Args:
        data: 聚类数据集
        dataset_name: 数据集名称
        label_col: 真实标签列名
    """
    feature_cols = [column for column in data.columns if column != label_col]

    print("=" * 60)
    print(f"{dataset_name}：双变量数据探索")
    print("=" * 60)
    print("--- 特征间相关性 ---")
    _print_correlation_matrix(data, feature_cols)
    print("--- 各真实簇下的特征均值差异 ---")
    _print_cluster_feature_diff(data, feature_cols, label_col)


def explore_regression_bivariate(
    data: DataFrame,
    dataset_name: str,
    target_col: str = "price",
) -> None:
    """
    对单个回归数据集执行双变量分析

    Args:
        data: 回归数据集
        dataset_name: 数据集名称
        target_col: 目标变量列名
    """
    feature_cols = [column for column in data.columns if column != target_col]

    print("=" * 60)
    print(f"{dataset_name}：双变量数据探索")
    print("=" * 60)
    print("--- 特征间相关性 ---")
    _print_correlation_matrix(data, feature_cols)
    print("--- 特征与目标变量的相关性 ---")
    _print_feature_target_corr(data, feature_cols, target_col)


# --- 按数据集类型的分析函数 ---


def _analyze_classification_bivariate(
    data: DataFrame, name: str, target_col: str = "label"
) -> None:
    """
    分类数据集的双变量分析

    包括:
      1. 特征间的相关性矩阵
      2. 各类别下每个特征的均值差异 (区分度分析)

    args:
        data(DataFrame): 分类数据集
        name(str): 数据集的描述名称
        target_col(str): 目标变量列名
    """
    feature_cols = [c for c in data.columns if c != target_col]

    print(f"数据集: {name}")
    print(f"样本数: {len(data)}")
    print(f"特征数: {len(feature_cols)}")
    print(f"类别数: {data[target_col].nunique()}")

    # 特征间相关性
    print("--- 特征间相关性 (皮尔逊) ---")
    _print_correlation_matrix(data, feature_cols)

    # 各类别下的特征均值差异
    print("--- 各类别下特征均值差异 (区分度分析) ---")
    _print_class_feature_diff(data, feature_cols, target_col)


def _analyze_regression_bivariate(
    data: DataFrame, name: str, target_col: str = "price"
) -> None:
    """
    回归数据集的双变量分析

    包括:
      1. 特征间的相关性矩阵 (检测多重共线性)
      2. 各特征与目标变量的相关系数 (皮尔逊 + 斯皮尔曼)

    args:
        data(DataFrame): 回归数据集
        name(str): 数据集的描述名称
        target_col(str): 目标变量列名
    """
    feature_cols = [c for c in data.columns if c != target_col]

    print(f"数据集: {name}")
    print(f"样本数: {len(data)}")
    print(f"特征数: {len(feature_cols)}")

    # 特征间相关性 (检测多重共线性)
    print("--- 特征间相关性 (多重共线性检测) ---")
    _print_correlation_matrix(data, feature_cols)

    # 特征与目标变量的相关性
    print("--- 特征与目标变量的相关性 ---")
    _print_feature_target_corr(data, feature_cols, target_col)


def _analyze_clustering_bivariate(
    data: DataFrame, name: str, label_col: str = "true_label"
) -> None:
    """
    聚类数据集的双变量分析

    包括:
      1. 特征间相关性
      2. 各真实簇下的特征均值差异

    args:
        data(DataFrame): 聚类数据集
        name(str): 数据集的描述名称
        label_col(str): 真实标签列名
    """
    feature_cols = [c for c in data.columns if c != label_col]

    print(f"数据集: {name}")
    print(f"样本数: {len(data)}")
    print(f"特征数: {len(feature_cols)}")
    print(f"真实簇数: {data[label_col].nunique()}")

    # 特征间相关性
    print("--- 特征间相关性 ---")
    _print_correlation_matrix(data, feature_cols)

    # 各簇下的特征差异
    print("--- 各真实簇下特征均值差异 ---")
    _print_cluster_feature_diff(data, feature_cols, label_col)


def _analyze_sequence_bivariate(data: DataFrame, name: str) -> None:
    """
    序列数据集(HMM)的双变量分析

    分析隐状态之间的转移概率，以及观测符号与隐状态的联合分布

    args:
        data(DataFrame): HMM 序列数据集
        name(str): 数据集的描述名称
    """
    print(f"数据集: {name}")
    print(f"序列长度: {len(data)}")

    # 状态转移和观测-状态联合分布
    print("--- 隐状态转移 & 观测-状态联合分布 ---")
    _print_sequence_transition(data)


# --- 主入口: 分析所有数据集 ---


def bivariate_analysis() -> None:
    """
    对所有 20 个数据集执行双变量分析

    按算法类别分组:
      1. 分类算法 (6个): LR / DT / SVC / NB / KNN / RF
      2. 回归算法 (4个): LR / SVR / DT / Regularization
      3. 聚类算法 (2个): KMeans / DBSCAN
      4. 集成学习 (4个): Bagging / GBDT / XGBoost / LightGBM
      5. 降维算法 (2个): PCA / LDA
      6. 概率模型 (2个): EM / HMM
    """
    # 导入所有数据集
    from data_generation import (
        logistic_regression_data,
        decision_tree_classification_data,
        svc_data,
        naive_bayes_data,
        knn_data,
        random_forest_data,
        linear_regression_data,
        svr_data,
        decision_tree_regression_data,
        regularization_data,
        kmeans_data,
        dbscan_data,
        em_data,
        hmm_data,
        bagging_data,
        gbdt_data,
        xgboost_data,
        lightgbm_data,
        pca_data,
        lda_data,
    )

    # --- 分类算法数据集 (6 个) ---

    print("=" * 50)
    print("分类算法数据集 (6 个)")
    print("=" * 50)

    # 1. 逻辑回归: 线性可分的高维二分类数据
    #    6个特征, 检查特征间冗余关系和类别区分度
    _analyze_classification_bivariate(
        logistic_regression_data,
        "LogisticRegression — 线性可分高维二分类",
    )

    # 2. 决策树: blob 多分类数据
    #    2个特征, 4个类别
    _analyze_classification_bivariate(
        decision_tree_classification_data,
        "DecisionTree — blob 多分类",
    )

    # 3. SVC: 同心圆二分类数据
    #    2个特征, 线性不可分 → 皮尔逊相关 可能不显著
    _analyze_classification_bivariate(
        svc_data,
        "SVC — 同心圆二分类",
    )

    # 4. 朴素贝叶斯: Iris 真实数据集
    #    4个特征, 3个类别, 特征间有一定相关性
    _analyze_classification_bivariate(
        naive_bayes_data,
        "NaiveBayes — Iris 真实数据集",
    )

    # 5. KNN: 双月牙二分类数据
    #    2个特征, 非线性边界
    _analyze_classification_bivariate(
        knn_data,
        "KNN — 双月牙二分类",
    )

    # 6. 随机森林: 高维多噪声三分类数据
    #    10个特征, 含冗余特征 → 预期会有较高的特征间相关性
    _analyze_classification_bivariate(
        random_forest_data,
        "RandomForest — 高维多噪声三分类",
    )

    # --- 回归算法数据集 (4 个) ---

    print("=" * 50)
    print("回归算法数据集 (4 个)")
    print("=" * 50)

    # 1. 线性回归: 手动合成的线性房价数据
    #    3个特征, 目标: price, 线性关系透明
    #    预期: 皮尔逊 和 斯皮尔曼 结果接近
    _analyze_regression_bivariate(
        linear_regression_data,
        "LinearRegression — 手动合成线性房价",
    )

    # 2. SVR: Friedman1 非线性回归数据
    #    10个特征, 前5个有效, 后5个纯噪声
    #    预期: 前5个特征与目标的斯皮尔曼 > 皮尔逊 (非线性关系)
    _analyze_regression_bivariate(
        svr_data,
        "SVR — Friedman1 非线性回归",
    )

    # 3. 决策树回归: 加利福尼亚房价真实数据集
    #    8个特征, 特征间可能存在共线性
    _analyze_regression_bivariate(
        decision_tree_regression_data,
        "DecisionTree(回归) — California Housing",
    )

    # 4. 正则化: 糖尿病数据集 + 共线性 + 噪声特征
    #    21个特征, 其中 bmi_corr/bp_corr/s5_corr 与原始特征高度相关
    #    预期: 共线性检测应显示强相关特征对
    _analyze_regression_bivariate(
        regularization_data,
        "Regularization — 糖尿病+共线性+噪声",
    )

    # --- 聚类算法数据集 (2 个) ---

    print("=" * 50)
    print("聚类算法数据集 (2 个)")
    print("=" * 50)

    # 1. KMeans: 球形多簇数据
    #    2个特征, 4个簇
    _analyze_clustering_bivariate(
        kmeans_data,
        "KMeans — 球形多簇",
    )

    # 2. DBSCAN: 双月牙非线性数据
    #    2个特征, 2个簇
    _analyze_clustering_bivariate(
        dbscan_data,
        "DBSCAN — 双月牙非线性",
    )

    # --- 集成学习数据集 (4 个) ---

    print("=" * 50)
    print("集成学习数据集 (4 个)")
    print("=" * 50)

    # 1. Bagging: 高噪声双月牙二分类
    _analyze_classification_bivariate(
        bagging_data,
        "Bagging — 高噪声双月牙二分类",
    )

    # 2. GBDT: 多类别中等难度分类
    _analyze_classification_bivariate(
        gbdt_data,
        "GBDT — 多类别中等难度分类",
    )

    # 3. XGBoost: California Housing 回归
    _analyze_regression_bivariate(
        xgboost_data,
        "XGBoost — California Housing 回归",
    )

    # 4. LightGBM: 高维四分类
    #    20个特征, 特征间可能存在冗余
    _analyze_classification_bivariate(
        lightgbm_data,
        "LightGBM — 高维四分类",
    )

    # --- 降维算法数据集 (2 个) ---

    print("=" * 50)
    print("降维算法数据集 (2 个)")
    print("=" * 50)

    # 1. PCA: 高维低秩合成数据
    #    10个特征, 只有3个方向有信息 → 预期特征间高度相关
    _analyze_classification_bivariate(
        pca_data,
        "PCA — 高维低秩合成数据",
    )

    # 2. LDA: Wine 真实数据集
    #    13个特征, 3个类别
    _analyze_classification_bivariate(
        lda_data,
        "LDA — Wine 真实数据集",
    )

    # --- 概率与序列模型数据集 (2 个) ---

    print("=" * 50)
    print("概率与序列模型数据集 (2 个)")
    print("=" * 50)

    # 1. EM (GMM): 高斯混合模型数据
    #    2个特征, 3个分量
    _analyze_clustering_bivariate(
        em_data,
        "EM(GMM) — 高斯混合模型",
    )

    # 2. HMM: 离散隐马尔可夫序列数据
    #    分析观测与隐状态之间的转移关系
    _analyze_sequence_bivariate(
        hmm_data,
        "HMM — 离散隐马尔可夫序列",
    )

    # --- 分析完成 ---

    print("=" * 50)
    print("双变量分析完成, 共分析 20 个数据集")
    print("=" * 50)


# --- 直接运行 ---

if __name__ == "__main__":
    bivariate_analysis()
