"""
data_exploration/multivariate.py
多变量分析模块

对每个数据集从全局视角分析多个变量之间的联合关系，包括:
  - 相关性矩阵全局统计 (平均相关强度、最强相关特征对)
  - 多重共线性检测 (VIF, 方差膨胀因子)
  - 主成分方差分析 (PCA 前的维度分析，解释方差比)
  - 类别可分性度量 (Fisher 判别比，类间/类内方差比)

使用方式:
    from data_exploration.multivariate import multivariate_analysis
    multivariate_analysis()

或直接运行:
    python -m data_exploration.multivariate
"""

from pandas import DataFrame
import numpy as np


# --- 通用工具 ---


def _print_corr_summary(data: DataFrame, columns: list[str]) -> None:
    """
    打印相关性矩阵的全局统计信息

    从整体角度描述特征间的相关性水平:
      - 平均绝对相关系数
      - 最强正相关和最强负相关的特征对
      - 高度相关特征组 (|r| >= 0.7)

    args:
        data(DataFrame): 数据
        columns(list[str]): 特征列名列表
    """
    if len(columns) < 2:
        print("特征数量不足，无法进行相关性分析")
        return

    corr_matrix = data[columns].corr(method="pearson")

    # 提取上三角所有特征对的相关系数
    pairs = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            r = corr_matrix.iloc[i, j]
            pairs.append((columns[i], columns[j], r))

    if not pairs:
        print("特征对数量不足")
        return

    # 平均绝对相关系数，衡量特征间整体关联强度
    abs_corrs = [abs(r) for _, _, r in pairs]
    avg_abs_corr = np.mean(abs_corrs)
    max_abs_corr = max(abs_corrs)
    min_abs_corr = min(abs_corrs)

    print(f"特征对总数: {len(pairs)}")
    print(f"平均绝对相关系数: {avg_abs_corr:.3f}")
    print(f"最大绝对相关系数: {max_abs_corr:.3f}")
    print(f"最小绝对相关系数: {min_abs_corr:.3f}")

    positive_pairs = [pair for pair in pairs if pair[2] > 0]
    negative_pairs = [pair for pair in pairs if pair[2] < 0]

    if positive_pairs:
        top_pos = max(positive_pairs, key=lambda x: x[2])
        print(f"最强正相关: {top_pos[0]} <-> {top_pos[1]} (r = {top_pos[2]:.3f})")
    else:
        print("最强正相关: 无")

    if negative_pairs:
        top_neg = min(negative_pairs, key=lambda x: x[2])
        print(f"最强负相关: {top_neg[0]} <-> {top_neg[1]} (r = {top_neg[2]:.3f})")
    else:
        print("最强负相关: 无")

    # 高度相关特征组 (|r| >= 0.7)，可能存在冗余
    highly_correlated = [(a, b, r) for a, b, r in pairs if abs(r) >= 0.7]
    if highly_correlated:
        print(f"高度相关特征对 (|r| >= 0.7): {len(highly_correlated)} 对")
        for a, b, r in highly_correlated:
            print(f"  {a} <-> {b}: r = {r:.3f}")
    else:
        print("无高度相关特征对 (所有 |r| < 0.7)")


def _print_vif(data: DataFrame, columns: list[str]) -> None:
    """
    计算并打印各特征的方差膨胀因子 (VIF)

    VIF 衡量多重共线性的严重程度:
      - VIF = 1: 完全无共线性
      - VIF > 5: 中等共线性，需要关注
      - VIF > 10: 严重共线性，强烈建议处理 (正则化/删除特征)

    使用公式: VIF_j = 1 / (1 - R²_j)
    其中 R²_j 是用其他所有特征回归预测第 j 个特征的 R²

    args:
        data(DataFrame): 数据
        columns(list[str]): 特征列名列表
    """
    if len(columns) < 2:
        print("特征数量不足，无法计算 VIF")
        return

    # 用 numpy 计算，避免依赖 statsmodels
    X = data[columns].values

    # 标准化，避免数值问题
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    # 防止标准差为 0 的情况
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std

    vif_values = []
    for j in range(len(columns)):
        # 用其他特征回归预测第 j 个特征
        y = X_norm[:, j]
        # 其他特征作为自变量
        other_idx = [i for i in range(len(columns)) if i != j]
        X_other = X_norm[:, other_idx]

        # 加截距列
        X_with_intercept = np.column_stack([np.ones(len(y)), X_other])

        # 最小二乘法求解
        try:
            beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            y_pred = X_with_intercept @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            vif = 1 / (1 - r_squared) if r_squared < 1 else float("inf")
        except np.linalg.LinAlgError:
            vif = float("inf")

        vif_values.append((columns[j], vif))

    # 按 VIF 从大到小排序
    vif_values.sort(key=lambda x: x[1], reverse=True)

    # 统计各等级的数量
    severe = sum(1 for _, v in vif_values if v > 10)
    moderate = sum(1 for _, v in vif_values if 5 < v <= 10)
    ok = sum(1 for _, v in vif_values if v <= 5)

    print(f"共 {len(vif_values)} 个特征")
    print(f"严重共线性 (VIF > 10): {severe} 个")
    print(f"中等共线性 (VIF 5~10): {moderate} 个")
    print(f"无共线性 (VIF <= 5): {ok} 个")

    # 逐个打印
    for col, vif in vif_values:
        if vif > 10:
            level = "严重"
        elif vif > 5:
            level = "中等"
        else:
            level = "正常"
        print(f"  {col}: VIF = {vif:.2f} ({level})")


def _print_pca_variance(data: DataFrame, columns: list[str]) -> None:
    """
    用 SVD 计算各主成分的解释方差比，判断数据的有效维度

    不实际做 PCA 降维，只分析方差分布:
      - 前 k 个主成分累计解释 >= 90% 方差 → 有效维度为 k
      - 有效维度远小于特征数 → 存在大量冗余，适合降维

    args:
        data(DataFrame): 数据
        columns(list[str]): 特征列名列表
    """
    if len(columns) < 2:
        print("特征数量不足，无法进行主成分分析")
        return

    X = data[columns].values

    # 标准化
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std

    # SVD 分解
    _, singular_values, _ = np.linalg.svd(X_norm, full_matrices=False)

    # 各主成分的解释方差比
    explained_variance = singular_values**2 / (len(X) - 1)
    total_variance = explained_variance.sum()
    explained_ratio = explained_variance / total_variance

    # 累计解释方差比
    cumulative_ratio = np.cumsum(explained_ratio)

    print(f"原始特征维度: {len(columns)}")

    # 找到累计解释 90% 和 95% 方差所需的主成分数
    dims_90 = int(np.searchsorted(cumulative_ratio, 0.90)) + 1
    dims_95 = int(np.searchsorted(cumulative_ratio, 0.95)) + 1
    dims_90 = min(dims_90, len(columns))
    dims_95 = min(dims_95, len(columns))

    print(f"累计解释 90% 方差所需维度: {dims_90}")
    print(f"累计解释 95% 方差所需维度: {dims_95}")

    if dims_90 < len(columns):
        reduction = (1 - dims_90 / len(columns)) * 100
        print(
            f"降维潜力: 可从 {len(columns)} 维降至 {dims_90} 维 (减少 {reduction:.0f}%)"
        )
    else:
        print("降维潜力: 各维度方差分布均匀，降维收益较低")

    # 打印前几个主成分的解释方差比 (最多显示 10 个)
    n_show = min(len(columns), 10)
    print(f"前 {n_show} 个主成分解释方差比:")
    for i in range(n_show):
        print(
            f"  PC{i + 1}: {explained_ratio[i]:.3f} (累计: {cumulative_ratio[i]:.3f})"
        )


def _print_fisher_ratio(
    data: DataFrame, feature_cols: list[str], target_col: str
) -> None:
    """
    计算 Fisher 判别比，衡量各特征的类别可分性

    Fisher 比 = 类间方差 / 类内方差
      - 比值越大，该特征对类别区分越有用
      - 类间方差: 各类别均值与总均值的加权方差
      - 类内方差: 各类别内部方差的加权平均

    args:
        data(DataFrame): 数据
        feature_cols(list[str]): 特征列名列表
        target_col(str): 目标变量列名 (离散类别)
    """
    classes = sorted(data[target_col].unique())
    n_total = len(data)

    results = []
    for col in feature_cols:
        overall_mean = data[col].mean()

        # 类间方差: sum(n_k * (mean_k - overall_mean)^2) / n_total
        between_var = 0.0
        # 类内方差: sum(n_k * var_k) / n_total
        within_var = 0.0

        for cls in classes:
            subset = data[data[target_col] == cls][col]
            n_k = len(subset)
            mean_k = subset.mean()
            var_k = subset.var()

            between_var += n_k * (mean_k - overall_mean) ** 2
            within_var += n_k * var_k

        between_var /= n_total
        within_var /= n_total

        # Fisher 判别比
        if within_var > 0:
            fisher = between_var / within_var
        else:
            fisher = float("inf")

        results.append((col, fisher, between_var, within_var))

    # 按 Fisher 比从大到小排序 (最有区分力的特征排前面)
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"共 {len(results)} 个特征, {len(classes)} 个类别")

    for col, fisher, b_var, w_var in results:
        if fisher > 1.0:
            level = "强区分"
        elif fisher > 0.3:
            level = "中等区分"
        elif fisher > 0.1:
            level = "弱区分"
        else:
            level = "几乎无区分"
        print(f"[{col}]")
        print(f"  Fisher比: {fisher:.3f} ({level})")
        print(f"  类间方差: {b_var:.3f}")
        print(f"  类内方差: {w_var:.3f}")


def explore_classification_multivariate(
    data: DataFrame,
    dataset_name: str,
    target_col: str = "label",
) -> None:
    """
    对单个分类数据集执行多变量分析

    Args:
        data: 分类数据集
        dataset_name: 数据集名称
        target_col: 标签列名
    """
    feature_cols = [column for column in data.columns if column != target_col]

    print("=" * 60)
    print(f"{dataset_name}：多变量数据探索")
    print("=" * 60)
    print("--- 相关性矩阵全局统计 ---")
    _print_corr_summary(data, feature_cols)

    if len(feature_cols) >= 2:
        print("--- 多重共线性检测 (VIF) ---")
        _print_vif(data, feature_cols)

    if len(feature_cols) >= 3:
        print("--- 主成分方差分析 ---")
        _print_pca_variance(data, feature_cols)

    print("--- Fisher 判别比 ---")
    _print_fisher_ratio(data, feature_cols, target_col)


def explore_clustering_multivariate(
    data: DataFrame,
    dataset_name: str,
    label_col: str = "true_label",
) -> None:
    """
    对单个聚类数据集执行多变量分析

    Args:
        data: 聚类数据集
        dataset_name: 数据集名称
        label_col: 真实标签列名
    """
    feature_cols = [column for column in data.columns if column != label_col]

    print("=" * 60)
    print(f"{dataset_name}：多变量数据探索")
    print("=" * 60)
    print("--- 相关性矩阵全局统计 ---")
    _print_corr_summary(data, feature_cols)

    if len(feature_cols) >= 2:
        print("--- 多重共线性检测 (VIF) ---")
        _print_vif(data, feature_cols)

    if len(feature_cols) >= 3:
        print("--- 主成分方差分析 ---")
        _print_pca_variance(data, feature_cols)

    print("--- Fisher 判别比（基于真实簇标签，仅评估用）---")
    _print_fisher_ratio(data, feature_cols, label_col)


def explore_regression_multivariate(
    data: DataFrame,
    dataset_name: str,
    target_col: str = "price",
) -> None:
    """
    对单个回归数据集执行多变量分析

    Args:
        data: 回归数据集
        dataset_name: 数据集名称
        target_col: 目标变量列名
    """
    feature_cols = [column for column in data.columns if column != target_col]

    print("=" * 60)
    print(f"{dataset_name}：多变量数据探索")
    print("=" * 60)
    print("--- 相关性矩阵全局统计 ---")
    _print_corr_summary(data, feature_cols)

    if len(feature_cols) >= 2:
        print("--- 多重共线性检测 (VIF) ---")
        _print_vif(data, feature_cols)

    if len(feature_cols) >= 3:
        print("--- 主成分方差分析 ---")
        _print_pca_variance(data, feature_cols)


# --- 按数据集类型的分析函数 ---


def _analyze_classification_multi(
    data: DataFrame, name: str, target_col: str = "label"
) -> None:
    """
    分类数据集的多变量分析

    包括:
      1. 相关性矩阵全局统计
      2. 多重共线性检测 (VIF)
      3. 主成分方差分析
      4. Fisher 判别比 (类别可分性)

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

    # 相关性全局统计
    print("--- 相关性矩阵全局统计 ---")
    _print_corr_summary(data, feature_cols)

    # 多重共线性 (特征数 >= 2 才有意义)
    if len(feature_cols) >= 2:
        print("--- 多重共线性检测 (VIF) ---")
        _print_vif(data, feature_cols)

    # 主成分方差分析 (特征数 >= 3 才有降维意义)
    if len(feature_cols) >= 3:
        print("--- 主成分方差分析 ---")
        _print_pca_variance(data, feature_cols)

    # Fisher 判别比
    print("--- Fisher 判别比 (类别可分性) ---")
    _print_fisher_ratio(data, feature_cols, target_col)


def _analyze_regression_multi(
    data: DataFrame, name: str, target_col: str = "price"
) -> None:
    """
    回归数据集的多变量分析

    包括:
      1. 相关性矩阵全局统计
      2. 多重共线性检测 (VIF) — 对回归模型尤为重要
      3. 主成分方差分析

    args:
        data(DataFrame): 回归数据集
        name(str): 数据集的描述名称
        target_col(str): 目标变量列名
    """
    feature_cols = [c for c in data.columns if c != target_col]

    print(f"数据集: {name}")
    print(f"样本数: {len(data)}")
    print(f"特征数: {len(feature_cols)}")

    # 相关性全局统计
    print("--- 相关性矩阵全局统计 ---")
    _print_corr_summary(data, feature_cols)

    # 多重共线性 (对回归来说这是最关键的分析)
    if len(feature_cols) >= 2:
        print("--- 多重共线性检测 (VIF) ---")
        _print_vif(data, feature_cols)

    # 主成分方差分析
    if len(feature_cols) >= 3:
        print("--- 主成分方差分析 ---")
        _print_pca_variance(data, feature_cols)


def _analyze_clustering_multi(
    data: DataFrame, name: str, label_col: str = "true_label"
) -> None:
    """
    聚类数据集的多变量分析

    包括:
      1. 相关性矩阵全局统计
      2. 主成分方差分析
      3. Fisher 判别比 (用真实标签评估特征的簇区分能力)

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

    # 相关性全局统计
    print("--- 相关性矩阵全局统计 ---")
    _print_corr_summary(data, feature_cols)

    # 主成分方差分析
    if len(feature_cols) >= 3:
        print("--- 主成分方差分析 ---")
        _print_pca_variance(data, feature_cols)

    # Fisher 判别比 (用真实标签评估)
    print("--- Fisher 判别比 (簇可分性，仅评估用) ---")
    _print_fisher_ratio(data, feature_cols, label_col)


def _analyze_sequence_multi(data: DataFrame, name: str) -> None:
    """
    序列数据集(HMM)的多变量分析

    HMM 是离散序列，多变量分析关注:
      - 状态的持续时间分布 (一个状态连续出现多少步)
      - 观测序列的 n-gram 分布

    args:
        data(DataFrame): HMM 序列数据集
        name(str): 数据集的描述名称
    """
    print(f"数据集: {name}")
    print(f"序列长度: {len(data)}")

    states = data["state_true"].values
    obs = data["obs"].values

    # 状态持续时间分布
    # 连续相同状态的段长度
    print("--- 状态持续时间分布 ---")
    durations = {}
    current_state = states[0]
    current_len = 1
    for i in range(1, len(states)):
        if states[i] == current_state:
            current_len += 1
        else:
            if current_state not in durations:
                durations[current_state] = []
            durations[current_state].append(current_len)
            current_state = states[i]
            current_len = 1
    # 最后一段
    if current_state not in durations:
        durations[current_state] = []
    durations[current_state].append(current_len)

    for s in sorted(durations.keys()):
        d = durations[s]
        print(f"[状态 {s}]")
        print(f"  出现段数: {len(d)}")
        print(f"  平均持续步数: {np.mean(d):.1f}")
        print(f"  最短持续: {min(d)} 步")
        print(f"  最长持续: {max(d)} 步")

    # 观测序列的 bigram (相邻两个观测的组合频率)
    print("--- 观测 bigram 分布 ---")
    bigram_counts = {}
    total_bigrams = len(obs) - 1
    for i in range(total_bigrams):
        pair = (obs[i], obs[i + 1])
        bigram_counts[pair] = bigram_counts.get(pair, 0) + 1

    # 按频率从高到低排列
    sorted_bigrams = sorted(bigram_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"共 {len(sorted_bigrams)} 种 bigram (相邻观测对)")
    for (o1, o2), cnt in sorted_bigrams:
        ratio = cnt / total_bigrams
        print(f"  ({o1}, {o2}): {cnt} 次 ({ratio * 100:.1f}%)")


# --- 主入口: 分析所有数据集 ---


def multivariate_analysis() -> None:
    """
    对所有 20 个数据集执行多变量分析

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

    # 1. 逻辑回归: 6个特征，含冗余特征
    #    预期: VIF 可能偏高，Fisher 比可以揭示真正有用的特征
    _analyze_classification_multi(
        logistic_regression_data,
        "LogisticRegression — 线性可分高维二分类",
    )

    # 2. 决策树: 2个特征, 4个类别
    #    特征少，VIF/PCA 意义不大，主要看 Fisher 比
    _analyze_classification_multi(
        decision_tree_classification_data,
        "DecisionTree — blob 多分类",
    )

    # 3. SVC: 同心圆数据, 2个特征
    #    非线性可分 → Fisher 比可能较低 (线性度量的局限)
    _analyze_classification_multi(
        svc_data,
        "SVC — 同心圆二分类",
    )

    # 4. 朴素贝叶斯: Iris, 4个特征
    #    经典数据集，特征间有一定相关性
    _analyze_classification_multi(
        naive_bayes_data,
        "NaiveBayes — Iris 真实数据集",
    )

    # 5. KNN: 双月牙, 2个特征
    _analyze_classification_multi(
        knn_data,
        "KNN — 双月牙二分类",
    )

    # 6. 随机森林: 10个特征 (5有效 + 3冗余 + 2噪声)
    #    预期: 冗余特征导致 VIF 偏高，PCA 显示可降维
    _analyze_classification_multi(
        random_forest_data,
        "RandomForest — 高维多噪声三分类",
    )

    # --- 回归算法数据集 (4 个) ---

    print("=" * 50)
    print("回归算法数据集 (4 个)")
    print("=" * 50)

    # 1. 线性回归: 3个独立特征
    #    预期: VIF 接近 1 (特征独立生成)
    _analyze_regression_multi(
        linear_regression_data,
        "LinearRegression — 手动合成线性房价",
    )

    # 2. SVR: 10个特征, 前5个有效
    #    预期: PCA 显示前几个主成分解释大部分方差
    _analyze_regression_multi(
        svr_data,
        "SVR — Friedman1 非线性回归",
    )

    # 3. 决策树回归: California Housing, 8个特征
    #    真实数据，特征间可能有复杂关系
    _analyze_regression_multi(
        decision_tree_regression_data,
        "DecisionTree(回归) — California Housing",
    )

    # 4. 正则化: 21个特征，含人工共线性
    #    预期: bmi_corr/bp_corr/s5_corr 的 VIF 极高
    _analyze_regression_multi(
        regularization_data,
        "Regularization — 糖尿病+共线性+噪声",
    )

    # --- 聚类算法数据集 (2 个) ---

    print("=" * 50)
    print("聚类算法数据集 (2 个)")
    print("=" * 50)

    # 1. KMeans: 2个特征, 4个簇
    _analyze_clustering_multi(
        kmeans_data,
        "KMeans — 球形多簇",
    )

    # 2. DBSCAN: 2个特征, 2个簇
    _analyze_clustering_multi(
        dbscan_data,
        "DBSCAN — 双月牙非线性",
    )

    # --- 集成学习数据集 (4 个) ---

    print("=" * 50)
    print("集成学习数据集 (4 个)")
    print("=" * 50)

    # 1. Bagging: 2个特征
    _analyze_classification_multi(
        bagging_data,
        "Bagging — 高噪声双月牙二分类",
    )

    # 2. GBDT: 8个特征
    _analyze_classification_multi(
        gbdt_data,
        "GBDT — 多类别中等难度分类",
    )

    # 3. XGBoost: California Housing 回归
    _analyze_regression_multi(
        xgboost_data,
        "XGBoost — California Housing 回归",
    )

    # 4. LightGBM: 20个特征
    #    预期: 高维数据，PCA 应显示明显降维潜力
    _analyze_classification_multi(
        lightgbm_data,
        "LightGBM — 高维四分类",
    )

    # --- 降维算法数据集 (2 个) ---

    print("=" * 50)
    print("降维算法数据集 (2 个)")
    print("=" * 50)

    # 1. PCA: 10个特征, 只有3个有效方向
    #    预期: PCA 分析显示前3个主成分解释绝大部分方差
    _analyze_classification_multi(
        pca_data,
        "PCA — 高维低秩合成数据",
    )

    # 2. LDA: Wine, 13个特征
    _analyze_classification_multi(
        lda_data,
        "LDA — Wine 真实数据集",
    )

    # --- 概率与序列模型数据集 (2 个) ---

    print("=" * 50)
    print("概率与序列模型数据集 (2 个)")
    print("=" * 50)

    # 1. EM (GMM): 2个特征, 3个分量
    _analyze_clustering_multi(
        em_data,
        "EM(GMM) — 高斯混合模型",
    )

    # 2. HMM: 离散序列
    #    分析状态持续时间和观测 bigram
    _analyze_sequence_multi(
        hmm_data,
        "HMM — 离散隐马尔可夫序列",
    )

    # --- 分析完成 ---

    print("=" * 50)
    print("多变量分析完成, 共分析 20 个数据集")
    print("=" * 50)


# --- 直接运行 ---

if __name__ == "__main__":
    multivariate_analysis()
