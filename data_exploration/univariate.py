"""
data_exploration/univariate.py
单变量分析模块

对每个数据集中的每个变量进行独立的统计分析，包括：
  - 连续变量: 集中趋势、离散程度、分布形态、异常值检测
  - 离散变量: 频率分布、类别占比
  - 目标变量: 分布均衡性分析

使用方式:
    from data_exploration import univariate_analysis
    univariate_analysis()

或直接运行:
    python -m data_exploration.univariate
"""

from pandas import DataFrame, Series


# --- 通用工具 ---


def _continuous_stats(series: Series) -> dict[str, float]:
    """
    计算一个连续变量的完整统计指标

    args:
        series: pandas Series, 一列连续数据

    Returns:
        dict: 包含均值、中位数、标准差、偏度、峰度、IQR、异常值数量
    """
    # 四分位数
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    # 四分位距 (Interquartile Range)
    iqr = q3 - q1

    # 异常值边界 (1.5 倍 IQR 法则)
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # 统计异常值数量
    outlier_count = ((series < lower_bound) | (series > upper_bound)).sum()

    return {
        "mean": series.mean(),  # 均值
        "median": series.median(),  # 中位数
        "std": series.std(),  # 标准差
        "min": series.min(),  # 最小值
        "max": series.max(),  # 最大值
        "skew": series.skew(),  # 偏度
        "kurt": series.kurt(),  # 峰度
        "q1": q1,  # 第一四分位数
        "q3": q3,  # 第三四分位数
        "iqr": iqr,  # 四分位距
        "lower_bound": lower_bound,  # 异常值下界
        "upper_bound": upper_bound,  # 异常值上界
        "outliers": outlier_count,  # 异常值个数
    }


def _skew_description(skew_val: float) -> str:
    """
    根据偏度值返回文字描述

    偏度 > 0.5  → 右偏(正偏): 右尾较长，大部分数据集中在左侧
    偏度 < -0.5 → 左偏(负偏): 左尾较长，大部分数据集中在右侧
    其他         → 近似对称
    """
    if skew_val > 0.5:
        return "右偏(正偏)"
    elif skew_val < -0.5:
        return "左偏(负偏)"
    else:
        return "近似对称"


def _kurt_description(kurt_val: float) -> str:
    """
    根据峰度值返回文字描述

    峰度 > 1  → 尖峰(厚尾): 数据集中在均值附近，但极端值更多
    峰度 < -1 → 扁平(薄尾): 数据分布较为均匀
    其他       → 近似正态
    """
    if kurt_val > 1:
        return "尖峰(厚尾)"
    elif kurt_val < -1:
        return "扁平(薄尾)"
    else:
        return "近似正态"


def _print_single_continuous(data: DataFrame, col: str) -> None:
    """
    打印单个连续变量的完整分析

    包括: 范围、集中趋势、离散程度、分布形态、异常值检测

    args:
        data(DataFrame): 数据
        col(str): 列名
    """
    s = _continuous_stats(data[col])

    print(f"  [{col}]")

    # 值域范围
    print(f"最小值: {s['min']:.3f}")
    print(f"最大值: {s['max']:.3f}")

    # 集中趋势: 均值和中位数的差异可以反映分布是否对称
    print(f"均值:   {s['mean']:.3f}")
    print(f"中位数: {s['median']:.3f}")

    # 离散程度: 标准差越大，数据越分散
    print(f"标准差: {s['std']:.3f}")
    print(f"Q1:     {s['q1']:.3f}")
    print(f"Q3:     {s['q3']:.3f}")
    print(f"IQR:    {s['iqr']:.3f}")

    # 分布形态
    print(f"偏度:   {s['skew']:.3f} ({_skew_description(s['skew'])})")
    print(f"峰度:   {s['kurt']:.3f} ({_kurt_description(s['kurt'])})")

    # 异常值检测 (基于 IQR 的 1.5 倍法则)
    if s["outliers"] > 0:
        print(
            f"异常值: {s['outliers']}个 "
            f"(低于 {s['lower_bound']:.3f} 或高于 {s['upper_bound']:.3f})"
        )
    else:
        print("    异常值: 无")

    # 空行分隔不同特征


def _print_discrete_distribution(
    data: DataFrame, col: str, display_name: str = ""
) -> None:
    """
    打印一个离散变量的频率分布

    args:
        data(DataFrame): 数据
        col(str): 列名
        display_name(str): 显示用的名称，默认使用列名
    """
    name = display_name or col
    counts = data[col].value_counts().sort_index()
    total = len(data)

    print(f"  [{name}] 共 {counts.nunique()} 个取值")

    # 逐个打印每个取值的频率和占比
    for val, cnt in counts.items():
        ratio = cnt / total
        print(f"    值 {val}: {cnt} 个 ({ratio * 100:.1f}%)")


# --- 按数据集类型的分析函数 ---


def _analyze_classification(
    data: DataFrame, name: str, target_col: str = "label"
) -> None:
    """
    分类数据集的单变量分析

    对所有连续特征逐一分析，再分析目标变量(离散标签)的分布

    args:
        data(DataFrame): 分类数据集
        name(str): 数据集的描述名称
        target_col(str): 目标变量列名，默认 "label"
    """
    # 获取特征列 (排除目标列)
    feature_cols = [c for c in data.columns if c != target_col]

    # 基本信息
    print(f"数据集: {name}")
    print(f"样本数: {len(data)}")
    print(f"特征数: {len(feature_cols)}")
    print(f"类别数: {data[target_col].nunique()}")
    print(f"特征列: {feature_cols}")

    # 缺失值检查
    missing = data.isnull().sum().sum()
    if missing == 0:
        print("  缺失值: 无")
    else:
        print(f"  缺失值: 共 {missing} 个")
        # 按列打印缺失情况
        for col in data.columns:
            col_missing = data[col].isnull().sum()
            if col_missing > 0:
                print(f"    {col}: {col_missing} 个缺失")

    # 逐个分析每个连续特征
    print("  --- 各特征单变量分析 ---")

    for col in feature_cols:
        _print_single_continuous(data, col)

    # 目标变量分布
    print("  --- 目标变量分布 ---")

    _print_discrete_distribution(data, target_col, "类别 " + target_col)


def _analyze_regression(data: DataFrame, name: str, target_col: str = "price") -> None:
    """
    回归数据集的单变量分析

    对所有连续特征逐一分析，再分析目标变量(连续值)的分布

    args:
        data(DataFrame): 回归数据集
        name(str): 数据集的描述名称
        target_col(str): 目标变量列名，默认 "price"
    """
    # 获取特征列 (排除目标列)
    feature_cols = [c for c in data.columns if c != target_col]

    # 基本信息
    print(f"数据集: {name}")
    print(f"样本数: {len(data)}")
    print(f"特征数: {len(feature_cols)}")
    print(f"特征列: {feature_cols}")

    # 缺失值检查
    missing = data.isnull().sum().sum()
    if missing == 0:
        print("缺失值: 无")
    else:
        print(f"缺失值: 共 {missing} 个")
        for col in data.columns:
            col_missing = data[col].isnull().sum()
            if col_missing > 0:
                print(f"    {col}: {col_missing} 个缺失")

    # 逐个分析每个连续特征
    print("  --- 各特征单变量分析 ---")

    for col in feature_cols:
        _print_single_continuous(data, col)

    # 目标变量也是连续的，同样做单变量分析
    print("  --- 目标变量分析 ---")

    _print_single_continuous(data, target_col)


def _analyze_clustering(
    data: DataFrame, name: str, label_col: str = "true_label"
) -> None:
    """
    聚类数据集的单变量分析

    聚类是无监督算法，true_label 仅用于评估对比，不参与训练

    args:
        data(DataFrame): 聚类数据集
        name(str): 数据集的描述名称
        label_col(str): 真实标签列名，默认 "true_label"
    """
    # 获取特征列 (排除真实标签列)
    feature_cols = [c for c in data.columns if c != label_col]

    # 基本信息
    print(f"数据集: {name}")
    print(f"样本数: {len(data)}")
    print(f"特征数: {len(feature_cols)}")
    print(f"真实簇数: {data[label_col].nunique()}")
    print(f"特征列: {feature_cols}")

    # 缺失值检查
    missing = data.isnull().sum().sum()
    if missing == 0:
        print("  缺失值: 无")
    else:
        print(f"  缺失值: 共 {missing} 个")

    # 逐个分析每个连续特征
    print("  --- 各特征单变量分析 ---")

    for col in feature_cols:
        _print_single_continuous(data, col)

    # 真实簇标签分布 (仅作为评估参考)
    print("  --- 真实簇标签分布 (仅评估用，训练时不可见) ---")

    _print_discrete_distribution(data, label_col, "true_label")


def _analyze_sequence(data: DataFrame, name: str) -> None:
    """
    序列数据集(HMM)的单变量分析

    HMM 数据是时间序列，包含:
      - time: 时间步 (递增整数)
      - obs: 观测符号 (离散)
      - state_true: 真实隐状态 (离散，训练时不可见)

    args:
        data(DataFrame): HMM 序列数据集
        name(str): 数据集的描述名称
    """
    # 基本信息
    print(f"数据集: {name}")
    print(f"序列长度: {len(data)}")
    print(f"字段: {list(data.columns)}")

    # 时间步完整性检查
    print("  --- 时间步检查 ---")

    t = data["time"]
    print(f"时间步范围: {t.min()} ~ {t.max()}")
    # 检查时间步是否连续递增、无间断
    is_continuous = (t.diff().dropna() == 1).all()
    if is_continuous:
        print("步长一致性: 连续无间断")
    else:
        print("步长一致性: 存在间断，请检查数据")

    # 观测符号分布
    print("  --- 观测符号分布 ---")

    _print_discrete_distribution(data, "obs", "观测符号 obs")

    # 隐状态分布
    print("  --- 隐状态分布 (训练时不可见) ---")

    _print_discrete_distribution(data, "state_true", "隐状态 state_true")


# --- 主入口: 分析所有数据集 ---


def univariate_analysis() -> None:
    """
    对所有 20 个数据集执行单变量分析

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
    print("  分类算法数据集 (6 个)")
    print("=" * 50)

    # 1. 逻辑回归: 线性可分的高维二分类数据
    #    6个特征 (3有效 + 1冗余 + 2噪声), 2个类别
    _analyze_classification(
        logistic_regression_data,
        "LogisticRegression — 线性可分高维二分类",
    )

    # 2. 决策树: blob 多分类数据
    #    2个特征, 4个类别, 分布在不同象限
    _analyze_classification(
        decision_tree_classification_data,
        "DecisionTree — blob 多分类",
    )

    # 3. SVC: 同心圆二分类数据
    #    2个特征, 2个类别, 线性不可分
    _analyze_classification(
        svc_data,
        "SVC — 同心圆二分类",
    )

    # 4. 朴素贝叶斯: Iris 真实数据集
    #    4个特征 (花萼/花瓣的长宽), 3个类别
    _analyze_classification(
        naive_bayes_data,
        "NaiveBayes — Iris 真实数据集",
    )

    # 5. KNN: 双月牙二分类数据
    #    2个特征, 2个类别, 非线性边界
    _analyze_classification(
        knn_data,
        "KNN — 双月牙二分类",
    )

    # 6. 随机森林: 高维多噪声三分类数据
    #    10个特征 (5有效 + 3冗余 + 2噪声), 3个类别
    _analyze_classification(
        random_forest_data,
        "RandomForest — 高维多噪声三分类",
    )

    # --- 回归算法数据集 (4 个) ---

    print("=" * 50)
    print("  回归算法数据集 (4 个)")
    print("=" * 50)

    # 1. 线性回归: 手动合成的线性房价数据
    #    3个特征 (面积/房间数/房龄), 目标: price
    #    真实关系: price = 2*面积 + 10*房间数 - 3*房龄 + noise + 50
    _analyze_regression(
        linear_regression_data,
        "LinearRegression — 手动合成线性房价",
    )

    # 2. SVR: Friedman1 非线性回归数据
    #    10个特征 (前5个有效, 后5个纯噪声), 目标: price
    _analyze_regression(
        svr_data,
        "SVR — Friedman1 非线性回归",
    )

    # 3. 决策树回归: 加利福尼亚房价真实数据集
    #    8个特征, 20640条数据, 目标: price
    _analyze_regression(
        decision_tree_regression_data,
        "DecisionTree(回归) — California Housing",
    )

    # 4. 正则化: 糖尿病数据集 + 人工共线性 + 纯噪声特征
    #    10个原始特征 + 3个相关特征 + 8个噪声特征, 目标: price
    #    适合对比 Ridge / Lasso / ElasticNet
    _analyze_regression(
        regularization_data,
        "Regularization — 糖尿病+共线性+噪声",
    )

    # --- 聚类算法数据集 (2 个) ---

    print("=" * 50)
    print("  聚类算法数据集 (2 个)")
    print("=" * 50)

    # 1. KMeans: 球形多簇数据
    #    2个特征, 4个簇, 各向同性高斯分布
    _analyze_clustering(
        kmeans_data,
        "KMeans — 球形多簇",
    )

    # 2. DBSCAN: 双月牙非线性数据
    #    2个特征, 2个簇, 线性不可分
    _analyze_clustering(
        dbscan_data,
        "DBSCAN — 双月牙非线性",
    )

    # --- 集成学习数据集 (4 个) ---

    print("=" * 50)
    print("  集成学习数据集 (4 个)")
    print("=" * 50)

    # 1. Bagging: 高噪声双月牙二分类数据
    #    2个特征, 2个类别, noise=0.35 (较高)
    #    高噪声使单棵树过拟合，Bagging 通过并行平均降低方差
    _analyze_classification(
        bagging_data,
        "Bagging — 高噪声双月牙二分类",
    )

    # 2. GBDT: 多类别中等难度分类数据
    #    8个特征 (4有效 + 2冗余 + 2噪声), 3个类别
    _analyze_classification(
        gbdt_data,
        "GBDT — 多类别中等难度分类",
    )

    # 3. XGBoost: 加利福尼亚房价真实数据集 (回归任务)
    #    与决策树回归使用相同数据集，体现 XGBoost 正则化优势
    _analyze_regression(
        xgboost_data,
        "XGBoost — California Housing 回归",
    )

    # 4. LightGBM: 高维四分类数据
    #    20个特征 (8有效 + 5冗余 + 7噪声), 4个类别
    _analyze_classification(
        lightgbm_data,
        "LightGBM — 高维四分类",
    )

    # --- 降维算法数据集 (2 个) ---

    print("=" * 50)
    print("  降维算法数据集 (2 个)")
    print("=" * 50)

    # 1. PCA: 高维低秩合成数据
    #    10个特征 (只有3个方向有信息), 适合展示解释方差比
    _analyze_classification(
        pca_data,
        "PCA — 高维低秩合成数据",
    )

    # 2. LDA: Wine 真实数据集
    #    13个化学成分特征, 3个类别
    _analyze_classification(
        lda_data,
        "LDA — Wine 真实数据集",
    )

    # --- 概率与序列模型数据集 (2 个) ---

    print("=" * 50)
    print("  概率与序列模型数据集 (2 个)")
    print("=" * 50)

    # 1. EM (GMM): 高斯混合模型数据
    #    2个特征, 3个高斯分量, 权重不等
    #    true_label 仅用于评估对比，EM 训练时不使用标签
    _analyze_clustering(
        em_data,
        "EM(GMM) — 高斯混合模型",
    )

    # 2. HMM: 离散隐马尔可夫序列数据
    #    3个隐状态, 3种观测符号, 序列长度 300
    _analyze_sequence(
        hmm_data,
        "HMM — 离散隐马尔可夫序列",
    )

    # --- 分析完成 ---

    print("=" * 50)
    print("  单变量分析完成, 共分析 20 个数据集")
    print("=" * 50)


# --- 直接运行 ---

if __name__ == "__main__":
    univariate_analysis()
