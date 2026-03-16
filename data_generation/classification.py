"""
data_generation/classification.py
分类算法的数据生成
"""

from pandas import DataFrame
from sklearn.datasets import (
    make_classification,
    make_blobs,
    make_circles,
    load_iris,
    make_moons,
)
from dataclasses import dataclass


@dataclass
class ClassificationData:
    """
    分类算法数据集

    用法:
        Classificationtion_data = ClassificationData()
        classification_data.make_moons()
    """

    # 共享属性
    n_samples: int = 400
    random_state: int = 42

    # --- 适合LogisticRegression的数据 ---
    # 特征数量
    lr_n_feature: int = 6
    # 有用的特征数量
    lr_n_informative: int = 3
    # 无用的特征数量
    lr_n_redundant: int = 1
    # 重复特征数量
    lr_n_repeated: int = 0
    # 类别数量
    lr_n_classes: int = 2
    # 类别不平衡程度
    lr_weights: list[float] = None
    # 类别之间的距离
    lr_class_sep: float = 1.2
    # 噪声程度
    lr_flip_y: float = 0.03

    # --- 适合DecisionTree的数据 ---
    # 类别数量
    dt_centers: int = 4
    # 类别之间的距离
    dt_cluster_std: float = 1.0

    # --- 适合SVC的数据 ---
    # 噪声程度
    svc_noise: float = 0.1
    # 特征
    svc_factor: float = 0.5

    # --- 适合KNN的数据 ---
    # 噪声程度
    knn_noise: float = 0.1

    # --- 适合RandomForest的数据 ---
    # 特征数量
    rf_n_features: int = 10
    # 有用的特征数量
    rf_n_informative: int = 5
    # 无用的特征数量
    rf_n_redundant: int = 3
    # 类别数量
    rf_n_classes: int = 3
    # 类别之间的距离
    rf_class_sep: float = 0.8
    # 噪声程度
    rf_flip_y: float = 0.05

    # 二分类数据集, 用于LogisticRegression
    def logistic_regression(self) -> DataFrame:
        """
        线性可分的高维二分类数据
        线性可分, 含有少量噪声和冗余特征

        Returns:
            data(DataFrame): 适合LogisticRegression的二分类数据
        """
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.lr_n_feature,
            n_informative=self.lr_n_informative,
            n_redundant=self.lr_n_redundant,
            n_repeated=self.lr_n_repeated,
            n_classes=self.lr_n_classes,
            weights=self.lr_weights,
            class_sep=self.lr_class_sep,
            flip_y=self.lr_flip_y,
            random_state=self.random_state,
        )
        columns = [f"x{i + 1}" for i in range(self.lr_n_feature)]
        data = DataFrame(X, columns=columns)
        data["label"] = y
        return data

    def decision_tree(self) -> DataFrame:
        """
        生成适合DecisionTree的blob数据
        特点: 4个类别, 分布在不同的象限, 与决策树的轴对齐分裂天然契合

        Returns:
            data(DataFrame): 适合DecisionTree的blob数据
        """
        X, y = make_blobs(
            n_samples=self.n_samples,
            centers=self.dt_centers,
            cluster_std=self.dt_cluster_std,
            random_state=self.random_state,
        )
        columns = [f"x{i + 1}" for i in range(2)]
        data = DataFrame(X, columns=columns)
        data["label"] = y
        return data

    def svc(self) -> DataFrame:
        """
        同心圆二分类数据
        适合SVM

        Returns:
            data(DataFrame): 适合SVM的二分类数据
        """
        X, y = make_circles(
            n_samples=self.n_samples,
            noise=self.svc_noise,
            factor=self.svc_factor,
            random_state=self.random_state,
        )
        columns = [f"x{i + 1}" for i in range(2)]
        data = DataFrame(X, columns=columns)
        data["label"] = y
        return data

    @staticmethod
    def naive_bayes() -> DataFrame:
        """
        使用真实数据集(Iris), 无需生成, 直接加载
        特点: 150个样本, 4个特征, 3个类别

        Returns:
            data(DataFrame): 适合朴素贝叶斯的真实数据集
        """
        iris = load_iris()
        data = DataFrame(iris.data, columns=iris.feature_names)
        data["label"] = iris.target
        return data

    def knn(self) -> DataFrame:
        """
        双月牙二分类数据
        特点：非线性边界，体现 KNN 局部感知能力

        Returns:
            data(DataFrame): 适合KNN的二分类数据
        """
        X, y = make_moons(
            n_samples=self.n_samples,
            noise=self.knn_noise,
            random_state=self.random_state,
        )
        columns = [f"x{i + 1}" for i in range(2)]
        data = DataFrame(X, columns=columns)
        data["label"] = y
        return data

    def random_forest(self) -> DataFrame:
        """
        高维、多噪声、多分类数据
        特点: 10 个特征(5 有效 + 3 冗余 + 2 纯噪声), 3 个类别
        体现随机森林相对单棵决策树的抗噪优势

        Returns:
            data(DataFrame): 适合RandomForest的blob数据
        """
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.rf_n_features,
            n_informative=self.rf_n_informative,
            n_redundant=self.rf_n_redundant,
            n_classes=self.rf_n_classes,
            class_sep=self.rf_class_sep,
            flip_y=self.rf_flip_y,
            random_state=self.random_state,
        )
        columns = [f"x{i + 1}" for i in range(self.rf_n_features)]
        data = DataFrame(X, columns=columns)
        data["label"] = y
        return data


classification_data = ClassificationData()
logistic_regression_data = classification_data.logistic_regression()
decision_tree_data = classification_data.decision_tree()
svc_data = classification_data.svc()
naive_bayes_data = classification_data.naive_bayes()
knn_data = classification_data.knn()
random_forest_data = classification_data.random_forest()
