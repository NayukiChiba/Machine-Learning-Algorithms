from pandas import DataFrame
from sklearn import make_classification
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

    # 适合LogisticRegression的数据
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

    # 二分类数据集, 用于LogisticRegression
    def logistic_regression(self) -> DataFrame:
        """
        线性可分的高维二分类数据
        线性可分, 含有少量噪声和冗余特征

        Returns:
            data(DataFrame): 适合LogisticRegression的二分类数据
        """
        X, y = make_classification(
            n_smaples=self.n_samples,
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


classification_data = ClassificationData()
logistic_regression_data = classification_data.logistic_regression()
