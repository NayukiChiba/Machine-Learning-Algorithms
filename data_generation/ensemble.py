"""
data_generation/ensemble.py
集成学习算法数据生成模块
统一管理 Bagging、GBDT、XGBoost、LightGBM 各自适用的数据集生成函数
"""

from dataclasses import dataclass
from pandas import DataFrame
from sklearn.datasets import (
    make_moons,
    make_classification,
    fetch_california_housing,
)


@dataclass
class EnsembleData:
    """
    集成学习算法数据生成器
    """

    # --- 共享属性 ---
    n_samples: int = 500
    random_state: int = 42

    # --- Bagging 专属参数 ---
    bagging_noise: float = 0.35  # 较高噪声, 用于体现 Bagging 降方差的优势

    # --- GBDT 专属参数 ---
    gbdt_n_classes: int = 3  # 多分类(GBDT 串行拟合残差, 适合复杂边界)
    gbdt_n_informative: int = 4  # 有效特征数
    gbdt_n_redundant: int = 2  # 冗余特征数
    gbdt_class_sep: float = 0.7  # 类别间隔较小, 分类难度中等

    # --- LightGBM 专属参数 ---
    lgbm_n_features: int = 20  # 高维特征(体现 LightGBM 处理大规模数据的速度优势)
    lgbm_n_informative: int = 10  # 有效特征数
    lgbm_n_redundant: int = 4  # 冗余特征数
    lgbm_n_classes: int = 4  # 四分类
    lgbm_class_sep: float = 1.2  # 类别间隔适中，既保留难度，也更适合教学展示
    lgbm_flip_y: float = 0.01  # 少量标签噪声，避免任务被故意做得过难

    def bagging(self) -> DataFrame:
        """
        双月牙二分类数据, 含较高噪声(make_moons)
        特点:高噪声使单棵决策树倾向过拟合, Bagging 通过并行平均有效降低方差
        注意:noise=0.35 比其他算法更高, 噪声差异是有意设计

        Returns:
            DataFrame: 列包含 x1, x2, label
        """
        X, y = make_moons(
            n_samples=self.n_samples,
            noise=self.bagging_noise,
            random_state=self.random_state,
        )
        return DataFrame({"x1": X[:, 0], "x2": X[:, 1], "label": y})

    def gbdt(self) -> DataFrame:
        """
        多类别分类数据, 中等难度(make_classification)
        特点:3 个类别, 类别间隔适中, GBDT 串行拟合残差逐步修正边界,
               适合展示 Boosting 相比单棵树的精度提升
        注意:n_samples 共享属性对此方法有效

        Returns:
            DataFrame: 列包含 x1~x{n_features}, label(0/1/2)
        """
        n_features = self.gbdt_n_informative + self.gbdt_n_redundant + 2
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=n_features,
            n_informative=self.gbdt_n_informative,
            n_redundant=self.gbdt_n_redundant,
            n_repeated=0,
            n_classes=self.gbdt_n_classes,
            n_clusters_per_class=1,
            class_sep=self.gbdt_class_sep,
            random_state=self.random_state,
        )
        columns = [f"x{i + 1}" for i in range(n_features)]
        df = DataFrame(X, columns=columns)
        df["label"] = y
        return df

    @staticmethod
    def xgboost() -> DataFrame:
        """
        加利福尼亚房价真实数据集(California Housing), 回归任务
        特点:8 个特征, 20640 条真实数据, XGBoost 在表格回归任务上
               综合表现最强, 真实数据能更好体现其正则化和剪枝优势
        注意:真实数据集, n_samples 参数对此方法无效

        Returns:
            DataFrame: 列包含 MedInc, HouseAge, AveRooms, ..., price
        """
        data = fetch_california_housing(as_frame=True)
        df = data.frame.rename(columns={"MedHouseVal": "price"})
        return df

    def lightgbm(self) -> DataFrame:
        """
        高维多类别分类数据(make_classification)
        特点:20 个特征(10 有效 + 4 冗余 + 6 噪声), 4 个类别, 每类一个簇
               这份数据依然保留高维多分类特征，但不过度压缩类别间隔，
               更适合教学中展示 LightGBM 在表格数据上的优势
        注意:n_samples 共享属性对此方法有效

        Returns:
            DataFrame: 列包含 x1~x20, label(0/1/2/3)
        """
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.lgbm_n_features,
            n_informative=self.lgbm_n_informative,
            n_redundant=self.lgbm_n_redundant,
            n_repeated=0,
            n_classes=self.lgbm_n_classes,
            n_clusters_per_class=1,
            class_sep=self.lgbm_class_sep,
            flip_y=self.lgbm_flip_y,
            random_state=self.random_state,
        )
        columns = [f"x{i + 1}" for i in range(self.lgbm_n_features)]
        df = DataFrame(X, columns=columns)
        df["label"] = y
        return df


ensemble_data = EnsembleData()
bagging_data = ensemble_data.bagging()
gbdt_data = ensemble_data.gbdt()
xgboost_data = EnsembleData.xgboost()
lightgbm_data = ensemble_data.lightgbm()
