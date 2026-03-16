"""
data_generation/regression.py
回归算法数据生成模块
统一管理 4 种回归算法各自适用的数据集生成函数
"""

from dataclasses import dataclass
import numpy as np
from pandas import DataFrame
from sklearn.datasets import (
    make_friedman1,
    fetch_california_housing,
    load_diabetes,
)


@dataclass
class RegressionData:
    """
    回归算法数据生成器
    """

    # --- 共享属性 ---
    n_samples: int = 200
    random_state: int = 42

    # --- LinearRegression 专属参数 ---
    lr_noise: float = 10.0  # 目标变量的高斯噪声标准差

    # --- SVR 专属参数 ---
    svr_noise: float = 1.0  # Friedman1 数据集的噪声强度

    # --- Regularization 专属参数 ---
    reg_add_noise_features: int = 8  # 额外纯噪声特征数量
    reg_add_corr_features: bool = True  # 是否添加相关特征(制造多重共线性)

    def linear_regression(self) -> DataFrame:
        """
        手动合成的线性房价数据
        真实关系:price = 2*面积 + 10*房间数 - 3*房龄 + noise + 50
        特点:线性关系完全透明, 适合展示线性回归的参数估计
        返回列:面积,  房间数,  房龄,  price
        """
        rng = np.random.RandomState(self.random_state)

        area = rng.uniform(low=20, high=80, size=self.n_samples)  # 面积 [20,  80]
        num = rng.uniform(low=1, high=5, size=self.n_samples)  # 房间数 [1,  5]
        age = rng.uniform(low=1, high=20, size=self.n_samples)  # 房龄 [1,  20]

        price = (
            2 * area
            + 10 * num
            - 3 * age
            + rng.normal(loc=0, scale=self.lr_noise, size=self.n_samples)
            + 50
        )

        return DataFrame({"面积": area, "房间数": num, "房龄": age, "price": price})

    def svr(self) -> DataFrame:
        """
        Friedman1 非线性回归数据
        真实关系:y = 10*sin(π*x1*x2) + 20*(x3-0.5)² + 10*x4 + 5*x5 + noise
        特点:高度非线性, 前 5 个特征有效, 后 5 个是纯噪声
               完美体现 SVR(RBF 核)拟合非线性的能力
        返回列:x1~x10,  price
        """
        X, y = make_friedman1(
            n_samples=self.n_samples,
            n_features=10,
            noise=self.svr_noise,
            random_state=self.random_state,
        )
        columns = [f"x{i + 1}" for i in range(X.shape[1])]
        df = DataFrame(X, columns=columns)
        df["price"] = y
        return df

    @staticmethod
    def decision_tree() -> DataFrame:
        """
        加利福尼亚房价真实数据集(California Housing)
        特点:8 个特征(地理位置、房龄、收入等), 20640 条真实数据
               特征间交互复杂, 适合展示决策树的非线性分裂能力
        返回列:MedInc,  HouseAge,  AveRooms,  ... ,  price
        注意:真实数据集, n_samples 参数对此方法无效
        """
        data = fetch_california_housing(as_frame=True)
        df = data.frame.rename(columns={"MedHouseVal": "price"})
        return df

    def regularization(self) -> DataFrame:
        """
        糖尿病数据集(Diabetes)+ 人工多重共线性 + 纯噪声特征
        特点:
            - 原始 10 个医学特征(bmi、bp、s5 等)
            - 添加 3 个高度相关特征(多重共线性)→ 让 Ridge 和 Lasso 产生差异
            - 添加 8 个纯噪声特征 → 让 Lasso 的稀疏性(归零)有用武之地
        适合:对比 LinearRegression / Ridge / Lasso / ElasticNet 四种效果
        返回列:原始特征 + bmi_corr/bp_corr/s5_corr + noise_1~noise_8 + price
        注意:真实数据集, n_samples 参数对此方法无效
        """
        rng = np.random.RandomState(self.random_state)

        data = load_diabetes(as_frame=True)
        df = data.frame.copy().rename(columns={"target": "price"})

        # 添加相关特征(制造多重共线性, Ridge 和 Lasso 处理方式不同)
        if self.reg_add_corr_features:
            df["bmi_corr"] = df["bmi"] * 0.9 + rng.normal(scale=0.02, size=len(df))
            df["bp_corr"] = df["bp"] * 0.9 + rng.normal(scale=0.02, size=len(df))
            df["s5_corr"] = df["s5"] * 0.9 + rng.normal(scale=0.02, size=len(df))

        # 添加纯噪声特征(Lasso 会把这些系数压缩为 0)
        for i in range(self.reg_add_noise_features):
            df[f"noise_{i + 1}"] = rng.normal(size=len(df))

        return df


regression_data = RegressionData()
linear_regression_data = regression_data.linear_regression()
svr_data = regression_data.svr()
decision_tree_regression_data = regression_data.decision_tree()
regularization_data = regression_data.regularization()
