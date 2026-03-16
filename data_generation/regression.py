from dataclasses import dataclass
from pandas import DataFrame
import numpy as np


@dataclass
class RegressionData:
    """
    回归算法的数据生成
    """

    # --- 共享参数 ---
    # 样本数量
    n_samples: int = 100
    # 随机种子
    random_state: int = 42

    # --- LinearRegression 专属参数 ---
    # 目标变量的高斯噪声标准差
    lr_noise: float = 10.0

    def linear_regression(self) -> DataFrame:
        """
        手动合成的线性房价数据
        真实关系: price = 2*面积 + 10*房间数 - 3*房龄 + noise + 50
        特点: 线性关系完全透明，适合展示线性回归的参数估计
        返回列: 面积, 房间数, 房龄, price
        """
        rng = np.random.RandomState(self.random_state)
        area = rng.uniform(low=20, high=80, size=self.n_samples)  # 面积 [20, 80]
        num = rng.uniform(low=1, high=5, size=self.n_samples)  # 房间数 [1, 5]
        age = rng.uniform(low=1, high=20, size=self.n_samples)  # 房龄 [1, 20]
        price = (
            2 * area
            + 10 * num
            - 3 * age
            + rng.normal(loc=0, scale=self.lr_noise, size=self.n_samples)
            + 50
        )
        data = DataFrame({"area": area, "num": num, "age": age, "price": price})
        return data
