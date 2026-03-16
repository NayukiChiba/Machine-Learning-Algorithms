"""
data_generation/dimensionality.py
降维算法数据生成模块
统一管理 PCA 和 LDA 各自适用的数据集生成函数
"""

from dataclasses import dataclass
import numpy as np
from pandas import DataFrame
from sklearn.datasets import load_wine


@dataclass
class DimensionalityData:
    """
    降维算法数据生成器
    """

    # --- 共享属性 ---
    n_samples: int = 400
    random_state: int = 42

    # --- PCA 专属参数 ---
    pca_n_features: int = 10  # 原始特征维度(降维前)
    pca_n_informative: int = 3  # 真正有信息的维度数(其余为噪声)
    pca_noise_std: float = 0.5  # 噪声标准差

    def pca(self) -> DataFrame:
        """
        高维合成数据(手动构造低秩结构)
        特点:10 个特征但只有 3 个独立方向有信息, 其余为噪声, 适合演示解释方差比
        注意:PCA 是无监督方法, label 不参与训练

        Returns:
            DataFrame: 列包含 x1~x{pca_n_features}, label(由主方向生成, 仅用于可视化着色)
        """
        rng = np.random.RandomState(self.random_state)

        # 构造低秩基础结构(3 个独立主方向)
        base = rng.randn(self.n_samples, self.pca_n_informative)

        # 随机投影矩阵:将低维结构映射到高维空间
        projection = rng.randn(self.pca_n_informative, self.pca_n_features)
        X = base @ projection

        # 叠加各向同性高斯噪声
        X += rng.randn(self.n_samples, self.pca_n_features) * self.pca_noise_std

        # 由主方向生成伪标签(仅用于可视化)
        label = (base[:, 0] > 0).astype(int) + (base[:, 1] > 0).astype(int)

        columns = [f"x{i + 1}" for i in range(self.pca_n_features)]
        df = DataFrame(X, columns=columns)
        df["label"] = label
        return df

    @staticmethod
    def lda() -> DataFrame:
        """
        红酒真实数据集(Wine Dataset)
        特点:13 个化学成分特征, 3 个类别, 类别间差异明显, 适合展示 LDA 判别方向
        注意:真实数据集, n_samples 参数对此方法无效

        Returns:
            DataFrame: 列包含 alcohol, malic_acid, ...(13 个特征), label
        """
        data = load_wine(as_frame=True)
        df = data.frame.copy().rename(columns={"target": "label"})
        return df


dimensionality_data = DimensionalityData()
pca_data = dimensionality_data.pca()
lda_data = DimensionalityData.lda()
