"""
生成房价和3个变量之间的数据
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import numpy as np
from pandas import DataFrame
from utils.decorate import print_func_info


@print_func_info
def generate_data(
    n_samples: int = 200, noise: int = 10, random_state: int = 42
) -> DataFrame:
    """
    生成模拟数据

    args:
        n_samples(int): 样本数量
        noise(int): 噪声水平
        random_state(int): 随机种子
    returns:
        DataFrame: 包含特征和目标变量的数据框
    """
    # 设置随机种子为random_state
    np.random.seed(random_state)

    # 生成特征
    # 使用[0, 1]均匀分布的随机变量
    # 房屋面积
    area = np.random.uniform(low=20, high=80, size=n_samples)
    # 房间数量
    num = np.random.uniform(low=1, high=5, size=n_samples)
    # 房龄
    age = np.random.uniform(low=1, high=20, size=n_samples)

    # 生成目标变量
    """
    真实关系: 价格 = 2 * 面积 + 10 * 房间数 - 3 * 房龄 + noise
    """
    price = (
        2 * area
        + 10 * num
        - 3 * age
        + np.random.normal(loc=0, scale=noise, size=n_samples)
        + 50
    )

    # 创建DataFrame
    data = DataFrame({"面积": area, "房间数": num, "房龄": age, "价格": price})

    return data
