"""
生成用于 LDA 降维的示例数据
"""

import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pandas import DataFrame
from sklearn.datasets import load_iris
from utils.decorate import print_func_info


@print_func_info
def generate_data() -> DataFrame:
    """
    加载 Iris 数据集作为 LDA 示例

    returns:
        DataFrame: 包含特征与标签 label 的数据表
    """
    data = load_iris(as_frame=True)
    df = data.frame.copy()
    df = df.rename(columns={"target": "label"})
    return df


if __name__ == "__main__":
    print(generate_data().head())
