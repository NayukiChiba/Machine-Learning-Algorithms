"""
Basic 模块

包含:
- Numpy: NumPy 基础教程
- Pandas: Pandas 数据处理教程
- ScikitLearn: Scikit-learn 机器学习教程
- Scipy: SciPy 科学计算教程
- Visualization: 数据可视化教程
"""

import sys
from pathlib import Path

# 将项目根目录添加到搜索路径
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
