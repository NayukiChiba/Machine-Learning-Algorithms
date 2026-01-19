"""Pandas 数据处理教程模块"""

import sys
from pathlib import Path

# 将项目根目录添加到搜索路径
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 导入配置
from config import get_output_dir, get_output_path

# 便捷函数
def output_dir():
    """获取 Pandas 模块的输出目录"""
    return get_output_dir("pandas")

def output_path(filename: str):
    """获取 Pandas 模块下文件的完整路径"""
    return get_output_path("pandas", filename)
