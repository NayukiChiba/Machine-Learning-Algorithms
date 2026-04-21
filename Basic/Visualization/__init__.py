"""数据可视化教程模块"""

from Basic import get_output_dir as resolve_output_dir
from Basic import get_output_path as resolve_output_path


def output_dir():
    """获取 Visualization 模块的输出目录"""
    return resolve_output_dir("visualization")


def output_path(filename: str):
    """获取 Visualization 模块下文件的完整路径"""
    return resolve_output_path("visualization", filename)


__all__ = ["output_dir", "output_path"]
