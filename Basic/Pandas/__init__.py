"""Pandas 数据处理教程模块"""

from Basic import get_output_dir as resolve_output_dir
from Basic import get_output_path as resolve_output_path


def output_dir():
    """获取 Pandas 模块的输出目录"""
    return resolve_output_dir("pandas")


def output_path(filename: str):
    """获取 Pandas 模块下文件的完整路径"""
    return resolve_output_path("pandas", filename)


__all__ = ["output_dir", "output_path"]
