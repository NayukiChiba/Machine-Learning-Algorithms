"""
src 包入口

功能：
1. 统一导出 `mlAlgorithms` 包；
2. 兼容仓库根目录下的 `src` 布局导入方式；
3. 为现有绝对导入提供稳定别名。
"""

from __future__ import annotations

import sys

from . import mlAlgorithms

# 为仓库内现有 `from mlAlgorithms...` 导入提供兼容别名。
sys.modules.setdefault("mlAlgorithms", mlAlgorithms)

__all__ = ["mlAlgorithms"]
