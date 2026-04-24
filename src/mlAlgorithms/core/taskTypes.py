"""
任务类型与运行类型定义
"""

from __future__ import annotations

from enum import Enum


class _StrEnum(str, Enum):
    """兼容 Python 3.10 的字符串枚举基类。"""

    def __str__(self) -> str:
        return self.value


class TaskType(_StrEnum):
    """任务类型。"""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY = "dimensionality"
    PROBABILISTIC = "probabilistic"


class DataKind(_StrEnum):
    """数据形态。"""

    TABULAR = "tabular"
    SEQUENCE = "sequence"


class RunnerType(_StrEnum):
    """Runner 类型。"""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY = "dimensionality"
    PROBABILISTIC = "probabilistic"
