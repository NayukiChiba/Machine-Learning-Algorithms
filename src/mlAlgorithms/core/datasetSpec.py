"""
数据集规格定义
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pandas import DataFrame

from .taskTypes import DataKind, TaskType


@dataclass(frozen=True)
class DatasetSpec:
    """
    描述一个数据集的加载方式与元信息。
    """

    id: str
    taskType: TaskType
    dataKind: DataKind
    loader: Callable[[], DataFrame]
    targetColumn: str | None
    featureColumns: list[str] | None
    description: str

    def load(self) -> DataFrame:
        """加载一份新的数据副本。"""
        return self.loader()

    def resolveFeatureColumns(self, data: DataFrame) -> list[str]:
        """解析当前数据集的特征列。"""
        if self.featureColumns is not None:
            return list(self.featureColumns)
        if self.targetColumn is None:
            return list(data.columns)
        return [column for column in data.columns if column != self.targetColumn]
