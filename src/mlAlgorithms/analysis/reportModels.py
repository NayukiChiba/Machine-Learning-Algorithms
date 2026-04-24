"""
数据探索报告模型
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetOverview:
    """基础概览。"""

    datasetId: str
    rowCount: int
    featureCount: int
    featureColumns: list[str]
    targetColumn: str | None
    missingCount: int
    description: str


@dataclass
class TabularExplorationReport:
    """表格任务探索报告。"""

    reportType: str
    overview: DatasetOverview
    numericSummary: dict[str, dict[str, float]]
    targetSummary: dict[str, Any]
    correlationSummary: dict[str, Any]
    relationSummary: dict[str, Any]
    multivariateSummary: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


@dataclass
class SequenceExplorationReport:
    """序列任务探索报告。"""

    reportType: str
    overview: DatasetOverview
    observationSummary: dict[str, Any]
    stateSummary: dict[str, Any]
    transitionSummary: dict[str, Any]
    durationSummary: dict[str, Any]
    bigramSummary: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
