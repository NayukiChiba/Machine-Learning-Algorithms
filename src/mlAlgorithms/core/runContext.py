"""
运行上下文定义
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pandas import DataFrame, Series

from .datasetSpec import DatasetSpec
from .pipelineSpec import PipelineSpec


@dataclass
class RunContext:
    """
    描述一次流水线运行所需的共享上下文。
    """

    spec: PipelineSpec
    datasetSpec: DatasetSpec
    data: DataFrame
    features: DataFrame | None
    target: Series | None
    outputDir: Path
    randomState: int
    analysisReport: Any | None = None
    extras: dict[str, Any] = field(default_factory=dict)
