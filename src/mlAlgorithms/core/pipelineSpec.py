"""
流水线规格定义
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .taskTypes import RunnerType, TaskType


@dataclass(frozen=True)
class PipelineSpec:
    """
    描述一个可执行流水线。
    """

    id: str
    taskType: TaskType
    datasetId: str
    runnerType: RunnerType
    trainer: Callable[..., Any]
    preprocessor: str | None
    splitter: str | None
    predictor: str | None
    evaluator: str | None
    analysisProfile: str
    dataPlots: list[str] = field(default_factory=list)
    resultPlots: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    outputKey: str = ""
    optionalDependencies: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
