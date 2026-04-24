"""
运行结果定义
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    """
    描述一次流水线执行后的核心结果。
    """

    model: Any
    predictions: Any = None
    scores: Any = None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Path] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)
