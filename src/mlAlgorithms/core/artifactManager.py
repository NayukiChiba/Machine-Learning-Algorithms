"""
运行产物管理
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from config import resolveOutputDir


@dataclass
class ArtifactManager:
    """管理本次运行产生的文件列表。"""

    outputKey: str
    artifacts: list[Path] = field(default_factory=list)

    def resolveOutputDir(self) -> Path:
        """解析输出目录。"""
        return resolveOutputDir(self.outputKey)

    def add(self, path: Path | None) -> None:
        """记录一个产物。"""
        if path is None:
            return
        self.artifacts.append(path)

    def extend(self, paths: list[Path]) -> None:
        """批量记录产物。"""
        for path in paths:
            self.add(path)
