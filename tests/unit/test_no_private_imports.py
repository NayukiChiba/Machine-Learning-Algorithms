"""
私有导入约束测试
"""

from __future__ import annotations

import re
from pathlib import Path


def testCoreSourceDoesNotImportPrivateFunctions():
    """核心源码不应导入以下划线开头的私有函数。"""
    sourceRoot = Path("src") / "mlAlgorithms"
    patterns = [
        re.compile(r"from\s+[\w\.]+\s+import\s+_[A-Za-z0-9_]+"),
        re.compile(r"import\s+[\w\.]+\._[A-Za-z0-9_]+"),
    ]
    violations: list[str] = []
    for path in sourceRoot.rglob("*.py"):
        content = path.read_text(encoding="utf-8")
        for pattern in patterns:
            if pattern.search(content):
                violations.append(str(path))
                break
    assert violations == []
