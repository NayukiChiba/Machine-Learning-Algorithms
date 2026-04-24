"""
图像保存工具
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def saveFigure(fig: plt.Figure, outputDir: Path, filename: str, dpi: int = 180) -> Path:
    """保存图像并返回路径。"""
    outputDir.mkdir(parents=True, exist_ok=True)
    path = outputDir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"图像已保存: {path}")
    return path
