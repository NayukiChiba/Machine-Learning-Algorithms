"""
配置与图像保存测试
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from config import OUTPUTS_ROOT, resolveOutputDir
from mlAlgorithms.visualization.figureSaver import saveFigure


def testResolveOutputDirKeepsExistingStructure():
    """输出目录应仍指向现有结构。"""
    assert resolveOutputDir("knn", create=False) == OUTPUTS_ROOT / "knn"
    assert (
        resolveOutputDir("visualization", create=False)
        == OUTPUTS_ROOT / "visualization"
    )


def testFigureSaverWritesIntoGivenDirectory():
    """图像保存工具应保存到指定目录。"""
    outputDir = OUTPUTS_ROOT / "visualization"
    outputPath = outputDir / "test_demo_figure.png"
    if outputPath.exists():
        outputPath.unlink()
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    outputPath = saveFigure(figure, outputDir, "test_demo_figure.png")
    assert outputPath.exists()
    assert outputPath.name == "test_demo_figure.png"
    outputPath.unlink()
