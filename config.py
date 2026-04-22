"""
项目全局配置

功能：
1. 统一管理项目根路径、源码路径和输出目录；
2. 提供 Basic 模块仍在使用的输出目录解析函数；
3. 提供新架构使用的统一输出目录解析接口。
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
RANDOM_STATE = 42
OPTIONAL_DEPENDENCIES = {"xgboost", "lightgbm", "hmmlearn"}

NUMPY_OUTPUT_DIR = OUTPUTS_ROOT / "numpy"
PANDAS_OUTPUT_DIR = OUTPUTS_ROOT / "pandas"
SKLEARN_OUTPUT_DIR = OUTPUTS_ROOT / "sklearn"
SCIPY_OUTPUT_DIR = OUTPUTS_ROOT / "scipy"
VISUALIZATION_OUTPUT_DIR = OUTPUTS_ROOT / "visualization"

DATA_VIS_ROOT = OUTPUTS_ROOT / "data_visualization"
DATA_VIS_DISTRIBUTION_DIR = DATA_VIS_ROOT / "distribution"
DATA_VIS_SCATTER_DIR = DATA_VIS_ROOT / "scatter"
DATA_VIS_CORRELATION_DIR = DATA_VIS_ROOT / "correlation"
DATA_VIS_FEATURE_SPACE_DIR = DATA_VIS_ROOT / "feature_space"

OUTPUT_DIR_MAP = {
    "numpy": NUMPY_OUTPUT_DIR,
    "pandas": PANDAS_OUTPUT_DIR,
    "sklearn": SKLEARN_OUTPUT_DIR,
    "scipy": SCIPY_OUTPUT_DIR,
    "visualization": VISUALIZATION_OUTPUT_DIR,
    "data_vis_distribution": DATA_VIS_DISTRIBUTION_DIR,
    "data_vis_scatter": DATA_VIS_SCATTER_DIR,
    "data_vis_correlation": DATA_VIS_CORRELATION_DIR,
    "data_vis_feature_space": DATA_VIS_FEATURE_SPACE_DIR,
}


def resolveOutputDir(outputKey: str, create: bool = True) -> Path:
    """
    解析输出目录。

    说明：
    1. 已知 key 映射到现有目录；
    2. 未知 key 直接映射到 `outputs/<key>`；
    3. 不改 `outputs/` 的顶层结构。
    """
    key = outputKey.lower()
    outputDir = OUTPUT_DIR_MAP.get(key, OUTPUTS_ROOT / key)
    if create:
        outputDir.mkdir(parents=True, exist_ok=True)
    return outputDir


def get_output_dir(module: str, create: bool = True) -> Path:
    """兼容 Basic 模块的输出目录接口。"""
    return resolveOutputDir(module, create=create)


def get_output_path(module: str, filename: str, create_dir: bool = True) -> Path:
    """兼容 Basic 模块的输出文件接口。"""
    return resolveOutputDir(module, create=create_dir) / filename


def get_model_output_dir(model_name: str, create: bool = True) -> Path:
    """兼容旧模型输出目录接口。"""
    return resolveOutputDir(model_name, create=create)
