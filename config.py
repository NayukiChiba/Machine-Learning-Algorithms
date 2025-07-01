"""
项目配置文件
统一管理输出目录和其他全局配置

使用方式：
    from config import get_output_dir, OUTPUTS_ROOT

    # 获取对应模块的输出目录
    output_dir = get_output_dir("numpy")  # 返回 outputs/numpy

    # 或者直接使用路径常量
    from config import NUMPY_OUTPUT_DIR
"""

import os
from pathlib import Path

# 项目根目录 (config.py 所在目录)
PROJECT_ROOT = Path(__file__).parent.absolute()

# 输出文件根目录
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"

# 各模块输出目录
NUMPY_OUTPUT_DIR = OUTPUTS_ROOT / "numpy"
PANDAS_OUTPUT_DIR = OUTPUTS_ROOT / "pandas"
SKLEARN_OUTPUT_DIR = OUTPUTS_ROOT / "sklearn"
SCIPY_OUTPUT_DIR = OUTPUTS_ROOT / "scipy"
VISUALIZATION_OUTPUT_DIR = OUTPUTS_ROOT / "visualization"


def get_output_dir(module: str, create: bool = True) -> Path:
    """
    获取指定模块的输出目录

    Args:
        module: 模块名称 (numpy, pandas, sklearn, scipy, visualization)
        create: 是否自动创建目录，默认 True

    Returns:
        Path: 输出目录的 Path 对象

    Example:
        >>> output_dir = get_output_dir("numpy")
        >>> filepath = output_dir / "array.npy"
        >>> np.save(filepath, arr)
    """
    module_dirs = {
        "numpy": NUMPY_OUTPUT_DIR,
        "pandas": PANDAS_OUTPUT_DIR,
        "sklearn": SKLEARN_OUTPUT_DIR,
        "scipy": SCIPY_OUTPUT_DIR,
        "visualization": VISUALIZATION_OUTPUT_DIR,
    }

    module_lower = module.lower()
    if module_lower not in module_dirs:
        # 未知模块，创建新目录
        output_dir = OUTPUTS_ROOT / module_lower
    else:
        output_dir = module_dirs[module_lower]

    if create:
        output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def get_output_path(module: str, filename: str, create_dir: bool = True) -> Path:
    """
    获取指定模块下文件的完整路径

    Args:
        module: 模块名称
        filename: 文件名
        create_dir: 是否自动创建目录

    Returns:
        Path: 文件的完整路径

    Example:
        >>> filepath = get_output_path("numpy", "array.npy")
        >>> np.save(filepath, arr)
    """
    output_dir = get_output_dir(module, create=create_dir)
    return output_dir / filename


def init_output_dirs():
    """
    初始化所有输出目录
    在项目启动时调用，确保所有目录存在
    """
    dirs = [
        OUTPUTS_ROOT,
        NUMPY_OUTPUT_DIR,
        PANDAS_OUTPUT_DIR,
        SKLEARN_OUTPUT_DIR,
        SCIPY_OUTPUT_DIR,
        VISUALIZATION_OUTPUT_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# 模块导入时自动初始化目录
init_output_dirs()


if __name__ == "__main__":
    # 测试配置
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"输出根目录: {OUTPUTS_ROOT}")
    print()
    print("各模块输出目录:")
    print(f"  NumPy:         {NUMPY_OUTPUT_DIR}")
    print(f"  Pandas:        {PANDAS_OUTPUT_DIR}")
    print(f"  Scikit-learn:  {SKLEARN_OUTPUT_DIR}")
    print(f"  SciPy:         {SCIPY_OUTPUT_DIR}")
    print(f"  Visualization: {VISUALIZATION_OUTPUT_DIR}")
    print()
    print("测试 get_output_path:")
    print(f"  get_output_path('numpy', 'test.npy') = {get_output_path('numpy', 'test.npy')}")
