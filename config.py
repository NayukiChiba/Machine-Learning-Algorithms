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

from pathlib import Path

# 项目根目录 (config.py 所在目录)
PROJECT_ROOT = Path(__file__).parent.absolute()

# 输出文件根目录
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"

# 基础模块输出目录
NUMPY_OUTPUT_DIR = OUTPUTS_ROOT / "numpy"
PANDAS_OUTPUT_DIR = OUTPUTS_ROOT / "pandas"
SKLEARN_OUTPUT_DIR = OUTPUTS_ROOT / "sklearn"
SCIPY_OUTPUT_DIR = OUTPUTS_ROOT / "scipy"
VISUALIZATION_OUTPUT_DIR = OUTPUTS_ROOT / "visualization"

# 数据可视化输出目录
DATA_VIS_ROOT = OUTPUTS_ROOT / "data_visualization"
DATA_VIS_DISTRIBUTION_DIR = DATA_VIS_ROOT / "distribution"
DATA_VIS_SCATTER_DIR = DATA_VIS_ROOT / "scatter"
DATA_VIS_CORRELATION_DIR = DATA_VIS_ROOT / "correlation"
DATA_VIS_FEATURE_SPACE_DIR = DATA_VIS_ROOT / "feature_space"


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
        "data_vis_distribution": DATA_VIS_DISTRIBUTION_DIR,
        "data_vis_scatter": DATA_VIS_SCATTER_DIR,
        "data_vis_correlation": DATA_VIS_CORRELATION_DIR,
        "data_vis_feature_space": DATA_VIS_FEATURE_SPACE_DIR,
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


def get_model_output_dir(model_name: str, create: bool = True) -> Path:
    """
    获取模型结果图输出目录

    Args:
        model_name: 模型名称，例如 knn、svr、lightgbm
        create: 是否自动创建目录，默认 True

    Returns:
        Path: 模型对应的输出目录，例如 outputs/knn
    """
    return get_output_dir(model_name, create=create)


if __name__ == "__main__":
    # 测试配置
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"输出根目录: {OUTPUTS_ROOT}")
    print()
    print("基础模块输出目录:")
    print(f"  NumPy:         {NUMPY_OUTPUT_DIR}")
    print(f"  Pandas:        {PANDAS_OUTPUT_DIR}")
    print(f"  Scikit-learn:  {SKLEARN_OUTPUT_DIR}")
    print(f"  SciPy:         {SCIPY_OUTPUT_DIR}")
    print(f"  Visualization: {VISUALIZATION_OUTPUT_DIR}")
    print()
    print("数据可视化输出目录:")
    print(f"  分布图:        {DATA_VIS_DISTRIBUTION_DIR}")
    print(f"  散点图:        {DATA_VIS_SCATTER_DIR}")
    print(f"  相关性热力图:   {DATA_VIS_CORRELATION_DIR}")
    print(f"  特征空间:      {DATA_VIS_FEATURE_SPACE_DIR}")
    print()
    print("模型结果图示例目录:")
    print(f"  knn:           {get_model_output_dir('knn', create=False)}")
    print(
        f"  linear_regression: {get_model_output_dir('linear_regression', create=False)}"
    )
    print()
    print("测试 get_output_path:")
    print(
        f"  get_output_path('numpy', 'test.npy') = {get_output_path('numpy', 'test.npy')}"
    )
