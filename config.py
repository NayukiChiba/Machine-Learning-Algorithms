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

# 模型训练输出目录
MODEL_TRAINING_ROOT = OUTPUTS_ROOT / "model_training"
MT_CLASSIFICATION_DIR = MODEL_TRAINING_ROOT / "classification"
MT_REGRESSION_DIR = MODEL_TRAINING_ROOT / "regression"
MT_CLUSTERING_DIR = MODEL_TRAINING_ROOT / "clustering"
MT_ENSEMBLE_DIR = MODEL_TRAINING_ROOT / "ensemble"
MT_DIMENSIONALITY_DIR = MODEL_TRAINING_ROOT / "dimensionality"
MT_PROBABILISTIC_DIR = MODEL_TRAINING_ROOT / "probabilistic"


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
        "mt_classification": MT_CLASSIFICATION_DIR,
        "mt_regression": MT_REGRESSION_DIR,
        "mt_clustering": MT_CLUSTERING_DIR,
        "mt_ensemble": MT_ENSEMBLE_DIR,
        "mt_dimensionality": MT_DIMENSIONALITY_DIR,
        "mt_probabilistic": MT_PROBABILISTIC_DIR,
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
        DATA_VIS_DISTRIBUTION_DIR,
        DATA_VIS_SCATTER_DIR,
        DATA_VIS_CORRELATION_DIR,
        DATA_VIS_FEATURE_SPACE_DIR,
        MODEL_TRAINING_ROOT,
        MT_CLASSIFICATION_DIR,
        MT_REGRESSION_DIR,
        MT_CLUSTERING_DIR,
        MT_ENSEMBLE_DIR,
        MT_DIMENSIONALITY_DIR,
        MT_PROBABILISTIC_DIR,
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
    print("数据可视化输出目录:")
    print(f"  分布图:        {DATA_VIS_DISTRIBUTION_DIR}")
    print(f"  散点图:        {DATA_VIS_SCATTER_DIR}")
    print(f"  相关性热力图:   {DATA_VIS_CORRELATION_DIR}")
    print(f"  特征空间:      {DATA_VIS_FEATURE_SPACE_DIR}")
    print()
    print("模型训练输出目录:")
    print(f"  分类 (Classification): {MT_CLASSIFICATION_DIR}")
    print(f"  回归 (Regression):     {MT_REGRESSION_DIR}")
    print(f"  聚类 (Clustering):     {MT_CLUSTERING_DIR}")
    print(f"  集成 (Ensemble):       {MT_ENSEMBLE_DIR}")
    print(f"  降维 (Dimensionality): {MT_DIMENSIONALITY_DIR}")
    print(f"  概率 (Probabilistic):  {MT_PROBABILISTIC_DIR}")
    print()
    print("测试 get_output_path:")
    print(
        f"  get_output_path('numpy', 'test.npy') = {get_output_path('numpy', 'test.npy')}"
    )
