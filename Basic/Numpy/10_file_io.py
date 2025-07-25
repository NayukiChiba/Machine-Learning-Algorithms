"""
NumPy 文件操作
对应文档: ../../docs/numpy/10-file-io.md

使用方式：
    python 10_file_io.py
"""

import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到搜索路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_output_dir


def demo_save_load_npy():
    """save 和 load (.npy 格式)"""
    print("=" * 50)
    print("1. save 和 load (.npy 二进制格式)")
    print("=" * 50)

    # 获取输出目录
    output_dir = get_output_dir("numpy")

    # 创建测试数组
    np.random.seed(42)
    arr = np.random.random((3, 4))
    print(f"原数组:\n{arr}")
    print()

    # 保存
    filepath = output_dir / "array.npy"
    np.save(filepath, arr)
    print(f"已保存到: {filepath}")
    print(f"文件大小: {filepath.stat().st_size} 字节")
    print()

    # 加载
    loaded = np.load(filepath)
    print(f"加载的数组:\n{loaded}")
    print(f"是否相同: {np.array_equal(arr, loaded)}")


def demo_savez():
    """savez 保存多个数组"""
    print("=" * 50)
    print("2. savez 保存多个数组 (.npz)")
    print("=" * 50)

    output_dir = get_output_dir("numpy")

    # 创建多个数组
    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.array([[1, 2], [3, 4]])
    arr3 = np.arange(10)

    # 保存多个数组
    filepath = output_dir / "arrays.npz"
    np.savez(filepath, a=arr1, b=arr2, c=arr3)
    print(f"已保存到: {filepath}")
    print()

    # 加载
    data = np.load(filepath)
    print(f"包含的数组: {list(data.keys())}")
    print(f"data['a']: {data['a']}")
    print(f"data['b']:\n{data['b']}")
    print(f"data['c']: {data['c']}")


def demo_savetxt_loadtxt():
    """savetxt 和 loadtxt (文本格式)"""
    print("=" * 50)
    print("3. savetxt 和 loadtxt (文本格式)")
    print("=" * 50)

    output_dir = get_output_dir("numpy")

    # 创建测试数组
    np.random.seed(42)
    arr = np.random.random((3, 4))
    print(f"原数组:\n{arr}")
    print()

    # 默认格式保存
    filepath = output_dir / "array.txt"
    np.savetxt(filepath, arr)
    print(f"默认格式保存到: {filepath}")

    with open(filepath, "r") as f:
        print(f"文件内容:\n{f.read()}")

    # 自定义格式保存 (CSV)
    filepath_csv = output_dir / "array.csv"
    np.savetxt(
        filepath_csv,
        arr,
        delimiter=",",
        fmt="%.4f",
        header="col1,col2,col3,col4",
        comments="",
    )
    print(f"CSV 格式保存到: {filepath_csv}")

    with open(filepath_csv, "r") as f:
        print(f"CSV 内容:\n{f.read()}")

    # 加载
    loaded = np.loadtxt(filepath)
    print(f"加载的数组:\n{loaded}")
    print(f"是否接近: {np.allclose(arr, loaded)}")


def demo_format_options():
    """不同格式选项"""
    print("=" * 50)
    print("4. 格式选项 (fmt)")
    print("=" * 50)

    output_dir = get_output_dir("numpy")

    arr = np.array([[1.23456789, 2.34567890], [3.45678901, 4.56789012]])
    print(f"原数组:\n{arr}")
    print()

    formats = [
        ("%.2f", "2位小数"),
        ("%.4f", "4位小数"),
        ("%d", "整数"),
        ("%.2e", "科学计数法"),
        ("%10.4f", "宽度10,4位小数"),
    ]

    for fmt, desc in formats:
        filepath = output_dir / f"fmt_{desc}.txt"
        try:
            np.savetxt(filepath, arr, fmt=fmt)
            with open(filepath, "r") as f:
                content = f.readline().strip()
            print(f"{desc} ({fmt}): {content}")
        except Exception as e:
            print(f"{desc} ({fmt}): 错误 - {e}")


def demo_header_skiprows():
    """带表头的文件处理"""
    print("=" * 50)
    print("5. 带表头的文件处理")
    print("=" * 50)

    output_dir = get_output_dir("numpy")

    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # 保存带表头的文件
    filepath = output_dir / "with_header.csv"
    np.savetxt(filepath, arr, delimiter=",", fmt="%d", header="A,B,C", comments="")

    print("保存的文件内容:")
    with open(filepath, "r") as f:
        print(f.read())

    # 加载时跳过表头
    loaded = np.loadtxt(filepath, delimiter=",", skiprows=1)
    print(f"加载的数组 (skiprows=1):\n{loaded}")


def demo_all():
    """运行所有演示"""
    demo_save_load_npy()
    print()
    demo_savez()
    print()
    demo_savetxt_loadtxt()
    print()
    demo_format_options()
    print()
    demo_header_skiprows()


if __name__ == "__main__":
    demo_all()
