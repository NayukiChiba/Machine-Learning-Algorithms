"""
NumPy 实用函数
对应文档: ../../docs/numpy/11-utilities.md

使用方式：
    python 11_utilities.py
"""

import numpy as np


def demo_sort():
    """排序函数"""
    print("=" * 50)
    print("1. 排序函数 (sort, argsort)")
    print("=" * 50)
    
    arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    print(f"原数组: {arr}")
    print()
    
    # np.sort: 返回排序后的副本
    sorted_arr = np.sort(arr)
    print(f"np.sort(arr): {sorted_arr}")
    print(f"原数组未变: {arr}")
    print()
    
    # argsort: 返回排序后的索引
    indices = np.argsort(arr)
    print(f"np.argsort(arr): {indices}")
    print(f"使用索引重建: {arr[indices]}")
    print()
    
    # 二维数组排序
    arr_2d = np.array([[3, 1, 2], [6, 4, 5]])
    print(f"二维数组:\n{arr_2d}")
    print(f"按行排序 (axis=1):\n{np.sort(arr_2d, axis=1)}")
    print(f"按列排序 (axis=0):\n{np.sort(arr_2d, axis=0)}")


def demo_unique():
    """唯一值和计数"""
    print("=" * 50)
    print("2. 唯一值 (unique)")
    print("=" * 50)
    
    arr = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    print(f"原数组: {arr}")
    print()
    
    # 获取唯一值
    unique = np.unique(arr)
    print(f"唯一值: {unique}")
    
    # 获取唯一值及其索引
    unique, indices = np.unique(arr, return_index=True)
    print(f"首次出现的索引: {indices}")
    
    # 获取唯一值及其计数
    unique, counts = np.unique(arr, return_counts=True)
    print(f"每个值的计数: {counts}")
    print(f"值-计数对: {list(zip(unique, counts))}")


def demo_set_operations():
    """集合操作"""
    print("=" * 50)
    print("3. 集合操作")
    print("=" * 50)
    
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([3, 4, 5, 6, 7])
    print(f"a = {a}")
    print(f"b = {b}")
    print()
    
    # 交集
    print(f"np.intersect1d(a, b) 交集: {np.intersect1d(a, b)}")
    
    # 并集
    print(f"np.union1d(a, b) 并集: {np.union1d(a, b)}")
    
    # 差集
    print(f"np.setdiff1d(a, b) 差集(a-b): {np.setdiff1d(a, b)}")
    print(f"np.setdiff1d(b, a) 差集(b-a): {np.setdiff1d(b, a)}")
    
    # 对称差集
    print(f"np.setxor1d(a, b) 对称差集: {np.setxor1d(a, b)}")
    
    # 成员检测
    print(f"np.in1d(a, [2, 4]): {np.in1d(a, [2, 4])}")


def demo_search():
    """搜索函数"""
    print("=" * 50)
    print("4. 搜索函数")
    print("=" * 50)
    
    arr = np.array([1, 5, 2, 8, 3, 9, 4, 7])
    print(f"数组: {arr}")
    print()
    
    # argmax, argmin
    print(f"最大值索引 argmax: {np.argmax(arr)}")
    print(f"最小值索引 argmin: {np.argmin(arr)}")
    print()
    
    # where
    indices = np.where(arr > 5)
    print(f"大于5的元素索引: {indices[0]}")
    print(f"大于5的元素值: {arr[indices]}")
    print()
    
    # nonzero
    arr_with_zeros = np.array([0, 1, 0, 2, 0, 3])
    print(f"数组: {arr_with_zeros}")
    print(f"非零元素索引: {np.nonzero(arr_with_zeros)[0]}")


def demo_clip_round():
    """裁剪和取整"""
    print("=" * 50)
    print("5. 裁剪和取整")
    print("=" * 50)
    
    # clip: 裁剪到指定范围
    arr = np.array([1, 5, 10, 15, 20])
    print(f"原数组: {arr}")
    clipped = np.clip(arr, 5, 15)
    print(f"np.clip(arr, 5, 15): {clipped}")
    print()
    
    # 取整函数
    arr_float = np.array([1.2, 2.5, 3.7, -1.2, -2.5, -3.7])
    print(f"浮点数组: {arr_float}")
    print(f"np.floor 向下取整: {np.floor(arr_float)}")
    print(f"np.ceil 向上取整: {np.ceil(arr_float)}")
    print(f"np.round 四舍五入: {np.round(arr_float)}")
    print(f"np.trunc 截断取整: {np.trunc(arr_float)}")


def demo_copy():
    """复制操作"""
    print("=" * 50)
    print("6. 复制操作 (copy vs view)")
    print("=" * 50)
    
    arr = np.array([1, 2, 3, 4, 5])
    print(f"原数组: {arr}")
    print()
    
    # 赋值（引用）
    arr_ref = arr
    arr_ref[0] = 100
    print(f"赋值 arr_ref = arr，修改 arr_ref[0]=100")
    print(f"原数组变化: {arr}")
    arr[0] = 1  # 恢复
    print()
    
    # 视图
    arr_view = arr.view()
    arr_view[1] = 200
    print(f"视图 arr.view()，修改 view[1]=200")
    print(f"原数组变化: {arr}")
    arr[1] = 2  # 恢复
    print()
    
    # 副本
    arr_copy = arr.copy()
    arr_copy[2] = 300
    print(f"副本 arr.copy()，修改 copy[2]=300")
    print(f"原数组不变: {arr}")


def demo_all():
    """运行所有演示"""
    demo_sort()
    print()
    demo_unique()
    print()
    demo_set_operations()
    print()
    demo_search()
    print()
    demo_clip_round()
    print()
    demo_copy()


if __name__ == "__main__":
    demo_all()
