"""
NumPy 综合实战
对应文档: ../../docs/numpy/12-practice.md

使用方式：
    python 12_practice.py
"""

import numpy as np


def demo_student_grades():
    """学生成绩分析"""
    print("=" * 50)
    print("1. 学生成绩分析")
    print("=" * 50)
    
    np.random.seed(42)
    
    # 创建成绩数据：5名学生，3门课程
    grades = np.random.randint(60, 101, size=(5, 3))
    students = ['学生A', '学生B', '学生C', '学生D', '学生E']
    courses = ['数学', '英语', '物理']
    
    print("各科成绩:")
    print(f"{'姓名':^6} {'数学':^6} {'英语':^6} {'物理':^6}")
    print("-" * 30)
    for i, student in enumerate(students):
        print(f"{student:^6} {grades[i,0]:^6} {grades[i,1]:^6} {grades[i,2]:^6}")
    print()
    
    # 计算每个学生的总分和平均分
    total_scores = np.sum(grades, axis=1)
    avg_scores = np.mean(grades, axis=1)
    
    print("个人统计:")
    print(f"{'姓名':^6} {'总分':^6} {'平均分':^8}")
    print("-" * 25)
    for i, student in enumerate(students):
        print(f"{student:^6} {total_scores[i]:^6} {avg_scores[i]:^8.1f}")
    print()
    
    # 计算每门课程的统计信息
    course_mean = np.mean(grades, axis=0)
    course_std = np.std(grades, axis=0)
    course_max = np.max(grades, axis=0)
    course_min = np.min(grades, axis=0)
    
    print("课程统计:")
    print(f"{'课程':^6} {'平均分':^8} {'标准差':^8} {'最高':^6} {'最低':^6}")
    print("-" * 40)
    for i, course in enumerate(courses):
        print(f"{course:^6} {course_mean[i]:^8.1f} {course_std[i]:^8.1f} {course_max[i]:^6} {course_min[i]:^6}")
    print()
    
    # 找出总分最高和最低的学生
    best_idx = np.argmax(total_scores)
    worst_idx = np.argmin(total_scores)
    print(f"总分最高: {students[best_idx]} ({total_scores[best_idx]}分)")
    print(f"总分最低: {students[worst_idx]} ({total_scores[worst_idx]}分)")
    print()
    
    # 按总分排名
    rank_indices = np.argsort(total_scores)[::-1]
    print("成绩排名:")
    for rank, idx in enumerate(rank_indices, 1):
        print(f"  第{rank}名: {students[idx]} - 总分{total_scores[idx]}")


def demo_linear_regression():
    """简单线性回归实现"""
    print("=" * 50)
    print("2. 线性回归实现")
    print("=" * 50)
    
    np.random.seed(42)
    
    # 生成数据: y = 2x + 1 + 噪声
    n = 50
    x = np.linspace(0, 10, n)
    true_slope = 2
    true_intercept = 1
    noise = np.random.normal(0, 1, n)
    y = true_slope * x + true_intercept + noise
    
    print(f"生成 {n} 个数据点")
    print(f"真实参数: y = {true_slope}x + {true_intercept}")
    print()
    
    # 使用正规方程求解: w = (X^T X)^(-1) X^T y
    X = np.column_stack([np.ones(n), x])  # 设计矩阵
    
    XTX = X.T @ X
    XTy = X.T @ y
    w = np.linalg.solve(XTX, XTy)
    
    estimated_intercept = w[0]
    estimated_slope = w[1]
    
    print(f"估计参数: y = {estimated_slope:.4f}x + {estimated_intercept:.4f}")
    print(f"斜率误差: {abs(estimated_slope - true_slope):.4f}")
    print(f"截距误差: {abs(estimated_intercept - true_intercept):.4f}")
    print()
    
    # 计算预测值和误差
    y_pred = estimated_slope * x + estimated_intercept
    
    # 计算 R² 分数
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    
    # 计算 RMSE
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    
    print(f"R² 分数: {r_squared:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print()
    
    # 预测新值
    new_x = np.array([5, 10, 15])
    new_y = estimated_slope * new_x + estimated_intercept
    print("预测新值:")
    for i, xi in enumerate(new_x):
        true_y = true_slope * xi + true_intercept
        print(f"  x={xi}: 预测y={new_y[i]:.2f}, 真实y={true_y:.2f}")


def demo_image_operations():
    """模拟图像操作"""
    print("=" * 50)
    print("3. 图像操作模拟")
    print("=" * 50)
    
    np.random.seed(42)
    
    # 创建一个简单的灰度图像 (8x8)
    image = np.random.randint(0, 256, size=(8, 8), dtype=np.uint8)
    print(f"原始图像 (8x8):\n{image}")
    print()
    
    # 图像统计
    print(f"图像统计:")
    print(f"  最小值: {image.min()}")
    print(f"  最大值: {image.max()}")
    print(f"  平均值: {image.mean():.1f}")
    print(f"  标准差: {image.std():.1f}")
    print()
    
    # 图像翻转
    flipped_h = image[:, ::-1]  # 水平翻转
    flipped_v = image[::-1, :]  # 垂直翻转
    print(f"水平翻转:\n{flipped_h}")
    print()
    
    # 图像旋转 90 度
    rotated = np.rot90(image)
    print(f"旋转90度:\n{rotated}")
    print()
    
    # 图像裁剪
    cropped = image[2:6, 2:6]
    print(f"裁剪 [2:6, 2:6] (4x4):\n{cropped}")
    print()
    
    # 图像归一化到 [0, 1]
    normalized = image.astype(np.float64) / 255.0
    print(f"归一化到[0,1]:\n{normalized.round(2)}")


def demo_statistics():
    """统计分析"""
    print("=" * 50)
    print("4. 统计分析")
    print("=" * 50)
    
    np.random.seed(42)
    
    # 生成正态分布数据
    data = np.random.normal(loc=100, scale=15, size=1000)
    
    print(f"正态分布数据 (μ=100, σ=15, n=1000):")
    print(f"  样本均值: {data.mean():.2f}")
    print(f"  样本标准差: {data.std():.2f}")
    print(f"  最小值: {data.min():.2f}")
    print(f"  最大值: {data.max():.2f}")
    print()
    
    # 百分位数
    percentiles = [25, 50, 75, 90, 95, 99]
    print("百分位数:")
    for p in percentiles:
        print(f"  第{p}百分位: {np.percentile(data, p):.2f}")
    print()
    
    # 直方图统计
    hist, bin_edges = np.histogram(data, bins=10)
    print("直方图统计 (10个区间):")
    for i in range(len(hist)):
        print(f"  [{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}): {hist[i]}")


def demo_all():
    """运行所有演示"""
    demo_student_grades()
    print()
    demo_linear_regression()
    print()
    demo_image_operations()
    print()
    demo_statistics()


if __name__ == "__main__":
    demo_all()
