# NumPy 基础入门

> 对应代码: [01_basics.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Numpy/01_basics.py)

## 学习目标

- 了解什么是 NumPy 及其重要性
- 掌握 NumPy 的安装和导入方法
- 理解 NumPy 与 Python 列表的区别

## 什么是 NumPy

NumPy（Numerical Python）是 Python 科学计算的基础库，提供：

- 高性能的多维数组对象 `ndarray`
- 丰富的数学函数库
- 线性代数、傅里叶变换、随机数生成等功能
- 与 C/C++ 和 Fortran 代码集成的工具

## 为什么学习 NumPy

| 优势             | 说明                                              |
| ---------------- | ------------------------------------------------- |
| **性能优势**     | 底层使用 C 语言实现，比纯 Python 快 10-100 倍     |
| **内存效率**     | 连续内存存储，支持向量化操作                      |
| **生态基础**     | Pandas、Scikit-learn、TensorFlow 等库都基于 NumPy |
| **科学计算标准** | 数据科学和机器学习的必备技能                      |

## 安装和导入

```python
# 安装
pip install numpy

# 导入（约定使用 np 别名）
import numpy as np

# 查看版本
print(np.__version__)
```

## NumPy 数组 vs Python 列表

| 特性         | NumPy 数组               | Python 列表            |
| ------------ | ------------------------ | ---------------------- |
| **数据类型** | 同质（所有元素类型相同） | 异质（可包含不同类型） |
| **内存**     | 连续存储，效率高         | 分散存储，效率低       |
| **运算**     | 支持向量化运算           | 需要循环遍历           |
| **速度**     | 快（C 实现）             | 慢（Python 实现）      |

### 运算行为对比

```python
# Python 列表
py_list = [1, 2, 3, 4, 5]
py_list * 2  # 结果: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]（列表重复）

# NumPy 数组
np_array = np.array([1, 2, 3, 4, 5])
np_array * 2  # 结果: [2, 4, 6, 8, 10]（元素级运算）
```

## 重要概念

| 概念           | 说明                                          |
| -------------- | --------------------------------------------- |
| **ndarray**    | NumPy 的多维数组对象，是 NumPy 的核心数据结构 |
| **向量化操作** | 对整个数组进行操作，无需显式循环              |
| **广播**       | 不同形状数组之间的运算规则                    |

## 常用配置函数

```python
# 显示 NumPy 配置信息
np.show_config()

# 获取打印选项
np.get_printoptions()

# 设置打印选项
np.set_printoptions(precision=4, suppress=True)
```

## 练习

运行代码文件查看完整演示：

```bash
python Basic/Numpy/01_basics.py
```
