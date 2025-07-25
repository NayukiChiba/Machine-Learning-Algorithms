"""
装饰器的部分
"""

from functools import wraps
import time


def print_func_info(func):
    def wrapper(*args, **kwargs):
        print("=" * 25 + f"调用函数{func.__name__}" + "=" * 25)
        result = func(*args, **kwargs)
        return result

    return wrapper


def timeit(func):
    """
    打印函数耗时的装饰器
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        cost = time.time() - start
        print(f"函数 {func.__name__} 耗时: {cost:.4f}s")
        return result

    return wrapper
