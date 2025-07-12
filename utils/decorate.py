'''
装饰器的部分
'''

def print_func_info(func):
    def wrapper(*args, **kwargs):
        print("="*25 + f"调用函数{func.__name__}" + "="*25)
        result = func(*args, **kwargs)
        return result
    return wrapper
