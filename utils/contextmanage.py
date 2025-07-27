"""
上下文管理工具
"""

from contextlib import contextmanager
import time
from typing import Iterator

import numpy as np


@contextmanager
def temp_seed(seed: int) -> Iterator[None]:
    """
    临时设置随机种子

    args:
        seed(int): 随机种子
    """
    state = np.random.get_state()
    np.random.seed(seed=seed)
    try:
        yield
    finally:
        np.random.set_state(state=state)


@contextmanager
def timer(name: str = "耗时", verbose: bool = True) -> Iterator[None]:
    """
    计时上下文管理器

    args:
        name(str): 计时标题
        verbose(bool): 是否打印耗时
    """
    start = time.time()
    try:
        yield
    finally:
        if verbose:
            cost = time.time() - start
            print(f"{name}: {cost:.4f}s")
