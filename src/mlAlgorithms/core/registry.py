"""
简单注册表实现
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Iterable, TypeVar


T = TypeVar("T")


@dataclass
class Registry(Generic[T]):
    """基于字典的简单注册表。"""

    items: dict[str, T] = field(default_factory=dict)

    def register(self, itemId: str, item: T) -> None:
        """注册一个对象。"""
        if itemId in self.items:
            raise KeyError(f"重复注册: {itemId}")
        self.items[itemId] = item

    def get(self, itemId: str) -> T:
        """获取一个对象。"""
        if itemId not in self.items:
            raise KeyError(f"未注册的条目: {itemId}")
        return self.items[itemId]

    def values(self) -> Iterable[T]:
        """返回所有注册值。"""
        return self.items.values()

    def keys(self) -> Iterable[str]:
        """返回所有键。"""
        return self.items.keys()

    def contains(self, itemId: str) -> bool:
        """判断条目是否存在。"""
        return itemId in self.items
