import numpy as np
from typing import Callable, Iterable


class BaseClass:
    def __init__(self, dictionary: dict) -> None:
        for key in dictionary:
            setattr(self, key, dictionary[key])


def print_in_int(ndarray: Iterable[float]) -> str:
    """リストを整数の形式で文字列に直す"""
    return " ".join(np.array(ndarray).astype(int).astype(str))


def flatten(list_of_list: list[list]) -> list:
    """リストのリストを、リストを結合したリストに直す
    """
    return sum(list_of_list, [])


def a_equals_to_b(a: float, b: float, digit: int = 3) -> bool:
    """<digit>の桁で、<a>と<b>が一致することを確認する"""
    return 2 * abs(a - b) / (abs(a) + abs(b) + 1e-10) < 10 ** (1 - digit)


def a_smaller_than_b(a: float, b: float, digit: int = 3) -> bool:
    """<digit>の桁で、<a>が<b>未満であることを確認する"""
    return not a_equals_to_b(a, b, digit) and a < b


def relation(a: float, b: float, digit: int = 3) -> str:
    """<a>と<b>の<digit>桁での関係を返す"""
    return \
        "=" if a_equals_to_b(a, b, digit) else \
        ">" if a > b else \
        "<"


def clip_decorator(min: float = - float("inf"), max: float = float("inf")) -> Callable:
    def decorator(func) -> Callable:
        def wrapper(*args) -> bool:
            return np.clip(func(*args), min, max)
        return wrapper
    return decorator


def set_and_sort(data: list) -> list:
    return sorted(set(data), key=data.index)


class Dict(dict):
    key_lambda: Callable

    def __init__(self, dictionary: dict, key: Callable) -> None:
        super().__init__(dictionary)
        self.key_lambda = key

    def key(self) -> str:
        return self.key_lambda(self)


class List_of_Dict(list[Dict]):
    key_lambda: Callable

    def __init__(self, content: list[dict], key_lambda: Callable) -> None:
        super().__init__([Dict(d, key_lambda) for d in content])
        self.key_lambda = key_lambda

    def get(self, key: str) -> Dict:
        for d in self:
            if d.key() == key:
                return d
        return {}

    def keys(self) -> list[str]:
        return [d.key() for d in self]

    def merge(self, other_list: list[dict]) -> None:
        other: "List_of_Dict" = List_of_Dict(other_list, self.key_lambda)
        self.__init__(
            [other.get(key) | self.get(key) for key in self.keys()],
            self.key_lambda,
        )


def merge_list_of_dict(list_of_list_of_dict: list[list[dict]], key_lambda: Callable) -> List_of_Dict:
    result: List_of_Dict = List_of_Dict(list_of_list_of_dict[0], key_lambda)
    for other in list_of_list_of_dict[1:]:
        result.merge(other)
    return result
