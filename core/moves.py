import random
from abc import abstractmethod, ABCMeta
from typing import cast

import numpy as np

from .structures import PartialSolution


class AbstractMove(metaclass=ABCMeta):
    """ Base class for classes that perform some moves in PartialSolution """
    def __init__(self):
        raise TypeError("Move class can't be instantiate.")

    @staticmethod
    @abstractmethod
    def move(ps: PartialSolution, **kwargs) -> None:
        """ Changes some PartialSolution inplace """
        raise NotImplementedError

    @classmethod
    def move_n(cls, ps: PartialSolution, n: int, **kwargs):
        """ Apply class move N times """
        for _ in range(n):
            cls.move(ps, **kwargs)


class GreedyTranspOne(AbstractMove):
    """ One greedy step with random element """
    @staticmethod
    def move(ps: PartialSolution, **kwargs) -> None:
        idx = random.randint(0, ps.n - 1)
        ps.reset_item(idx)
        ps.put_item(idx, cast(int, np.argmin(ps.sums)))


class GreedyTranspMany(AbstractMove):
    """ One greedy step with random element """
    @staticmethod
    def move(ps: PartialSolution, **kwargs) -> None:
        idx_list = random.sample(range(ps.n), random.randint(0, ps.n - 1))

        for idx in idx_list:
            ps.reset_item(idx)

        for value, idx in sorted(((ps.cost[idx], idx) for idx in idx_list), reverse=True):
            ps.put_item(idx, cast(int, np.argmin(ps.sums)))


class ReassignOne(AbstractMove):
    """ One element reassignment """
    @staticmethod
    def move(ps: PartialSolution, **kwargs) -> None:
        ReassignOne._transp(ps, **kwargs)

    @staticmethod
    def _transp(ps: PartialSolution, i: int = -1, j: int = -1):
        if i == -1 or j == -1:
            raise ValueError("Element and Set")
        ps.put_item(i, j)


class MaxToMin(AbstractMove):
    """ Random element from max-sum set to min-sum set """
    @staticmethod
    def move(ps: PartialSolution, **kwargs) -> None:
        max_idx = cast(int, np.argmax(ps.sums))
        min_idx = cast(int, np.argmin(ps.sums))
        elem_idx = random.choice(ps.get_index_list(max_idx))
        ps.put_item(elem_idx, min_idx)
