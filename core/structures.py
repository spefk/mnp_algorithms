from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from typing import List, Optional

import numpy as np
from more_itertools import flatten


logger = logging.getLogger(__name__)


Elem_T = int
Instance_T = List[Elem_T]
Solution_T = List[List[Elem_T]]


class PartialSolution:
    """ Structure to contain a partial solution of MNP """
    def __init__(self, data: Instance_T, m=2):
        self.n = len(data)
        self.m = m
        self.perfect = self.get_perfect(data, m)
        self.cost = np.array(data)
        self.partitioning = np.zeros((m, self.n), dtype=bool)
        self.sums = np.zeros(self.m, dtype=int)

    def cost_item(self, i: int) -> int:
        return self.cost[i]

    @staticmethod
    def from_solution(sol: Solution_T):
        obj = PartialSolution(list(flatten(sol)), m=len(sol))
        it = 0
        for j, x in enumerate(sol):
            for i in range(it, it + len(x)):
                it += 1
                obj.put_item(i, j)
        return obj

    def find_item(self, i: int) -> Optional[int]:
        for j in range(self.m):
            if self.partitioning[j][i]:
                return j
        return None

    def reset_item(self, i: int) -> None:
        j = self.find_item(i)
        if j is not None:
            self.partitioning[j][i] = False
            self.sums[j] -= self.cost[i]

    def put_item(self, i: int, j: int) -> None:
        self.reset_item(i)
        self.partitioning[j][i] = True
        self.sums[j] += self.cost[i]

    def recalculate_sums(self) -> None:
        for j in range(self.m):
            self.sums[j] = np.dot(self.partitioning[j], self.cost)

    def get_index_list(self, j: int) -> np.ndarray:
        return np.where(self.partitioning[j])[0]

    @staticmethod
    def get_perfect(data: Instance_T, m: int) -> float:
        return sum(data) / m

    @staticmethod
    def _squared_error(sums: np.ndarray, perfect: float):
        return sum((sums - perfect) ** 2)

    @property
    def squared_error(self) -> float:
        return self._squared_error(self.sums, self.perfect)

    @property
    def abs_error(self) -> float:
        return sum(np.abs(self.sums - self.perfect))

    @property
    def is_full(self) -> bool:
        return np.sum(self.partitioning) == self.n

    @property
    def solution(self) -> List[List[Elem_T]]:
        if not self.is_full:
            raise RuntimeError("Partial solution is not complete, can't construct solution.")

        return [
            [
                self.cost[i]
                for i in range(self.n)
                if self.partitioning[j][i]
            ]
            for j in range(self.m)
        ]

    def __eq__(self, other: PartialSolution) -> bool:
        if not isinstance(other, PartialSolution):
            raise TypeError("Types are incompatible for ==")
        return (
            self.n == other.n
            and self.m == other.m
            and np.array_equal(self.partitioning, other.partitioning)
        )

    def __hash__(self):
        return hash(self.partitioning.tobytes())

    def hash_put(self, i: int, j: int):
        s_idx = self.find_item(i)
        self.put_item(i, j)
        hs = hash(self)
        self.put_item(i, s_idx)
        return hs

    def delta_put(self, i: int, j: int):
        sum_copy = self.sums.copy()
        s_idx = self.find_item(i)
        if s_idx:
            sum_copy[s_idx] -= self.cost[i]
        sum_copy[j] += self.cost[i]
        return self.squared_error - self._squared_error(sum_copy, self.perfect)

    def delta_swap(self, i_1: int, i_2: int):
        set_1 = self.find_item(i_1)
        set_2 = self.find_item(i_2)
        return (
            abs(self.sums[set_1] - self.perfect)
            + abs(self.sums[set_2] - self.perfect)
            - abs(self.sums[set_1] - self.cost[i_1] + self.cost[i_2] - self.perfect)
            - abs(self.sums[set_2] - self.cost[i_2] + self.cost[i_1] - self.perfect)
        )

    @staticmethod
    def nested_list_to_str(sol: Solution_T):
        return "\n".join(
            (
                f"{i}: " + " ".join(map(str, x))
                for i, x in enumerate(sol)
            )
        )

    def __repr__(self):
        return f"Statistics for partial solution {id(self)}.\n"\
               f"IS_FULL:{self.is_full}\n"\
               f"SETS:\n"\
               f"{self.nested_list_to_str(self.solution)}\n"\
               f"Diffs: {self.sums - self.perfect}\n" \
               f"Sum Abs Errors: {self.abs_error}\n" \
               f"STD: {np.std(self.sums - self.perfect)}\n"


class AbstractSolver(metaclass=ABCMeta):
    """ Base class for all algorithms solving MNP """
    @abstractmethod
    def solve(self, data: Instance_T, m: int) -> PartialSolution:
        raise NotImplementedError


class LocalSearch(AbstractSolver, metaclass=ABCMeta):
    pass
