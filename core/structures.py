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

    @property
    def mse(self) -> float:
        return sum((self.sums - self.perfect) ** 2)

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

    def __hash__(self):
        return hash(self.partitioning.tobytes())

    @staticmethod
    def nested_list_to_str(sol: Solution_T):
        return "\n".join(
            map(
                lambda x: " ".join(map(str, x)),
                sol
            )
        )

    def __repr__(self):
        return f"Statistics for partial solution {id(self)}.\n"\
               f"IS_FULL:{self.is_full}\n"\
               f"SETS:\n"\
               f"{self.nested_list_to_str(self.solution)}\n"\
               f"MSE: {self.mse}\n"


class AbstractSolver(metaclass=ABCMeta):
    """ Base class for all algorithms solving MNP """
    @abstractmethod
    def solve(self, data: Instance_T, m: int) -> PartialSolution:
        raise NotImplementedError
