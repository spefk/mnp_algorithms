from __future__ import annotations

from typing import List, Optional
from heapq import heappop, heappush

import numpy as np

from core import Instance_T, Solution_T, AbstractSolver, PartialSolution


class _PartialSet:
    def __init__(self, m: int = 2, first_element: Optional[int] = None):
        self.m = m
        self.sets: Solution_T = [[] for _ in range(m)]
        self.errors: np.ndarray = np.zeros(m, dtype=int)
        if first_element:
            self.sets[-1].append(first_element)
            self.errors[-1] = first_element

    @property
    def error(self) -> int:
        return sum(self.errors)

    def _ground_errors(self) -> None:
        _m = min(self.errors)
        self.errors = [x - _m for x in self.errors]

    def _reorder(self) -> None:
        self.errors, self.sets = map(
            list,
            zip(*[[x, y] for x, y in sorted(zip(self.errors, self.sets))])
        )

    def __or__(self, other: _PartialSet) -> _PartialSet:
        # merges two PartialPlace in memory of first one (self)
        if self.m != other.m:
            raise ValueError("Wrong operands! Number of sets in operands is not equal.")

        for x, y in zip(self.sets, reversed(other.sets)):
            x.extend(y)
        self.errors = [x + y for x, y in zip(self.errors, reversed(other.errors))]
        self._ground_errors()
        self._reorder()
        return self

    def __lt__(self, other: _PartialSet) -> bool:
        # for heapq lesser is one with greater error
        return self.error > other.error

    def __repr__(self):
        return f">> PartialSet >>"\
               f"\nid: {id(self)},"\
               f"\nsets: {self.sets},"\
               f"\nerrors: {self.errors},"\
               f"\nerror: {self.error}."

    def __str__(self):
        return self.__repr__()


class KarmarkarKarp(AbstractSolver):
    """ Karmarkar-Karp Algorithm implementation """
    def __init__(self, *args, **kwargs):
        super(KarmarkarKarp, self).__init__(*args, **kwargs)
        self._solution = None

    def solve(self, data: Instance_T, m: int = 2) -> PartialSolution:
        if len(data) == 0:
            return PartialSolution.from_solution([[] for _ in range(m)])

        queue: List[_PartialSet] = []
        for x in data:
            heappush(queue, _PartialSet(m=m, first_element=x))

        while len(queue) > 1:
            heappush(queue, heappop(queue) | heappop(queue))

        self._solution = queue[0]
        return PartialSolution.from_solution(self._solution.sets)


if __name__ == '__main__':
    kk = KarmarkarKarp()
    print(kk.solve([1, 2, 3, 4, 5, 6, 7], m=2))
    print(kk._solution)
    print('\n')

    print(kk.solve([1, 2, 3, 4, 5, 6, 7, 8], m=2))
    print(kk.solve([1, 2, 3, 4, 5, 6, 7, 8, 9], m=2))
    print(kk.solve([1, 2, 3, 4, 5, 6, 7, 8, 9], m=4))
    print(kk.solve(range(0, 100, 3), m=7))
    print(kk.solve(range(0, 1000), m=7))
