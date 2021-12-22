import copy
import logging
import random
from abc import ABCMeta, abstractmethod
from collections import deque
from itertools import product
from typing import cast

import numpy as np

from core import Instance_T, PartialSolution, LocalSearch, AbstractMove


logger = logging.getLogger(__name__)


class TabuMove(AbstractMove, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def _tabu_move(ps: PartialSolution, seen=None):
        raise NotImplementedError


class TranspOneTabu(TabuMove):
    @staticmethod
    def move(ps: PartialSolution, **kwargs) -> None:
        TranspOneTabu._tabu_move(ps, **kwargs)

    @staticmethod
    def _tabu_move(ps: PartialSolution, seen=None):
        if seen is None:
            raise ValueError()

        best_d = 0
        i = random.randint(0, ps.n - 1)
        best_j = None
        for j in range(ps.m):
            if ps.hash_put(i, j) not in seen:
                d = ps.delta_put(i, j)
                if d > best_d:
                    best_j = j
                    best_d = d
        if best_j:
            ps.put_item(i, best_j)


class SwapMinMaxTabu(TabuMove):
    @staticmethod
    def move(ps: PartialSolution, **kwargs) -> None:
        SwapMinMaxTabu._tabu_move(ps, **kwargs)

    @staticmethod
    def _tabu_move(ps: PartialSolution, seen=None):
        max_idx = cast(int, np.argmax(ps.sums))
        min_idx = cast(int, np.argmin(ps.sums))
        max_indices = ps.get_index_list(max_idx)
        min_indices = ps.get_index_list(min_idx)
        best_d = 0
        best_pair = None
        for i, j in product(max_indices, min_indices):
            d = ps.delta_swap(i, j)
            if d > best_d:
                best_d = d
                best_pair = i, j
        if best_pair:
            ps.put_item(best_pair[0], min_idx)
            ps.put_item(best_pair[1], max_idx)


class TabuSearch(LocalSearch):
    """ Tabu Search algorithm for MNP """
    def __init__(
            self,
            ps: PartialSolution,
            move: AbstractMove,
            ttl: int = 10,
            max_iter: int = 1000,
            **kwargs,
    ):
        if not ps.is_full:
            raise ValueError("Stating solution must be full assignment!")

        super(TabuSearch, self).__init__(**kwargs)

        self.ps = copy.deepcopy(ps)
        self.move = move
        self.ttl = ttl
        self.max_iter = max_iter

    def solve(self, data: Instance_T, m: int) -> PartialSolution:
        cur_sol = copy.deepcopy(self.ps)
        best_sol = self.ps
        seen = deque([hash(cur_sol)])
        for _ in range(self.max_iter):
            self.move.move(cur_sol, seen=seen)

            if cur_sol.squared_error < best_sol.squared_error:
                best_sol = copy.deepcopy(cur_sol)

            seen.append(hash(cur_sol))
            # чистим устаревшие запреты
            while len(seen) >= self.ttl:
                seen.popleft()

        return best_sol

