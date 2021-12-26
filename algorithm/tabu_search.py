import copy
import logging
from collections import deque

from core import Instance_T, PartialSolution, LocalSearch, AbstractMover

logger = logging.getLogger(__name__)


class TabuSearch(LocalSearch):
    """ Tabu Search algorithm for MNP """
    def __init__(
            self,
            ps: PartialSolution,
            move: AbstractMover,
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

