from typing import cast

import numpy as np

from core import AbstractSolver, Instance_T, PartialSolution, Solution_T


class GreedySolver(AbstractSolver):
    """ Greedy heuristics to solve MNP problem """
    def solve(self, data: Instance_T, m: int) -> PartialSolution:
        ps = PartialSolution(data, m=m)
        for value, idx in sorted(((x, idx) for idx, x in enumerate(data)), reverse=True):
            ps.put_item(idx, cast(int, np.argmin(ps.sums)))
        return ps
