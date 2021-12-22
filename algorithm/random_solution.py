import random

from core import AbstractSolver, PartialSolution, Instance_T


class RandomSolver(AbstractSolver):
    def solve(self, data: Instance_T, m: int) -> PartialSolution:
        ps = PartialSolution(data, m=m)
        for i in range(ps.n):
            ps.put_item(i, random.randint(0, ps.m - 1))
        return ps
