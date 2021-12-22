from typing import Optional

import gurobipy as grp

from gurobipy import GRB

from core import Instance_T, AbstractSolver, PartialSolution


class GurobiSolver(AbstractSolver):
    """ Uses Gurobi optimizer to solve MNP problem. """
    def __init__(self, timelimit: Optional[int] = 300):
        self.timelimit = timelimit

    def solve(self, data: Instance_T, m: int) -> PartialSolution:
        model = grp.Model("ip")

        if self.timelimit:
            model.setParam('TimeLimit', self.timelimit)

        n = len(data)
        _perfect = PartialSolution.get_perfect(data, m)

        _vars = model.addVars(
            (
                (i, j)
                for j in range(m)
                for i in range(n)
            ),
            vtype=GRB.BINARY,
        )
        model.addConstrs(
            sum(_vars[i, j] for j in range(m)) == 1
            for i in range(n)
        )
        model.setObjective(
            sum([
                (sum([elem * _vars[i, j] for i, elem in enumerate(data)]) - _perfect) ** 2
                for j in range(m)
            ]),
            GRB.MINIMIZE
        )
        model.optimize()

        sets = [
            [elem for i, elem in enumerate(data) if _vars[i, j].x == 1]
            for j in range(m)
        ]

        return PartialSolution.from_solution(sets)
