import gurobipy as grp

from typing import Collection
from gurobipy import GRB

from core import MNPAlgorithm, MNP_Number, MNP_Answer


class GurobiAlgorithm(MNPAlgorithm):
    """ Uses Gurobi optimizer to solve MNP problem. """
    def run(self, data: Collection[MNP_Number]) -> MNP_Answer:
        model = grp.Model("ip")

        if self.settings.get('TimeLimit', None):
            model.setParam('TimeLimit', self.settings.get('TimeLimit', None))

        n = len(data)
        m = self.number_of_sets
        _perfect = self.get_perfect_mnp_value(data)

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

        return sets
