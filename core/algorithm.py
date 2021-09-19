import numpy as np
import gurobipy as grp

from gurobipy import GRB
from abc import abstractmethod, ABC
from typing import Union, Iterable, Tuple, List, Collection


class Algorithm(ABC):
    """Base class for algorithms.
    """
    def run(self, data: Iterable):
        _of, _solution = self.execute_algorithm(data)
        return _of, _solution

    @abstractmethod
    def execute_algorithm(self, data: Iterable) -> Tuple[Union[int, float, None], Union[List, None]]:
        raise NotImplementedError()

    @staticmethod
    def calculate_minsq_of(perfect, sets):
        return sum([(sum(s) - perfect) ** 2 for s in sets])


class MNPAlgorithm(Algorithm, ABC):

    def __init__(self, number_of_sets: int = 1, **kwargs):
        super(MNPAlgorithm, self).__init__()
        self.number_of_sets = number_of_sets
        self.settings = dict(**kwargs)

    def get_perfect_mnp(self, data):
        return float(sum(data)) / self.number_of_sets


class GurobiAlgorithm(MNPAlgorithm):
    """Uses Gurobi optimizer to solve problem
    """
    def execute_algorithm(self, data: Collection):
        model = grp.Model("ip")

        if self.settings.get('TimeLimit', None):
            model.setParam('TimeLimit', self.settings.get('TimeLimit', None))

        n = len(data)

        # variables
        _vars = [
            [
                model.addVar(vtype=GRB.BINARY, name=f"x_{i},{j}", lb=0, ub=1)
                for j in range(self.number_of_sets)
            ]
            for i in range(n)
        ]

        # constraints
        for i, col in enumerate(_vars):
            model.addConstr(sum(col) == 1, "c_{i}")

        # objective
        _perfect = self.get_perfect_mnp(data)
        model.setObjective(
            sum([
                (sum([elem * _vars[i][j] for i, elem in enumerate(data)]) - _perfect) ** 2
                for j in range(self.number_of_sets)
            ]),
            GRB.MINIMIZE
        )
        model.optimize()

        return model.objVal, []  # TODO: sets return


class GreedyAlgorithm(MNPAlgorithm):
    """Greedy heuristics to solve multiway number partitioning problem.
    """
    def execute_algorithm(self, data: Collection):
        _data_sorted = sorted(data, reverse=True)
        _iter = _data_sorted.__iter__()
        _sums = [_iter.__next__() for _ in range(self.number_of_sets)]
        sets = [{elem} for elem in _sums]
        for elem in _iter:
            _cur_min = np.argmin(_sums)
            _sums[_cur_min] += elem
            sets[_cur_min].add(elem)
        _perfect = self.get_perfect_mnp(_data_sorted)
        return self.calculate_minsq_of(_perfect, sets), sets
