from collections import defaultdict

import numpy as np

from .random_solution import RandomSolver
from core import AbstractSolver, Instance_T, PartialSolution


class ACO(AbstractSolver):
    """ Ant Colony Optimizaiton algorithm """

    def __init__(
        self,
        ants_n: int = 10,
        max_iter: int = 100,
        growth_rate: float = 50.,
        residual_rate: float = 0.7,
        threshold: float = 0.01,
    ):
        self.ants_n = ants_n
        self.max_iter = max_iter
        self.pheromones = None
        self.growth_r = growth_rate
        self.residual_r = residual_rate
        self.threshold = threshold

    def _reset_pheromones(self):
        self.pheromones = defaultdict(ACO._default_pheromone)

    @staticmethod
    def _default_pheromone():
        return 0.1

    def solve(self, data: Instance_T, m: int) -> PartialSolution:
        self._reset_pheromones()
        best_ps = RandomSolver().solve(data, m)
        for _ in range(self.max_iter):

            # run N Ants, each returns some solution
            ants_result = [self._one_iteration(data, m) for _ in range(self.ants_n)]
            # update all pheromones (evaporation)
            self.pheromones = defaultdict(
                ACO._default_pheromone,
                (
                    (k, v * self.residual_r)
                    for k, v in self.pheromones.items()
                    if v * self.residual_r >= self.threshold
                )
            )
            # update pheromones according to solutions
            for ps in ants_result:
                _of = ps.abs_error
                _sums = np.zeros(ps.m)
                for i, x in enumerate(data):
                    self.pheromones[(ps.get_sum_hash(_sums), i, ps.find_item(i))] += self.growth_r / _of
                    _sums[ps.find_item(i)] += ps.cost[i]
                if ps.abs_error < best_ps.abs_error:
                    best_ps = ps

        return best_ps

    def _one_iteration(self, data: Instance_T, m: int) -> PartialSolution:
        ps = PartialSolution(data, m=m)
        for i, x in enumerate(data):
            # choose where to put i by pheromones
            phero = np.array([self.pheromones[(ps.sums_hash, i, j)] for j in range(m)])
            j = np.random.choice(
                list(range(m)),
                1,
                p=(phero / sum(phero)),
            )[0]
            ps.put_item(i, j)
        return ps
