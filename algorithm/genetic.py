import copy
import random
from abc import ABCMeta, abstractmethod
from typing import List

from more_itertools import flatten

from core import (
    AbstractMover, PartialSolution, AbstractSolver, Instance_T,
)
from .random_solution import RandomSolver


class AbstractCrossover(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def cross(ps1: PartialSolution, ps2: PartialSolution) -> PartialSolution:
        raise NotImplementedError()


class HalfCrossover(AbstractCrossover):
    @staticmethod
    def cross(ps1: PartialSolution, ps2: PartialSolution) -> PartialSolution:
        h = ps1.n // 2
        ps_out = copy.deepcopy(ps1)
        ps_out.partitioning[h:] = ps2.partitioning[h:]
        ps_out.recalculate_sums()
        return ps_out


def _get_error(ps: PartialSolution):
    return ps.abs_error


class GeneticAlgorithm(AbstractSolver):

    def __init__(
            self,
            crossover: AbstractCrossover,
            mutation: AbstractMover,
            population_size: int = 1000,
            max_iter: int = 1000,
            mutations_n: int = 200,
            crossover_n: int = 50,
            old_livers_n: int = 25,
            elite_n: int = 100,
            selection_n: int = 100,
    ):
        assert population_size >= selection_n
        assert crossover_n + mutations_n + old_livers_n + elite_n >= selection_n
        self.crossover = crossover
        self.mutation = mutation
        self.population_size = population_size
        self.max_iter = max_iter
        self.mutations_n = mutations_n
        self.crossover_n = crossover_n
        self.old_livers_n = old_livers_n
        self.elite_n = elite_n
        self.selection_n = selection_n

    def solve(self, data: Instance_T, m: int) -> PartialSolution:
        return min(self._genetic_run(data, m, self.max_iter), key=_get_error)

    def _genetic_run(self, data, m, iter_n: int = 1, population=None):
        population: List[PartialSolution] = population or [
            RandomSolver().solve(data, m)
            for _ in range(self.population_size)
        ]
        for _ in range(iter_n):
            # Selection (K best, T worst)
            K, T = int(self.selection_n * 0.9), int(self.selection_n * 0.1)
            population.sort(key=_get_error)
            population = population[:K] + population[-T:]

            # Recombination
            mutations = [
                self.mutation.move(copy.deepcopy(random.choice(population)))
                for _ in range(self.mutations_n)
            ]
            crossovers = [
                self.crossover.cross(
                    random.choice(population),
                    random.choice(population),
                )
                for _ in range(self.crossover_n)
            ]

            # New Population
            population = population[:self.elite_n] \
                         + random.sample(population, self.old_livers_n) \
                         + mutations \
                         + crossovers

        return population


class ParallelGenetic(GeneticAlgorithm):

    def __init__(self, *args, part_to_share=0.1, experiments_n=2, **kwargs):
        super(ParallelGenetic, self).__init__(*args, **kwargs)
        self.part_to_share = part_to_share
        self.experiments_n = experiments_n

    def solve(self, data: Instance_T, m: int) -> PartialSolution:
        _populations = [self._genetic_run(data, m, 0) for _ in range(self.experiments_n)]
        for _ in range(self.max_iter):
            # mix
            all_pop = list(flatten(_populations))
            sz = len(_populations[0])
            for p in _populations:
                p.extend(random.sample(all_pop, int(sz * self.part_to_share)))
            # genetic
            _populations = [
                self._genetic_run(None, None, 1, population=p)
                for p in _populations
            ]

        return min(flatten(_populations), key=_get_error)
