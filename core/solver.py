import logging

from abc import abstractmethod, ABCMeta
from typing import Any

from .data_source import DataSource
from .algorithm import MNPAlgorithm, MNP_OF, MNP_Partition


logger = logging.getLogger(__name__)


class Solver(metaclass=ABCMeta):
    """ Base solver interface. Puts together DataSource and Algorithm objects. """
    @abstractmethod
    def solve_problem(self):
        raise NotImplementedError()


class MNPSolver(Solver):
    """General solver interface for MNP."""

    def __init__(
        self,
        source: DataSource[Any, int],
        algorithm: MNPAlgorithm,
    ):
        self.source = source
        self.algorithm = algorithm

        self._of = None
        self._partitioning = None
        self._perfect = self.algorithm.get_perfect_mnp_value(self.source.get_data())

    @property
    def perfect_value(self):
        return self._perfect

    @property
    def solution_partitioning(self):
        if self._of is None:
            raise RuntimeError("Partitioning accessed before solving!")
        return self._partitioning

    @property
    def of_value(self):
        if self._of is None:
            raise RuntimeError("OF accessed before solving!")
        return self._of

    def solve_problem(self):
        self._partitioning = self.algorithm.run(
            self.source.get_data()
        )
        self._of = MNPAlgorithm.calculate_sum_diff_sq(partitioning=self._partitioning, perfect=self._perfect)
        return self._of, self._partitioning

    def __str__(self):
        if self._of is None:
            _str = 'NOT SOLVED'
        else:
            _str = f'OF: {self._of}' \
                   + '\nSETS:\n' \
                   + '\n'.join(f'SET #{i}: {s}' for i, s in enumerate(self._partitioning)) \
                   + '\nSET SUMS: ' \
                   + ' '.join(list(map(lambda x: str(sum(x)), self._partitioning))) \
                   + '\n'
        return _str

    def __repr__(self):
        return 'MNP SOLVER INSTANCE\n' \
               + f'SOURCE: {self.source}\n'\
               + f'ALGORITHM: {self.algorithm}\n'\
               + self.__str__()
