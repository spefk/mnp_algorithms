
from abc import abstractmethod, ABCMeta
from typing import Iterable


class Algorithm(metaclass=ABCMeta):
    """Base class for algorithms.
    """
    def run(self, data: Iterable):
        _of, _solution = self.execute_algorithm(data)
        return _of, _solution

    @abstractmethod
    def execute_algorithm(self, data: Iterable):
        raise NotImplementedError()


class MNPAlgorithm(Algorithm, metaclass=ABCMeta):
    """Base class for all Multiway Number Partitioning"""
    def __init__(self, number_of_sets: int = 1, **kwargs):
        super(MNPAlgorithm, self).__init__()
        self.number_of_sets = number_of_sets
        self.settings = dict(**kwargs)

    def get_perfect_mnp(self, data):
        return float(sum(data)) / self.number_of_sets

    @staticmethod
    def calculate_sum_diff_sq(state=None, perfect=None):
        return sum((sum(s) - perfect) ** 2 for s in state)
