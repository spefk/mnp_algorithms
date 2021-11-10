import logging

from abc import abstractmethod, ABCMeta
from typing import Collection, Generic, TypeVar, List


logger = logging.getLogger(__name__)


V = TypeVar('V')
A = TypeVar('A')


class Algorithm(Generic[V, A], metaclass=ABCMeta):
    """ Base interface for algorithms. """

    @abstractmethod
    def run(self, data: Collection[V]) -> A:
        raise NotImplementedError()


MNP_Number = int
MNP_OF = float
MNP_Partition = List[List[int]]
MNP_Answer = MNP_Partition


class MNPAlgorithm(Algorithm[MNP_Number, MNP_Answer], metaclass=ABCMeta):
    """ Base class for Multiway Number Partitioning algorithms. """
    def __init__(self, number_of_sets: int = 1, **kwargs):
        super(MNPAlgorithm, self).__init__()
        self.number_of_sets = number_of_sets
        self.settings = kwargs

    def get_perfect_mnp_value(self, data) -> MNP_OF:
        return sum(data) / self.number_of_sets

    @staticmethod
    def calculate_sum_diff_sq(perfect: MNP_OF, partitioning: MNP_Partition) -> MNP_OF:
        return sum((sum(s) - perfect) ** 2 for s in partitioning)
