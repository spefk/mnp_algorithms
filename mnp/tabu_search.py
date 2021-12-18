import copy

from typing import Collection, Callable

from core import MNPAlgorithm, MNP_Partition


def _get_list_hash_xor(lst: list) -> int:
    _xor = 0
    for e in lst:
        _xor ^= e
    return _xor


def _get_partitioning_hash(partition: MNP_Partition):
    # not universal hash
    _xor = 0
    for s in partition:
        _xor ^= _get_list_hash_xor(s)
    return _xor


class TabuAlgorithm(MNPAlgorithm):
    """ Tabu search heuristic to solve MNP problem. """

    def __init__(
            self,
            initial_partitioning: MNP_Partition,
            transition_func: Callable,
            max_iter=10**4,
            **kwargs,
    ):
        super(TabuAlgorithm, self).__init__(**kwargs)

        self.initial_partitioning = copy.deepcopy(initial_partitioning)
        self.transition_func = transition_func
        self.max_iter = max_iter

    def run(self, data: Collection):
        _tabu_set = {}

