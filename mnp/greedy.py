import numpy as np

from typing import Collection

from core import MNPAlgorithm, MNP_Partition


class GreedyAlgorithm(MNPAlgorithm):
    """Greedy heuristics to solve MNP problem.
    """
    def run(self, data: Collection):
        _data_sorted = sorted(data, reverse=True)
        _sums = [0 for _ in range(self.number_of_sets)]
        sets: MNP_Partition = [[] for _ in _sums]
        for elem in _data_sorted:
            _min_idx = np.argmin(_sums)
            _sums[_min_idx] += elem
            sets[_min_idx].append(elem)
        return sets
