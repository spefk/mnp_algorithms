import pytest
import logging

from more_itertools import flatten
from core.algorithm import MNPAlgorithm


logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "sets_n,sets,expected",
    [
        (1, [[0]], 0),
        (1, [[10_000]], 0),
        (2, [[0], [4]], 2 * 2 ** 2),
        (3, [[1, 3], [1, 1, 1, 1], [4]], 0),
        (3, [[12], [0], [0]], 2 * 4 ** 2 + 8 ** 2),
    ]
)
def test_mnp(sets_n, sets, expected):
    class Dummy(MNPAlgorithm):
        def run(self, data):
            pass

    alg = Dummy(sets_n)
    assert expected == alg.calculate_sum_diff_sq(sets, alg.get_perfect_mnp_value(flatten(sets)))
