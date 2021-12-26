import pytest
import logging

from core import PythonSource
from algorithm.greedy import GreedySolver


logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    'instance,expected',
    [
        ([1, 2], [[2], [1]]),
        ([100, 1, 3, 2], [[100], [1, 3, 2]]),
        ([8, 7, 6, 5], [[8, 5], [7, 6]]),
    ]
)
def test_greedy_trivial(instance, expected):
    ext_s_one = PythonSource(
        data=instance,
        mapper=int,
        gatherer=list,
    )

    solution = GreedySolver().solve(ext_s_one.get_data(), 2).solution
    assert solution == expected

