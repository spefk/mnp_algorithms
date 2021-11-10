import pytest
import logging

from core import PythonSource, MNPSolver
from mnp import GreedyAlgorithm


logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    'instance,expected',
    [
        ([1, 2], [[2], [1]]),
        ([100, 1, 2, 3], [[100], [3, 2, 1]]),
        ([8, 7, 6, 5], [[8, 5], [7, 6]]),
    ]
)
def test_greedy_trivial(instance, expected):
    ext_s_one = PythonSource(
        data=instance,
        item_type=float,
        to_collection=list,
    )

    solver = MNPSolver(source=ext_s_one, algorithm=GreedyAlgorithm(number_of_sets=2))
    _sets = solver.solve_problem()
    assert _sets == expected

