import numpy as np
from core import PartialSolution, MaxToMin, GreedyTranspOne


def test_maxtomin():
    ps = PartialSolution([3, 5, 11], 2)
    ps.put_item(0, 0)
    ps.put_item(1, 0)
    ps.put_item(2, 1)
    assert np.array_equal(ps.get_index_list(0), [0, 1])
    assert np.array_equal(ps.get_index_list(1), [2])
    MaxToMin.move(ps)
    assert np.array_equal(ps.get_index_list(0), [0, 1, 2])
    assert np.array_equal(ps.get_index_list(1), [])


def test_greedy():
    ps = PartialSolution([100], 2)
    ps.put_item(0, 1)
    assert np.array_equal(ps.get_index_list(0), [])
    assert np.array_equal(ps.get_index_list(1), [0])
    GreedyTranspOne.move(ps)
    assert np.array_equal(ps.get_index_list(0), [0])
    assert np.array_equal(ps.get_index_list(1), [])
