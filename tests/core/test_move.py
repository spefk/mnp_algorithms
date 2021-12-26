import numpy as np
from core import PartialSolution, max_to_min, greedy_transp_one


def test_maxtomin():
    ps = PartialSolution([3, 5, 11], 2)
    ps.put_item(0, 0)
    ps.put_item(1, 0)
    ps.put_item(2, 1)
    assert np.array_equal(ps.get_index_list(0), [0, 1])
    assert np.array_equal(ps.get_index_list(1), [2])
    max_to_min(ps)
    assert np.array_equal(ps.get_index_list(0), [0, 1, 2])
    assert np.array_equal(ps.get_index_list(1), [])


def test_greedy():
    ps = PartialSolution([100], 2)
    ps.put_item(0, 1)
    assert np.array_equal(ps.get_index_list(0), [])
    assert np.array_equal(ps.get_index_list(1), [0])
    greedy_transp_one(ps)
    assert np.array_equal(ps.get_index_list(0), [0])
    assert np.array_equal(ps.get_index_list(1), [])
